import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Enable TF32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Imports from local project
from model.det.dbnet import DBNetPP
from model.det.loss import DBLoss
from src.det_dataset import DBDataset

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

def calculate_metrics(pred, gt, mask):
    """
    Calculate simple pixel-level metrics for validation.
    pred_binary: (B, 1, H, W)
    gt: (B, 1, H, W)
    mask: (B, 1, H, W)
    """
    # Simple binary segmentation metrics
    pred_mask = (pred > 0.3).float()
    
    intersection = (pred_mask * gt * mask).sum()
    union = (pred_mask * mask).sum() + (gt * mask).sum() - intersection
    
    iou = intersection / (union + 1e-6)
    
    # Precision / Recall
    tp = intersection
    fp = (pred_mask * mask).sum() - tp
    fn = (gt * mask).sum() - tp
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {'iou': iou.item(), 'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()}

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_l_prob = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        image = batch['image'].to(device)
        batch_gpu = {
            'gt': batch['gt'].to(device),
            'mask': batch['mask'].to(device),
            'thresh_map': batch['thresh_map'].to(device),
            'thresh_mask': batch['thresh_mask'].to(device)
        }
        
        optimizer.zero_grad()

        preds = model(image)
        loss, loss_dict = criterion(preds, batch_gpu)
        
        loss.backward()
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        total_l_prob += loss_dict['l_prob'].mean().item()
        
        pbar.set_postfix({'loss': loss.item(), 'prob_loss': loss_dict['l_prob'].mean().item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            image = batch['image'].to(device)
            batch_gpu = {
                'gt': batch['gt'].to(device),
                'mask': batch['mask'].to(device),
                'thresh_map': batch['thresh_map'].to(device),
                'thresh_mask': batch['thresh_mask'].to(device)
            }
            
            preds = model(image)
            loss, _ = criterion(preds, batch_gpu)
            total_loss += loss.item()
            
            # Calculate pixel metrics on 'binary' map
            batch_metrics = calculate_metrics(preds['binary'].float(), batch_gpu['gt'], batch_gpu['mask'])
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
                
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Create Datasets
    train_dataset = DBDataset(args['train_dir'], image_size=1024, is_train=True)
    val_dataset = DBDataset(args['val_dir'], image_size=1024, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    
    # 2. Model
    model = DBNetPP(backbone=args['backbone'], pretrained=True)
    model = model.to(device)
    
    # 3. Loss & Optimizer
    criterion = DBLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    # Initialize log file
    log_file_path = os.path.join(args['save_dir'], 'training_log.csv')
    with open(log_file_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,precision,recall,f1,iou\n')
    
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch+1}/{args['epochs']}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Log to file
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},{val_metrics['f1']:.4f},{val_metrics['iou']:.4f}\n")
        
        scheduler.step(val_loss)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            save_path = os.path.join(args['save_dir'], 'best_model_detection.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            print(f"Saved best model to {save_path}")
            
        # Save Last Model
        save_path = os.path.join(args['save_dir'], 'last_model.pth')
        save_checkpoint(model, optimizer, epoch, val_metrics, save_path)

if __name__ == '__main__':
    # Default Args for running directly
    # Can be overridden by notebook
    config = {
        'train_dir': '../../data/train',
        'val_dir': '../../data/val',
        'save_dir': '../../best_model/det',
        'backbone': 'resnet18',
        'batch_size': 8,
        'epochs': 20,
        'lr': 1e-3
    }

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
        
    main(config)