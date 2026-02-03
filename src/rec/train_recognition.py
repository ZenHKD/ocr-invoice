import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Imports
from model.rec.parseq import ParSeq
from model.rec.vocab import VOCAB
from src.rec.rec_dataset import RecDataset

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

def levenshtein_distance(s1, s2):
    """
    Computes Levenshtein distance between two strings/sequences.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_metrics(preds_str, targets_str):
    """
    Calculate Sequence Accuracy (Exact Match) and CER.
    """
    total_dist = 0
    total_len = 0
    correct_seqs = 0
    
    for pred, target in zip(preds_str, targets_str):
        # Exact Match (Sequence)
        if pred == target:
            correct_seqs += 1
            
        # CER
        dist = levenshtein_distance(pred, target)
        total_dist += dist
        total_len += len(target)
        
    seq_acc = correct_seqs / len(targets_str)
    cer = total_dist / max(1, total_len)
    
    return seq_acc, cer

def train_one_epoch(model, dataloader, criterion, optimizer, device, pad_id):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward (Teacher Forcing)
        logits = model(images, target=targets) 
        
        loss_target = targets[:, 1:] 
        if logits.size(1) != loss_target.size(1):
             min_len = min(logits.size(1), loss_target.size(1))
             logits = logits[:, :min_len, :]
             loss_target = loss_target[:, :min_len]

        loss = criterion(logits.flatten(0, 1), loss_target.flatten())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    total_seq_acc = 0
    total_cer = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            raw_texts = batch['text'] # Use raw text for Ground Truth
            
            # 1. Loss (Teacher Forcing)
            logits_tf = model(images, target=targets)
            loss_target = targets[:, 1:]
            
            logits_tf_trimmed = logits_tf
            if logits_tf.size(1) != loss_target.size(1):
                 min_len = min(logits_tf.size(1), loss_target.size(1))
                 logits_tf_trimmed = logits_tf[:, :min_len, :]
                 loss_target_trimmed = loss_target[:, :min_len]
            else:
                loss_target_trimmed = loss_target
                
            loss = criterion(logits_tf_trimmed.flatten(0, 1), loss_target_trimmed.flatten())
            total_loss += loss.item()
            
            # 2. Inference (Greedy Autoregressive)
            logits_inf = model(images, target=None) 
            probs = logits_inf.softmax(-1)
            preds_ids = probs.argmax(-1)
            
            # Decode
            preds_str = tokenizer.decode(preds_ids)
            
            # Metrics
            seq_acc, cer = calculate_metrics(preds_str, raw_texts)
            
            total_seq_acc += seq_acc
            total_cer += cer
            
            pbar.set_postfix({'loss': loss.item(), 'acc': total_seq_acc / (pbar.n + 1)})
            
    n = len(dataloader)
    return total_loss / n, total_seq_acc / n, total_cer / n

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets
    train_dataset = RecDataset(args['train_dir'], image_size=(128, 32), max_len=args['max_len'], is_train=True)
    val_dataset = RecDataset(args['val_dir'], image_size=(128, 32), max_len=args['max_len'], is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    
    # 2. Model
    model = ParSeq(
        img_size=(32, 128), 
        patch_size=(4, 8), 
        embed_dim=768, 
        enc_depth=12, 
        num_heads=12, 
        charset=VOCAB,
        max_len=args['max_len']
    )
    model = model.to(device)
    
    # 3. Loss & Optimizer
    pad_id = train_dataset.tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    best_val_seq_acc = 0.0
    
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
        
    log_file_path = os.path.join(args['save_dir'], 'training_log.csv')
    with open(log_file_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_seq_acc,val_cer\n')
    
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch+1}/{args['epochs']}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, pad_id)
        v_loss, v_acc, v_cer = validate(model, val_loader, criterion, device, val_dataset.tokenizer)
        
        print(f"Train Loss: {t_loss:.4f}")
        print(f"Val Loss: {v_loss:.4f} | Seq Acc: {v_acc:.4f} | CER: {v_cer:.4f}")
        
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1},{t_loss:.4f},{v_loss:.4f},{v_acc:.4f},{v_cer:.4f}\n")
        
        scheduler.step(v_loss)
        
        # Save checkpoints
        if v_acc > best_val_seq_acc: 
            best_val_seq_acc = v_acc
            save_path = os.path.join(args['save_dir'], 'best_model_rec_acc.pth')
            save_checkpoint(model, optimizer, epoch, {'acc': best_val_seq_acc, 'cer': v_cer, 'loss': v_loss}, save_path)
            print(f"Saved best model (Seq Acc) to {save_path}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_path = os.path.join(args['save_dir'], 'best_model_rec_loss.pth')
            save_checkpoint(model, optimizer, epoch, {'acc': v_acc, 'cer': v_cer, 'loss': best_val_loss}, save_path)
            
        save_path = os.path.join(args['save_dir'], 'last_model_rec.pth')
        save_checkpoint(model, optimizer, epoch, {'acc': v_acc}, save_path)

if __name__ == '__main__':
    # Default Config
    config = {
        'train_dir': 'data/train_crop',
        'val_dir': 'data/val_crop',
        'save_dir': 'best_model/rec',
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-4,
        'max_len': 64
    }
    
    main(config)
