import torch.nn as nn

class PARSeqLoss(nn.Module):
    def __init__(self, ignore_index, smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=smoothing)

    def forward(self, pred, target):
        # pred: (B, T, num_classes)
        # target: (B, T)
        return self.loss_fn(pred.flatten(0, 1), target.flatten())
