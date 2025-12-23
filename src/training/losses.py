""""""""""""""""""""""""""""""""""
Focal Loss Implementation
Focal Loss를 구현한 코드입니다.
바로 Train에 적용하기보단, 따로 적용하여 끼워넣는게 좋을 것 같아 따로 작성하였습니다.

""""""""""""""""""""""""""""""""""


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs : logits
        # targets : labels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss