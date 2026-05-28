import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary focal loss for classification.

    Args:
        alpha (float): weight for the positive class (0 < alpha < 1)
        gamma (float): focusing parameter (gamma >= 0)
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        """
        # alpha > 0.5 positives get more weight
        # alpha < 0.5 negatives get more weight

        # alpha_t weight of target label
        # p_t probability of target label

        p = torch.sigmoid(logits)
        targets = targets.float()

        # Compute focal loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        # (target 0) If p = 1.0, loss = 0.7 * 0.0 ** 2 * bce_loss
        # (target 1) If p = 1.0, loss = 0.3 * 0.0 ** 2 * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
