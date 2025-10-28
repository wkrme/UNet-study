import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):

        loss = self.bce(inputs, targets)

        return loss


class BCEwithDice_Loss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        BCE_LOSS = self.BCE(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        DICE_LOSS = 1 - dice.mean()
        final_loss = BCE_LOSS + DICE_LOSS
    
        return final_loss, BCE_LOSS.detach(), DICE_LOSS.detach()