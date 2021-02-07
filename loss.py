import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        # pred shape = [2, 4, 128, 128, 128], target shape = [2, 128, 128, 128]
        eps = 0.0001
        encoded_target = pred.data.clone().zero_()
        # unsqueeze(1) 在位置1上增加一个维度
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersect = pred * encoded_target
        union = pred + encoded_target
        # intersect.sum(0).sum(1).sum(1).sum(1): [24333]->[4333]->[433]->[43]->[4]
        numerator = 2 * intersect.sum(0).sum(1).sum(1).sum(1)
        denominator = union.sum(0).sum(1).sum(1).sum(1) + eps

        dice_val_per_channel = numerator / denominator
        dice_loss_per_channel = 1 - dice_val_per_channel
        weighted_dice_loss = dice_loss_per_channel * torch.tensor([0, 1, 1, 1], dtype=torch.double).cuda()
        dice_loss = weighted_dice_loss.sum()/3

        f = open("./0111-701-800.txt", 'a')
        print('{:.2f} {:.2f} {:.2f} {:.2f} weighted:{:.2f}'.format(
            dice_loss_per_channel[0].data,
            dice_loss_per_channel[1].data,
            dice_loss_per_channel[2].data,
            dice_loss_per_channel[3].data,
            dice_loss.data), file=f)
        return dice_loss
