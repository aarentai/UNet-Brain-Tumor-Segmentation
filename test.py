import torch
from torch.utils.data import DataLoader
from dataset import BraTS_dataset
from transform import ToTensor
import numpy as np
import nibabel as nib
import csv
import matplotlib.pyplot as plt

num_epochs = 30
num_workers = 4  
batch_size = 1  


def merge(output):
    # output.shape = [1, 4, 128, 128, 128]
    output = output.cpu().detach().numpy()
    mask = np.zeros([128, 128, 128])
    for i in range(output.shape[2]):
        for j in range(output.shape[3]):
            for k in range(output.shape[4]):
                if output[0, 0, i, j, k] == max(output[0, :, i, j, k]):
                    mask[i, j, k] = 0
                elif output[0, 1, i, j, k] == max(output[0, :, i, j, k]):
                    mask[i, j, k] = 1
                elif output[0, 2, i, j, k] == max(output[0, :, i, j, k]):
                    mask[i, j, k] = 2
                else:
                    mask[i, j, k] = 3

    return mask


def intersection(x, y, target):
    counter = 0
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if x[i, j, k] == y[i, j, k] and x[i, j, k] == target:
                    counter = counter + 1

    return counter


# 1+2+3
def intersection_wt(x, y):
    counter = 0
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if bool(y[i, j, k]) and bool(x[i, j, k]):
                    counter = counter + 1

    return counter


# 1+3
def intersection_tc(x, y):
    counter = 0
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if (y[i, j, k] != 0 and y[i, j, k] != 2) and (x[i, j, k] != 0 and x[i, j, k] != 2):
                    counter = counter + 1

    return counter


def save_nii(pred, gt, name):
    pred = nib.Nifti1Image(pred, None)
    # gt = nib.Nifti1Image(gt, None)
    nib.save(pred, 'result/' + name + '_pred.nii')
    # nib.save(gt, 'compare/groundtruth/' + name+ '_groundtruth.nii')


def save_nii_NCR_NET(pred, gt, name):
    pred[pred != 1] = 0
    pred = nib.Nifti1Image(pred, None)
    gt[gt != 1] = 0
    gt = nib.Nifti1Image(gt, None)
    nib.save(pred, 'compare/NCR_NET_40/' + name + '_prediction.nii')
    nib.save(gt, 'compare/NCR_NET_40/' + name + '_gt.nii')


def save_dice_csv(pred, gt, name, writer):
    dice_0 = intersection(pred, gt, 0) * 2.0 / (np.count_nonzero(pred == 0) + np.count_nonzero(gt == 0))
    # print(intersection(pred, gt, 0) * 2.0, '/', (np.count_nonzero(pred == 0) + np.count_nonzero(gt == 0)))
    dice_1 = intersection(pred, gt, 1) * 2.0 / (np.count_nonzero(pred == 1) + np.count_nonzero(gt == 1))
    # print(intersection(pred, gt, 1) * 2.0, '/', (np.count_nonzero(pred == 1) + np.count_nonzero(gt == 1)))
    dice_2 = intersection(pred, gt, 2) * 2.0 / (np.count_nonzero(pred == 2) + np.count_nonzero(gt == 2))
    # print(intersection(pred, gt, 2) * 2.0, '/', (np.count_nonzero(pred == 2) + np.count_nonzero(gt == 2)))
    # dice_2 = intersection_wt(pred, gt) * 2.0 / (np.count_nonzero(pred) + np.count_nonzero(gt))
    # print(intersection_wt(pred, gt) * 2.0, '/', (np.count_nonzero(pred) + np.count_nonzero(gt)))
    dice_3 = intersection(pred, gt, 3) * 2.0 / (np.count_nonzero(pred == 3) + np.count_nonzero(gt == 3))
    # print(intersection(pred, gt, 3) * 2.0, '/', (np.count_nonzero(pred == 3) + np.count_nonzero(gt == 3)))
    print(name, dice_0, dice_1, dice_2, dice_3)
    writer.writerow([name, dice_0, dice_1, dice_2, dice_3])


def save_dice_ettcwt(pred, gt, name, writer):
    et = intersection(pred, gt, 3) * 2.0 / (np.count_nonzero(pred == 3) + np.count_nonzero(gt == 3))
    wt = intersection_wt(pred, gt) * 2.0 / (np.count_nonzero(pred != 0) + np.count_nonzero(gt != 0))
    tc = intersection_tc(pred, gt) * 2.0 / (np.count_nonzero(pred == 1) + np.count_nonzero(pred == 3)
                                            + np.count_nonzero(gt == 1) + np.count_nonzero(gt == 3))
    print(name, et, wt, tc)
    writer.writerow([name, et, wt, tc])


# def show_slice(input, output, gt):
#     slice = 60
#     plt.figure("compare")
#     plt.subplot(3, 4, 1)
#     plt.imshow(input.cpu().detach().numpy()[0, 0, slice, :, :])
#     plt.subplot(3, 4, 2)
#     plt.imshow(input.cpu().detach().numpy()[0, 1, slice, :, :])
#     plt.subplot(3, 4, 3)
#     plt.imshow(input.cpu().detach().numpy()[0, 2, slice, :, :])
#     plt.subplot(3, 4, 4)
#     plt.imshow(input.cpu().detach().numpy()[0, 3, slice, :, :])
#     plt.subplot(3, 1, 2)
#     plt.imshow(output[slice, :, :])
#     plt.subplot(3, 1, 3)
#     plt.imshow(gt[slice, :, :])
#     plt.show()


def ceshi(model):
    model_path = "epoch_693_checkpoint.pth.tar"

    csvFile = open("accuracy_0111_ettcwt_in.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerow(['name', 'ET', 'WT', 'TC'])

    dataset = BraTS_dataset('data/', 'images/', 'labels/', transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for i, sample in enumerate(dataloader):
        name = sample['name']
        image = sample['image']
        label = sample['label']

        print(image.shape)
        output = model(image)
        pred = merge(output)
        gt = label[0, :, :, :].cpu().detach().numpy()

        save_nii(pred, gt, name[0])
        # save_nii_NCR_NET(pred, gt, name[0])
        # save_dice_csv(pred, gt, name[0], writer)
        save_dice_ettcwt(pred, gt, name[0], writer)
        print(i)

    csvFile.close()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.DoubleTensor')
    model = torch.load('MyUNet_710_in.pkl')
    ceshi(model)
