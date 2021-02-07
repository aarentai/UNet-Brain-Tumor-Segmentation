from __future__ import print_function, division
import os
import nibabel as nib
from torch.utils.data import Dataset
import warnings
import numpy as np
import torch
import csv

# Ignore warnings
warnings.filterwarnings("ignore")


def intensity_mean(original):
    original = original.flatten()
    new_ori = np.delete(original, np.where(original == 0))
    return np.mean(new_ori)


def intensity_std(original):
    original = original.flatten()
    new_ori = np.delete(original, np.where(original == 0))
    return np.std(new_ori)


def intensity_norm(original, mean, std):
    normalized = (original - mean)/std
    normalized[normalized > 5] = 5
    # normalized[normalized < -5] = -5
    # normalized = (normalized + 5)/(2*5)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return normalized


class BraTS_dataset(Dataset):
    def __init__(self, root_dir, img_folder, label_folder, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_folder)
        self.label_dir = os.path.join(root_dir, label_folder)
        self.folder_name = [f for f in os.listdir(self.img_dir)]
        self.folder_name.sort()
        self.transform = transform

    def __len__(self):
        return len(self.folder_name)

    def __getitem__(self, index):
        folder = self.folder_name[index]
        t1_path = self.img_dir + folder + '/' + folder + '_t1.nii.gz'
        # occ_path = self.img_dir + folder + '/'  + folder + '_occ.nii.gz'
        t1ce_path = self.img_dir + folder + '/' + folder + '_t1ce.nii.gz'
        t2_path = self.img_dir + folder + '/' + folder + '_t2.nii.gz'
        flair_path = self.img_dir + folder + '/' + folder + '_flair.nii.gz'
        label_path = self.label_dir + folder + '_seg.nii.gz'

        # load 4-channel input
        t1 = nib.load(t1_path).get_data()
        # occ = nib.load(occ_path).get_data()
        t1ce = nib.load(t1ce_path).get_data()
        t2 = nib.load(t2_path).get_data()
        flair = nib.load(flair_path).get_data()

        # calculate mean and std
        mean = [intensity_mean(t1), intensity_mean(t1ce), intensity_mean(t2), intensity_mean(flair)]
        # mean = [intensity_mean(occ), intensity_mean(t1ce), intensity_mean(t2), intensity_mean(flair)]
        std = [intensity_std(t1), intensity_std(t1ce), intensity_std(t2), intensity_std(flair)]
        # std = [intensity_std(occ), intensity_std(t1ce), intensity_std(t2), intensity_std(flair)]

        # normalization
        t1 = intensity_norm(t1, mean[0], std[0])
        # occ = intensity_norm(occ, mean[0], std[0])
        t1ce = intensity_norm(t1ce, mean[1], std[1])
        t2 = intensity_norm(t2, mean[2], std[2])
        flair = intensity_norm(flair, mean[3], std[3])

        # crop
        t1 = t1[56:56 + 128, 56:56 + 128, 14:14 + 128]
        # occ = occ[56:56 + 128, 56:56 + 128, 14:14 + 128]
        t1ce = t1ce[56:56 + 128, 56:56 + 128, 14:14 + 128]
        t2 = t2[56:56 + 128, 56:56 + 128, 14:14 + 128]
        flair = flair[56:56 + 128, 56:56 + 128, 14:14 + 128]

        image = np.stack([t1, t1ce, t2, flair])
        # image = np.stack([occ, t1ce, t2, flair])

        label = nib.load(label_path).get_data()
        label = label[56:56 + 128, 56:56 + 128, 14:14 + 128]
        label[label > 3] = 3

        sample = {'name': folder,
                  'image': torch.from_numpy(image).type('torch.DoubleTensor'),
                  'label': torch.from_numpy(label).type('torch.DoubleTensor')}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    mean_file = open("mean.csv", "w")
    mean_writer = csv.writer(mean_file)
    mean_writer.writerow(['name', 't1', 't1ce', 't2', 'flair'])

    std_file = open("std.csv", "w")
    std_writer = csv.writer(std_file)
    std_writer.writerow(['name', 't1', 't1ce', 't2', 'flair'])

    t1_50largest_file = open("t1_50largest.csv", "w")
    t1_50largest_writer = csv.writer(t1_50largest_file)
    t1_50largest_writer.writerow(['name'])

    t1ce_50largest_file = open("t1ce_50largest.csv", "w")
    t1ce_50largest_writer = csv.writer(t1ce_50largest_file)
    t1ce_50largest_writer.writerow(['name'])

    t2_50largest_file = open("t2_50largest.csv", "w")
    t2_50largest_writer = csv.writer(t2_50largest_file)
    t2_50largest_writer.writerow(['name'])

    flair_50largest_file = open("flair_50largest.csv", "w")
    flair_50largest_writer = csv.writer(flair_50largest_file)
    flair_50largest_writer.writerow(['name'])
    
    img_dir = 'data/images/'
    folder_name = [f for f in os.listdir(img_dir)]
    for index in range(len(folder_name)):
        folder = folder_name[index]
        print(index, folder)
        t1_path = img_dir + folder + '/' + folder + '_t1.nii.gz'
        t1ce_path = img_dir + folder + '/' + folder + '_t1ce.nii.gz'
        t2_path = img_dir + folder + '/' + folder + '_t2.nii.gz'
        flair_path = img_dir + folder + '/' + folder + '_flair.nii.gz'

        # load 4-channel input
        t1 = nib.load(t1_path).get_data()
        t1ce = nib.load(t1ce_path).get_data()
        t2 = nib.load(t2_path).get_data()
        flair = nib.load(flair_path).get_data()

        # calculate mean and std
        mean = [intensity_mean(t1), intensity_mean(t1ce), intensity_mean(t2), intensity_mean(flair)]
        std = [intensity_std(t1), intensity_std(t1ce), intensity_std(t2), intensity_std(flair)]

        # Normalization
        t1 = intensity_norm(t1, mean[0], std[0])
        t1ce = intensity_norm(t1ce, mean[1], std[1])
        t2 = intensity_norm(t2, mean[2], std[2])
        flair = intensity_norm(flair, mean[3], std[3])

        mean_writer.writerow([folder, mean[0], mean[1], mean[2], mean[3]])
        std_writer.writerow([folder, std[0], std[1], std[2], std[3]])
        t1_50largest_writer.writerow([folder, [i for i in np.sort(t1, axis=None)[-50:]]])
        t1ce_50largest_writer.writerow([folder, [i for i in np.sort(t1ce, axis=None)[-50:]]])
        t2_50largest_writer.writerow([folder, [i for i in np.sort(t2, axis=None)[-50:]]])
        flair_50largest_writer.writerow([folder, [i for i in np.sort(flair, axis=None)[-50:]]])
        

    mean_file.close()
    std_file.close()
    t1_50largest_file.close()
    t2_50largest_file.close()
    t1ce_50largest_file.close()
    flair_50largest_file.close()

