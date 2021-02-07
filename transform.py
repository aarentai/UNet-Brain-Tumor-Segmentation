from __future__ import print_function, division
import torch
import numpy as np
from skimage import io, transform


class Rescale(object):
    """按照给定尺寸更改一个图像的尺寸
    Args:
        output_size (tuple or int): 要求输出的尺寸.  如果是个元组类型, 输出
        和output_size匹配. 如果时int类型,图片的短边和output_size匹配, 图片的
        长宽比保持不变.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        transformed_image = transform.resize(image, (new_h, new_w))
        transformed_label = transform.resize(label, (new_h, new_w))

        return {'image': transformed_image, 'label': transformed_label}


class RandomCrop(object):
    """随机裁剪图片
    Args:
        output_size (tuple or int): 期望输出的尺寸, 如果时int类型, 裁切成正方形.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, z = 56, 56, 14
        image = image[:, :, x: x + self.output_size, y: y + self.output_size, z: z + self.output_size]
        label = label[:, :, x: x + self.output_size, y: y + self.output_size, z: z + self.output_size]

        return {'image': image, 'label': label}


class ToTensor(object):
    """将ndarrays的样本转化为Tensors的样本"""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 交换颜色通道, 因为
        # numpy: H x W x C
        # torch: C x H x W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}