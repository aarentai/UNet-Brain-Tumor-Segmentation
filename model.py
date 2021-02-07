import torch
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.functional as F
# from torch.utils import model_zoo
# from torchvision import models


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-channel as input
        self.conv4_8 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv8_8 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv8_16 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv16_16 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv16_32 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv32_32 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv32_64 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv64_64 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv64_32 = nn.Conv3d(64, 32, 3, padding=1)
        self.conv32_16 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv16_8 = nn.Conv3d(16, 8, 3, padding=1)
        self.conv8_4 = nn.Conv3d(8, 4, 3, padding=1)
        # https://zhuanlan.zhihu.com/p/32506912
        self.up_conv64_32 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.up_conv32_16 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.up_conv16_8 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)

    def forward(self, input):
        input = input.type('torch.cuda.DoubleTensor')
        enc1 = self.conv4_8(input)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()
        enc1 = self.conv8_8(enc1)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()

        enc2 = F.max_pool3d(enc1, 2, 2)
        enc2 = self.conv8_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()
        enc2 = self.conv16_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()

        enc3 = F.max_pool3d(enc2, 2, 2)
        enc3 = self.conv16_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()
        enc3 = self.conv32_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()

        btm = F.max_pool3d(enc3, 2, 2)
        btm = self.conv32_64(btm)
        btm = self.conv64_64(btm)

        dec3 = self.up_conv64_32(btm)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.conv64_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()
        dec3 = self.conv32_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()

        dec2 = self.up_conv32_16(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.conv32_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()
        dec2 = self.conv16_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()

        dec1 = self.up_conv16_8(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.conv16_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()
        dec1 = self.conv8_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()

        output = self.conv8_4(dec1)
        output = F.softmax(output, dim=1)

        return output
