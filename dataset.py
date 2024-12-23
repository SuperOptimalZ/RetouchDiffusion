import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as f


class MyDataset(Dataset):
    def __init__(self, in_data, ref_data):

        self.in_data_list = in_data
        self.ref_data_list = ref_data
        # 将应用于输入图像（in_data）的一系列转换，包括将图像大小调整为（128，128）像素并转换为张量。
        self.in_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        #一个调整图像亮度的转换
        self.bri_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=(1,2))
        ])

    def __getitem__(self, index):
        randindex = random.randint(0,9)
        in_image_name = self.in_data_list[index + randindex]
        ref_image_name = self.ref_data_list[index]
        in_fp = open(in_image_name, 'rb')
        in_image = Image.open(in_fp)
        ref_fp = open(ref_image_name, 'rb')
        ref_image = Image.open(ref_fp)

        in_tensor = self.in_transforms(in_image)
        ref_tensor = self.in_transforms(ref_image)
        
        in_fp.close()
        ref_fp.close()
        return in_tensor, ref_tensor

    def __len__(self):
        return len(self.ref_data_list)
