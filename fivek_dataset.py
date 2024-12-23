import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
#from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp

import torchvision
from torchvision import transforms
import torch
from PIL import Image
import io
import random
import webdataset as wds  # pylint: disable=import-outside-toplevel
import copy
import glob
from torch.utils.data import Dataset
import pandas as pd
from PE_Net import PE_Net
from torchvision.transforms import ToPILImage


def create_webdataset(data_dir, image_size=[512,512], random_flip=True):
    return ImageDataset(data_dir, image_size, random_flip)

class RandomNoise(object):
    def __init__(self, max_poisson=0.8, max_gaussian=0.4):
        self.max_poisson = max_poisson
        self.max_gaussian = max_gaussian

    def __call__(self, image_tensor):

        # Poisson Noise
        noise_poisson = torch.poisson(image_tensor) 
        noise_poisson_sign = torch.randint(low=0, high=2, size=image_tensor.shape) * 2 - 1
        noise_poisson = noise_poisson * noise_poisson_sign * random.uniform(0, self.max_poisson)

        # Gaussian Noise
        noise_gaussian = torch.randn(image_tensor.shape)
        noise_gaussian = noise_gaussian * random.uniform(0, self.max_gaussian)

        image_noise_tensor = image_tensor + noise_gaussian + noise_poisson 
        return torch.clamp(image_noise_tensor, 0, 1)


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size=[256,256], random_flip=True):

        self.hint_net = PE_Net()
        pe_net_weights = torch.load('savemodel/pe_net.pt', map_location='cpu')  
        self.hint_net.load_state_dict(pe_net_weights)
        self.dir = data_dir
        input_dir = os.path.join(data_dir, 'input')
        self.image_input_list = glob.glob(input_dir)
        print("Found", len(self.image_input_list), "image dirs")
        assert len(self.image_input_list) > 0

        target_dir = os.path.join(data_dir, 'target')
        self.image_target_list = glob.glob(target_dir)
        print("Found", len(self.image_target_list), "image dirs")
        assert len(self.image_target_list) > 0

        input_names = sorted(os.listdir(input_dir))
        target_names = sorted(os.listdir(target_dir))

        self.pairs = []
        for i in range(len(input_names)):
            input_img_name = os.path.join('input', input_names[i])
            target_img_name = os.path.join('target', target_names[(i) % len(target_names)])  # 错位配对
            self.pairs.append((input_img_name, target_img_name))  

        if random_flip:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size[0]*3//2),
                transforms.RandomCrop((image_size[0],image_size[1])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),]
            )
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size[0]*3//2),
                transforms.RandomCrop((image_size[0],image_size[1])),
                transforms.ToTensor()]
            )
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.apply_noise = torchvision.transforms.RandomApply([RandomNoise()], p=0.5)

    def __len__(self):
        return len(self.pairs)

    def process(self, idx):
        # print("preprocess!!!")
        output = {}
        # 获取输入和目标图像的路径
        input_name, target_name = self.pairs[idx]
        to_tensor = transforms.ToTensor()

        # 加载输入和目标图像
        low_img = Image.open(os.path.join(self.dir, input_name)).convert('RGB')
        high_img = Image.open(os.path.join(self.dir, target_name)).convert('RGB')
        low_img_tensor = self.image_transform(low_img)  
        high_img_tensor = self.image_transform(high_img)

        with torch.no_grad():
            hint_img = self.hint_net(low_img_tensor.unsqueeze(0), high_img_tensor.unsqueeze(0))[0]

        output["jpg"] = self.normalize(copy.deepcopy(low_img_tensor))
        output["hint"] = self.apply_noise(hint_img)
        # output["hint"] = self.normalize(copy.deepcopy(low_img_tensor))
        # output["jpg"] = self.apply_noise(hint_img)
        # output["ref_jpg"] = self.normalize(copy.deepcopy(high_img))
        # output["ref_hint"] = self.apply_noise(high_img)
        # output["hint1"] = torch.cat([output["hint"], output["ref_hint"]], dim=0)
        output["txt"] = ""
        output["input_path"] = input_name
        output["ref_path"] = target_name
        return output
    
    def __getitem__(self, idx):
        return self.process(idx)

class RandomNoise(object):
    def __init__(self, max_poisson=0.8, max_gaussian=0.4):
        self.max_poisson = max_poisson
        self.max_gaussian = max_gaussian

    def __call__(self, image_tensor):

        # Poisson Noise
        noise_poisson = torch.poisson(image_tensor) 
        noise_poisson_sign = torch.randint(low=0, high=2, size=image_tensor.shape) * 2 - 1
        noise_poisson = noise_poisson * noise_poisson_sign * random.uniform(0, self.max_poisson)

        # Gaussian Noise
        noise_gaussian = torch.randn(image_tensor.shape)
        noise_gaussian = noise_gaussian * random.uniform(0, self.max_gaussian)

        image_noise_tensor = image_tensor + noise_gaussian + noise_poisson 
        return torch.clamp(image_noise_tensor, 0, 1)
