import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp
from torchvision import transforms

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.config.data.train_dir,
                                          patch_size=self.config.data.patch_size)
        val_dataset = AllWeatherDataset(self.config.data.val_dir,
                                        patch_size=self.config.data.patch_size, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()

        self.dir = dir
        self.patch_size = patch_size
        
        # Added resizing transform to resize all images to 512x512
        self.resize_transform = transforms.Resize((512, 512))

        # 获取 input 和 target 文件夹中的文件列表
        input_dir = os.path.join(dir, 'input')
        target_dir = os.path.join(dir, 'target')

        input_names = sorted(os.listdir(input_dir))
        target_names = sorted(os.listdir(target_dir))

        # 实现错位排序配对
        self.pairs = []
        for i in range(len(input_names)):
            input_img_name = os.path.join('input', input_names[i])
            target_img_name = os.path.join('target', target_names[(i + 1) % len(target_names)])  # 错位配对
            self.pairs.append((input_img_name, target_img_name))

        if train:
            self.transforms = PairCompose([
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        # 获取输入和目标图像的路径
        input_name, target_name = self.pairs[index]

        # 加载输入和目标图像
        low_img = Image.open(os.path.join(self.dir, input_name)).convert('RGB')
        high_img = Image.open(os.path.join(self.dir, target_name)).convert('RGB')

        # Resize images to 512x512
        low_img = self.resize_transform(low_img)
        high_img = self.resize_transform(high_img)

        # 进行数据增强
        low_img, high_img = self.transforms(low_img, high_img)

        img_id = input_name.split('/')[-1]

        return torch.cat([low_img, high_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.pairs)
