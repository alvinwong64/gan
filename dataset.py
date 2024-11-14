import os
import numpy as np

# import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize



def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class TrainImageDataset(Dataset):
    def __init__(self, img_dir, hr_shape):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir) 
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                ToPILImage(),
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.RandomCrop(hr_height),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )



    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_name)
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img_hr)


        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.img_names)

class TestImageDataset(Dataset):
    def __init__(self,img_dir, scaling):
        super().__init__()
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir) 
        self.scaling = scaling

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_names[index])
        hr_img = Image.open(img_name)
        w,h = hr_img.size
        crop_size = calculate_valid_crop_size(min(w,h), self.scaling)
        lr_scale = Resize(crop_size // self.scaling, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_img)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    
    def __len__(self):
        return len(self.img_names)

