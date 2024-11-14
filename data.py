import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *

# Define the path to your dataset
# data_path = 'path/to/your/image/dataset'
if __name__ == "__main__":

    # Define a simple transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataloader = DataLoader(
        TrainImageDataset(r"D:\alvin\gan\DIV2K_train_HR\DIV2K_train_HR", hr_shape=(96,96)),
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Initialize variables to accumulate mean and std values
    mean = torch.zeros(3)
    std = torch.zeros(3)

    # Loop through the dataset to calculate mean and std
    for images in tqdm(dataloader):
        imgs_lr = images["lr"]
        images = images["hr"]
        # Reshape to (batch_size, channels, height * width)
        images = images.view(images.size(0), images.size(1), -1)
        # Calculate mean and std per channel
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # Divide by the total number of images to get the mean and std
    mean /= 800
    std /= 800

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
