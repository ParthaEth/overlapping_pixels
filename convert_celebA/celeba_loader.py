import os
from PIL import Image
import torch

import configs

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


celeba_transform = transforms.Compose([
    transforms.CenterCrop(128),  # Crop the images to 128x128
    transforms.Resize(128),  # Resize the images to 128x128
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
])


class ImageFolderOffsatable(Dataset):
    def __init__(self, start_offset, image_dir=None, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if image_dir is None:
            image_dir = os.path.join(configs.celeba_config.datapath, 'celeba/img_align_celeba')
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.start_offset = start_offset

    def __len__(self):
        return len(self.image_files) - self.start_offset

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('Index out of bound')
        img_name = os.path.join(self.image_dir, self.image_files[idx + self.start_offset])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB to ensure 3 channels

        if self.transform:
            image = self.transform(image)

        return image, os.path.splitext(os.path.basename(img_name))[0]


def get_offsatable_data_loader(batch_size, start_idx):
    dataset = ImageFolderOffsatable(start_idx, transform=celeba_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)


def get_celeba_data_loaders(batch_size, return_testloader=False):
    # Set the directory where the CelebA data will be downloaded (or is already located)
    data_root = configs.celeba_config.datapath

    # Load the CelebA dataset
    celeba_train = datasets.CelebA(root=data_root, split='train', target_type='attr', transform=celeba_transform, download=False)
    celeba_valid = datasets.CelebA(root=data_root, split='valid', target_type='attr', transform=celeba_transform, download=False)
    celeba_test = datasets.CelebA(root=data_root, split='test', target_type='attr', transform=celeba_transform, download=False)

    # Create DataLoader for training, validation, and test sets
    train_loader = DataLoader(celeba_train, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(celeba_valid, batch_size=batch_size, shuffle=False, num_workers=8)
    if return_testloader:
        test_loader = DataLoader(celeba_test, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_celeba_data_loaders(64)
    # Example: Iterate over the training dataset
    for images, attributes in train_loader:
        # Here you can use images and attributes
        print(images.size(), attributes.size())  # E.g., torch.Size([64, 3, 128, 128]) for images
        break  # Breaking after one iteration for demonstration
