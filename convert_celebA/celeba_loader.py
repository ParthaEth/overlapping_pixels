import configs

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transform to preprocess the data
transform = transforms.Compose([
    transforms.CenterCrop(128),  # Crop the images to 128x128
    transforms.Resize(128),  # Resize the images to 128x128
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
])

# Set the directory where the CelebA data will be downloaded (or is already located)
data_root = configs.datapath

# Load the CelebA dataset
celeba_train = datasets.CelebA(root=data_root, split='train', target_type='attr', transform=transform, download=True)
celeba_valid = datasets.CelebA(root=data_root, split='valid', target_type='attr', transform=transform, download=True)
celeba_test = datasets.CelebA(root=data_root, split='test', target_type='attr', transform=transform, download=True)

# Create DataLoader for training, validation, and test sets
train_loader = DataLoader(celeba_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(celeba_valid, batch_size=64, shuffle=False)
test_loader = DataLoader(celeba_test, batch_size=64, shuffle=False)

# Example: Iterate over the training dataset
for images, attributes in train_loader:
    # Here you can use images and attributes
    print(images.size(), attributes.size())  # E.g., torch.Size([64, 3, 128, 128]) for images
    break  # Breaking after one iteration for demonstration
