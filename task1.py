import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# load and preparing the dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)


# loading 64 images at a time per training step
# shuffling improves generalization and learning speed
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Check that data loads
for images, labels in train_loader:
    print(images.shape, labels.shape)
    # break

