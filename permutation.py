import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt  # <-- import for visualization

# 1. Function to permute pixels of a single image
def permutate_image_pixels(image, permutation):
    '''
    image: a torch Tensor of shape (H, W) or (1, H, W)
    permutation: a list or array of indices of length H*W
    
    Returns: permuted image tensor reshaped back to (H, W)
    '''
    # Flatten the image to a vector (length H*W)
    flat_image = image.view(-1)
    
    # Apply permutation
    permuted_flat = flat_image[permutation]
    
    # Reshape back to original image shape
    permuted_image = permuted_flat.view(image.shape)
    
    return permuted_image


# 2. Class to apply the transformation on a dataset
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


# 3. Create a fixed random permutation for all images (28*28 = 784 pixels)
permutation = torch.randperm(28*28)

# 4. Define a transform function that applies the permutation
def permute_transform(image):
    return permutate_image_pixels(image, permutation)

# 5. Load original MNIST dataset (as tensor images)
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# 6. Wrap MNIST with TransformedDataset applying the permutation
permuted_mnist_train = TransformedDataset(mnist_train, transform=permute_transform)

# Example usage: get first permuted image and label
image, label = permuted_mnist_train[0]
print(image.shape)  # Should be (1, 28, 28)
print(label)

# --- Visualization ---

# Also get the original image and label for comparison
original_image, original_label = mnist_train[0]

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title(f'Original: {original_label}')
plt.imshow(original_image.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f'Permuted: {label}')
plt.imshow(image.squeeze(), cmap='gray')
plt.axis('off')

plt.show()
