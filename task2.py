import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset with ToTensor transform
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print(f"Training samples: {len(mnist_trainset)}, Test samples: {len(mnist_testset)}")

# Config dictionary with image details
config = {'size': 28, 'channels': 1, 'classes': 10}


# Visualization functions
def multi_context_barplot(axis, accs, title=None):
    '''
    Generate barplot using the values in [accs].
    Args:
        axis: matplotlib axis object
        accs: list or array of accuracy values (percent)
        title: optional title string
    '''
    contexts = len(accs)
    axis.bar(range(contexts), accs, color='k')
    axis.set_ylabel('Testing Accuracy (%)')
    axis.set_xticks(range(contexts))
    axis.set_xticklabels([f'Context {i+1}' for i in range(contexts)])
    if title is not None:
        axis.set_title(title)


def plot_examples(axis, dataset, context_id=None):
    '''
    Plot 25 examples from [dataset].
    Args:
        axis: matplotlib axis object
        dataset: a PyTorch Dataset object
        context_id: optional int for title
    '''
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)
    image_tensor, _ = next(iter(data_loader))  # batch of 25 images
    # Make grid of images (5 rows, 5 columns)
    image_grid = make_grid(image_tensor, nrow=5, pad_value=1)
    # Convert tensor to numpy and transpose channels for matplotlib
    np_img = np.transpose(image_grid.numpy(), (1, 2, 0))
    axis.imshow(np_img.squeeze(), cmap='gray')  # squeeze to remove channel dim if 1
    if context_id is not None:
        axis.set_title(f"Context {context_id+1}")
    axis.axis('off')


# Example usage: plot 25 samples from training data
fig, ax = plt.subplots(figsize=(6,6))
plot_examples(ax, mnist_trainset, context_id=0)
plt.show()

# Example usage: plot dummy accuracy barplot for 3 contexts
fig, ax = plt.subplots()
example_accuracies = [92.5, 85.7, 78.9]
multi_context_barplot(ax, example_accuracies, title="Example Test Accuracies")
plt.show()
