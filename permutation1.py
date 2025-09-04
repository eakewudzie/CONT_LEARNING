import torch
import matplotlib.pyplot as plt

# Create a fake "digit-like" image (28x28) for demo
# We'll just make a white square in the middle
image = torch.zeros((28,28))
image[8:20, 10:18] = 1.0  # block to represent a digit-like shape

# Two different random permutations
perm1 = torch.randperm(28*28)
perm2 = torch.randperm(28*28)

def permute_image(image, permutation):
    flat = image.view(-1)
    permuted = flat[permutation]
    return permuted.view(image.shape)

# Apply both permutations
image_perm1 = permute_image(image, perm1)
image_perm2 = permute_image(image, perm2)

# Plot: original, permuted1, permuted2
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title('Original (fake digit)')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Permutation 1')
plt.imshow(image_perm1, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Permutation 2')
plt.imshow(image_perm2, cmap='gray')
plt.axis('off')

plt.show()
