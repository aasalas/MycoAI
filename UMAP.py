import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import umap # Ensure 'umap-learn' is installed (pip install umap-learn)
import matplotlib.pyplot as plt # Ensure 'matplotlib' is installed (pip install matplotlib)
import numpy as np
import os
import random

# --- 1. Configuration and Data Preparation ---
# Path to your dataset root directory.
# MAKE SURE this path is EXACTLY correct on your system.
data_dir = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'

# Names of your three class folders.
# IMPORTANT: Verify these names EXACTLY match your folder names.
class_folder_names = ['Ectomicorriza', 'Endomicorriza', 'Ectendomicorriza'] # Common correction

# Image dimensions to which images will be resized before flattening.
# Using 224x224, a common size.
IMG_SIZE = 224
BATCH_SIZE = 32 # For efficient batch loading

# Set seeds for reproducibility.
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Detect device (CPU or GPU). DataLoader can benefit from a GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations:
# 1. Resize: Ensures all images are IMG_SIZE x IMG_SIZE.
# 2. ToTensor: Converts PIL Image to a PyTorch Tensor (scales pixels to [0, 1]).
# 3. Normalize: Normalizes pixel values using ImageNet statistics.
image_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the full dataset using ImageFolder.
# ImageFolder automatically infers class labels from folder names.
full_dataset = datasets.ImageFolder(data_dir, transform=image_transforms)

# Get the class names detected by ImageFolder.
class_names = full_dataset.classes
print(f"Classes detected by ImageFolder: {class_names}")

# Basic check for the expected number of classes.
if len(class_names) != 3:
    print(f"Warning: Expected 3 classes, but ImageFolder detected {len(class_names)}: {class_names}")
    print("Please ensure your 'data_dir' is correct and directly contains the three class folders.")

# Create a DataLoader to load images in batches.
data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4 if device == "cuda:0" else 0,
                         pin_memory=True if device == "cuda:0" else False)

# --- 2. Load and Flatten Images ---
print("\nLoading images and flattening into pixel vectors...")
pixel_vectors = []
labels_list = []

# No gradient calculation needed for this task.
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)

        # Flatten each image into a 1D vector.
        # For a 224x224 RGB image, this results in 3 * 224 * 224 = 150528 features per image.
        flattened_images = images.view(images.size(0), -1)

        pixel_vectors.append(flattened_images.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# Concatenate all pixel vectors and labels into NumPy arrays.
all_pixel_vectors = np.concatenate(pixel_vectors, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

print(f"Total images loaded: {len(all_labels)}")
print(f"Dimensions of flattened pixel vectors: {all_pixel_vectors.shape}")

# --- 3. Apply Supervised UMAP for 2D Dimensionality Reduction ---
print("\nApplying Supervised UMAP to reduce dimensionality to 2D...")
# Initialize the UMAP reducer for supervised learning.
# 'n_components=2' for a 2D projection.
# 'random_state=42' for reproducibility.
# The 'y' parameter in fit_transform enables supervised UMAP.
reducer = umap.UMAP(n_components=2, random_state=42)

# Train UMAP using the pixel vectors (all_pixel_vectors) AND their corresponding labels (all_labels).
embedding = reducer.fit_transform(all_pixel_vectors, y=all_labels)

print(f"Supervised UMAP embedding generated with dimensions: {embedding.shape}")

# --- 4. Visualize the Clusters in 2D ---
print("Generating the 2D Supervised UMAP plot...")
plt.figure(figsize=(10, 8))

# Create the scatter plot, coloring points by their true class labels.
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='viridis', s=10, alpha=0.8)

# Add a legend to map colors to class names.
legend_handles = scatter.legend_elements()[0]
plt.legend(handles=legend_handles, labels=class_names,
           title="Mycorrhiza Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('2D Supervised UMAP Projection of Raw Image Pixels')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("2D Supervised UMAP plot generated successfully.")