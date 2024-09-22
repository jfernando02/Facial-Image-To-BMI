import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import functional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms as T
from torchvision.transforms import ToTensor, InterpolationMode
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, to_tensor, to_pil_image


# helper class for random distortion
class RandomDistortion(torch.nn.Module):
    def __init__(self, probability=0.25, grid_width=2, grid_height=2, magnitude=8):
        super().__init__()
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            return T.functional.affine(img, 0, [0, 0], 1, [self.magnitude, self.magnitude], interpolation=T.InterpolationMode.NEAREST, fill=[0, 0, 0])
        else:
            return img

# helper class for random adjust contrast
class RandomAdjustContrast(torch.nn.Module):
    def __init__(self, probability=.5, min_factor=0.8, max_factor=1.2):
        super().__init__()
        self.probability = probability
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
            return adjust_contrast(img, factor)
        else:
            return img

# Helper class for Canny edge detection
class CannyEdgeDetection(torch.nn.Module):
    def __init__(self, threshold1=50, threshold2=100):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, img):
        img = np.array(to_pil_image(img))
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = self.auto_canny(img)
        edges = Image.fromarray(edges).convert('RGB')
        img = to_tensor(edges)
        return img

    def auto_canny(self, img, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(img, lower, upper)

class Segmentation(torch.nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, img):
        # Convert to PIL image and then to numpy array
        img = np.array(to_pil_image(img))

        if img.ndim == 3 and img.shape[2] == 3:
            # Reshaping the image to a 2D array of pixels
            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)

            # Define the criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert back to 8-bit values
            centers = np.uint8(centers)
            labels = labels.flatten()

            # Convert all pixels to the color of the centroids
            segmented_image = centers[labels.flatten()]

            # Reshape back to the original image dimension
            segmented_image = segmented_image.reshape(img.shape)
        else:
            raise ValueError("Image is not in expected format or number of channels")

        # Convert back to PIL Image and then to Tensor
        segmented_image = Image.fromarray(segmented_image)
        img = to_tensor(segmented_image)

        return img

# transforms for data augmentation
augmentation_transforms = T.Compose([
    T.RandomRotation(5),
    T.RandomHorizontalFlip(p=0.5),
    RandomDistortion(probability=0.25, grid_width=2, grid_height=2, magnitude=8),
    T.RandomApply([T.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(0.0, 0.1))], p=1),
    RandomAdjustContrast(probability=0.5, min_factor=0.8, max_factor=1.2),
    T.Lambda(lambda img: adjust_brightness(img, torch.rand(1).item() + 0.5))
])

#transforms for data augmentation and canny edge detection
edge_detection_transforms = T.Compose([
    CannyEdgeDetection()
])

#transforms for data augmentation and GLCM texture detection
segmentation_transforms = T.Compose([
    Segmentation()
])

# transforms for vit
vit_transforms = T.Compose([
    T.Resize([518], interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop([518]),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# the original dataset
class BMIDataset(Dataset):
    def __init__(self, csv_path, image_folder, y_col_name, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.image_folder = image_folder

        # Drop the rows where the image does not exist
        images = os.listdir(image_folder)
        self.csv = self.csv[self.csv['name'].isin(images)]
        self.csv.reset_index(drop=True, inplace=True)

        self.y_col_name = y_col_name
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.csv.iloc[idx, 4])
        image = Image.open(image_path)

        # check the channel number
        if image.mode != 'RGB':
            image = image.convert('RGB')

        y = self.csv.loc[idx, self.y_col_name]

        if self.transform:
            image = self.transform(image)

        return image, y



# the augmented dataset
class AugmentedBMIDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return 5 * len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx // 5]

        if self.transforms and (idx % 5 != 0):
            image = self.transforms(image)

        return image, y



# the dataset transformed (by default for vit inputs)
class TransformedDataset(Dataset):
    def __init__(self, original_dataset, transforms=vit_transforms):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, y


# show 5 sample images
def show_sample_image(dataset):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        image, label = dataset.__getitem__(i)
        ax.imshow(image.detach().cpu().permute(1, 2, 0), cmap='gray')
        ax.set_title('gt BMI: ' + str(label))
        ax.axis('off')  # Hide axes
    plt.show()

# split dataset and (optionally) augment and/ or transform it for vit
def train_val_test_split(dataset, augmented=True, vit_transformed=True, detection=None):
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    if augmented:
        train_dataset = AugmentedBMIDataset(train_dataset, augmentation_transforms)

    if detection=="edge":
        train_dataset = TransformedDataset(train_dataset, edge_detection_transforms)
        val_dataset = TransformedDataset(val_dataset, edge_detection_transforms)
        test_dataset = TransformedDataset(test_dataset, edge_detection_transforms)

    elif detection=="segmentation":
        train_dataset = TransformedDataset(train_dataset, segmentation_transforms)
        val_dataset = TransformedDataset(val_dataset, segmentation_transforms)
        test_dataset = TransformedDataset(test_dataset, segmentation_transforms)

    if vit_transformed:
        train_dataset = TransformedDataset(train_dataset)
        val_dataset = TransformedDataset(val_dataset)
        test_dataset = TransformedDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset



# get dataloaders
def get_dataloaders(batch_size=16, augmented=True, vit_transformed=True, show_sample=False, detection=None):
    bmi_dataset = BMIDataset('../data/data.csv', '../data/Images', 'bmi', ToTensor())
    if show_sample:
        train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed, detection)
        show_sample_image(train_dataset)
    train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed, detection)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, test_loader, val_loader



# for test
if __name__ == "__main__":
    get_dataloaders(augmented=False, vit_transformed=True, show_sample=True, detection="segmentation")
    get_dataloaders(augmented=True, vit_transformed=True, show_sample=True, detection="segmentation")
    get_dataloaders(augmented=False, vit_transformed=True, show_sample=True, detection="edge")
    get_dataloaders(augmented=True, vit_transformed=True, show_sample=True, detection="edge")