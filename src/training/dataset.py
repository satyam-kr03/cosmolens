import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedCosmologyDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, augment=False, augmentation_idx=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.augmentation_idx = augmentation_idx  # For deterministic augmentations
        
        # Define deterministic augmentations: 4 rotations + 2 flips = 8 total
        self.deterministic_augmentations = [
            # Rotations: 0°, 90°, 180°, 270°
            lambda img: img,  # 0°
            lambda img: np.rot90(img, 1).copy(),  # 90°
            lambda img: np.rot90(img, 2).copy(),  # 180°
            lambda img: np.rot90(img, 3).copy(),  # 270°
            # Flips
            lambda img: np.fliplr(img).copy(),  # Horizontal flip
            lambda img: np.flipud(img).copy(),  # Vertical flip
            # Combined: 90° + horizontal flip
            lambda img: np.fliplr(np.rot90(img, 1)).copy(),
            # Combined: 90° + vertical flip  
            lambda img: np.flipud(np.rot90(img, 1)).copy(),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)

        if self.augment:
            if self.augmentation_idx is not None:
                # Deterministic augmentation for TTA
                aug_func = self.deterministic_augmentations[self.augmentation_idx % len(self.deterministic_augmentations)]
                image = aug_func(image)
            else:
                # Random augmentation (legacy behavior)
                if np.random.rand() > 0.5:
                    image = np.fliplr(image).copy()
                if np.random.rand() > 0.5:
                    image = np.flipud(image).copy()
                k = np.random.randint(0, 2)*2 # only 0 or 180 degrees to preserve shape
                if k > 0:
                    image = np.rot90(image, k).copy()
                if np.random.rand() > 0.7:
                    noise = np.random.randn(*image.shape) * 0.01
                    image = image + noise

        if self.transform:
            image = self.transform(image)
            image = image.float()  # Ensure float32 type

        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            label = torch.from_numpy(label)
            return image, label
        else:
            return image