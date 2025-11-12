import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedCosmologyDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, augment=False):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)

        if self.augment:
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