import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import torch

class RootSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.resize = A.Resize(180, 240)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        resized = self.resize(image=image, mask=mask)
        image = resized["image"]
        mask = resized["mask"]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(mask).long().unsqueeze(0)

        return image, mask
