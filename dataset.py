import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

def elastic_deformation(img):
    H, W = img.shape[:2]

    dx = np.random.randn(3, 3) * 10
    dy = np.random.randn(3, 3) * 10

    map_x = cv2.resize(dx, (W, H), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(dy, (W, H), interpolation=cv2.INTER_CUBIC)

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (x + map_x).astype(np.float32)
    map_y = (y + map_y).astype(np.float32)

    output = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return output

class UNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.train = train
        self.size = (572, 572)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.train:
            image = elastic_deformation(image)
            label = elastic_deformation(label)

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        label = torch.from_numpy(label).unsqueeze(0).float() / 255.0

        return image, label