import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class MycoDataLoader:
    def __init__(self, data_dir, img_size=224, batch_size=50, seed=42, device='cpu'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.set_seed()
        self.class_names = self.check_classes()
        self.train_loader, self.val_loader = self.prepare_dataloaders()

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def check_classes(self):
        folders = os.listdir(self.data_dir)
        print(f"Carpetas encontradas: {folders}")
        dataset = datasets.ImageFolder(self.data_dir)
        return dataset.classes

    def prepare_dataloaders(self):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.25, 0.25), shear=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(self.data_dir)
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        train_size = int(0.8 * len(full_dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(datasets.ImageFolder(self.data_dir, transform=transform_train), train_indices)
        val_dataset = Subset(datasets.ImageFolder(self.data_dir, transform=transform_val), val_indices)

        num_workers = 4 if self.device.type == 'cuda' else 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader
