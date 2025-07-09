import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter


class MycoDataLoader:
    def __init__(self, data_dir, img_size=224, batch_size=50, seed=42, device='cpu'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device if isinstance(device, str) else device.type
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

    def compute_mean_std(self, dataset):
        loader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)
        mean = 0.
        std = 0.
        total = 0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total += batch_samples
        mean /= total
        std /= total
        #print(f"Media calculada: {mean}")
        #print(f"Desviación estándar calculada: {std}")
        return mean, std

    def prepare_dataloaders(self):

        resize_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor()
        ])
        # Dataset sin transformaciones para calcular mean/std
        base_dataset = datasets.ImageFolder(self.data_dir, transform=resize_transform)

        indices = list(range(len(base_dataset)))
        random.shuffle(indices)
        train_size = int(0.8 * len(base_dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Subset de entrenamiento para estadísticas
        raw_train_dataset = Subset(base_dataset, train_indices)
        mean, std = self.compute_mean_std(raw_train_dataset)

        # Transformaciones actualizadas con estadísticas reales
        transform_train = transforms.Compose([
            # 1. Transformaciones Geométricas
            transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.5)), # Ya lo tienes
            transforms.RandomHorizontalFlip(), # Ya lo tienes
            transforms.RandomRotation(30), # Ya lo tienes
            transforms.RandomAffine(0, translate=(0.25, 0.25), shear=15), # Ya lo tienes
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # NUEVO: Distorsión de perspectiva
            # transforms.ElasticTransform(alpha=250.0, sigma=10.0, p=0.5), # Opcional: para deformaciones más "orgánicas"

            # 2. Transformaciones de Color y Tono
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # NUEVO: Variaciones de color
            transforms.RandomAutocontrast(p=0.2), # NUEVO: Ajuste automático de contraste
            transforms.RandomEqualize(p=0.2), # NUEVO: Ecualización de histograma

            # 3. Transformaciones de Nitidez y Borrosidad
            transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2), # NUEVO: Ajuste de nitidez
            transforms.GaussianBlur(kernel_size=(5, 9)), # NUEVO: Desenfoque gaussiano (puedes ajustar kernel_size)

            # 4. Convertir a Tensor y Normalizar (SIEMPRE al final para la mayoría)
            transforms.ToTensor(), # Convierte a Tensor (valores [0,1])

            # 6. Normalizar (después de ToTensor y añadir ruido)
            transforms.Normalize(mean.tolist(), std.tolist()) # Usando tus estadísticas
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ])

        # Dataset final con transformaciones reales
        full_dataset_train = datasets.ImageFolder(self.data_dir, transform=transform_train)
        full_dataset_val = datasets.ImageFolder(self.data_dir, transform=transform_val)

        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)

        num_workers = 4 if self.device == 'cuda' else 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader

    def get_class_weights(self):
        targets = [self.train_loader.dataset.dataset.targets[i] for i in self.train_loader.dataset.indices]
        class_counts = Counter(targets)
        total = sum(class_counts.values())
        weights = [total / class_counts[i] for i in range(len(self.class_names))]
        weights = [w / sum(weights) for w in weights]
        return torch.FloatTensor(weights).to(self.device)
