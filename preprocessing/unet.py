# unet_model.py

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        x5 = self.middle(self.pool4(x4))

        x = self.conv1(torch.cat([self.up1(x5), x4], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x3], dim=1))
        x = self.conv3(torch.cat([self.up3(x), x2], dim=1))
        x = self.conv4(torch.cat([self.up4(x), x1], dim=1))

        return torch.sigmoid(self.out_conv(x))
    

    def save(self, path):
        """Guarda los pesos del modelo en el archivo especificado."""
        torch.save(self.state_dict(), path)
        print(f"Modelo guardado en: {path}")


    def load(self, path, map_location=None):
        """Carga los pesos del modelo desde un archivo."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"Modelo cargado desde: {path}")


    @staticmethod
    def dice_coef(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

    @staticmethod
    def iou_score(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return intersection / (union + 1e-8)