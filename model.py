import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights
)

class MycoModel:
    def __init__(self, num_classes, device, architecture="resnet50"):
        self.device = device
        self.num_classes = num_classes
        self.architecture = architecture.lower()
        self.model = self.build_model()

    def build_model(self):
        if self.architecture == "mobilenet":
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

            # Congelar todas las capas
            for param in model.parameters():
                param.requires_grad = False

            # Reemplazar y descongelar la capa final
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
            for param in model.classifier[1].parameters():
                param.requires_grad = True

        elif self.architecture == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Congelar todas las capas
            for param in model.parameters():
                param.requires_grad = False

            # Reemplazar y descongelar la capa final
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            for param in model.fc.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Arquitectura '{self.architecture}' no soportada. Usa 'mobilenet' o 'resnet50'.")

        return model.to(self.device)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'architecture': self.architecture
        }, path)
        print(f"{path} saved successfully.")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        self.architecture = checkpoint.get('architecture', 'resnet50')
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"\n{path} loaded successfully.")
