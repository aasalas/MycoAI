import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

class MycoModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Congelar capas base
        for param in model.parameters():
            param.requires_grad = False

        # Reemplazar la capa final
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)

        return model.to(self.device)

    def save(self, path):
        """
        Guarda el modelo en la ruta especificada.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"{path} saved successfully.")

    def load(self, path):
        """
        Carga el modelo desde la ruta especificada.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        self.model = self.build_model()  # reconstruir con el número correcto de clases
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"{path} Loaded successfully.")
