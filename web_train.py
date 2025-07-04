import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cv2 # Importar OpenCV
from PIL import Image # Importar PIL para trabajar con imágenes

# --- 1. Configuración de rutas y parámetros ---
data_dir = r'/content/drive/MyDrive/Micorrizas-DataSet'
class_folder_names = ['Ectomicorriza', 'Endomicorriza', 'Ecten- domicorriza']

for folder_name in class_folder_names:
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        print(f"Advertencia: El directorio '{folder_path}' no existe o está vacío.")
        print("Asegúrate de que la ruta es correcta, las imágenes están allí, y que has ejecutado 'rearrange_folders.py'.")
        exit()

print(f"Número de archivos en Ectomicorriza: {len(os.listdir(os.path.join(data_dir, 'Ectomicorriza')))}")
print(f"Número de archivos en Endomicorriza: {len(os.listdir(os.path.join(data_dir, 'Endomicorriza')))}")
print(f"Número de archivos en Ectendomicorriza: {len(os.listdir(os.path.join(data_dir, 'Ecten- domicorriza')))}")


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 400
LEARNING_RATE = 0.001

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# --- CLASE DE TRANSFORMACIÓN PERSONALIZADA PARA DETECCIÓN DE BORDES ---
class EdgeDetectionTransform:
    def __init__(self, method='canny', low_threshold=50, high_threshold=150):
        self.method = method
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, img):
        # Convertir PIL Image a OpenCV (NumPy array)
        img_np = np.array(img)

        # Convertir a escala de grises para detección de bordes
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if self.method == 'sobel':
            # Aplicar Sobel en X e Y
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # Combinar los resultados
            edge_img = np.sqrt(sobelx**2 + sobely**2)
            # Normalizar la imagen de bordes a 0-255 y convertir a uint8
            edge_img = np.uint8(255 * edge_img / edge_img.max())
        elif self.method == 'canny':
            # Canny Edge Detector
            edge_img = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        elif self.method == 'laplacian':
            edge_img = cv2.Laplacian(gray, cv2.CV_64F)
            edge_img = np.uint8(255 * (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min()))
        else:
            raise ValueError(f"Método de detección de bordes '{self.method}' no soportado. Usa 'canny', 'sobel' o 'laplacian'.")

        # Convertir de nuevo a PIL Image y asegurarse de que sea RGB
        # Replicamos el canal de bordes 3 veces para que coincida con la entrada RGB del modelo
        edge_img_pil = Image.fromarray(edge_img).convert('RGB')
        return edge_img_pil

class RandomEdgeDetection(object):
    """
    Aplica la detección de bordes con una probabilidad dada.
    Si no se aplica, devuelve la imagen original.
    """
    def __init__(self, method='canny', low_threshold=50, high_threshold=150, p=0.5):
        self.edge_transform = EdgeDetectionTransform(method=method, low_threshold=low_threshold, high_threshold=high_threshold)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.edge_transform(img)
        return img

# --- 2. Preparación del conjunto de datos ---
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    RandomEdgeDetection(method='canny', low_threshold=50, high_threshold=150, p=0.3), # Aplica Canny en el 30% de las imágenes de entrenamiento
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    # Generalmente, no aplicas aumentos aleatorios en validación.
    # Si quieres evaluar el modelo con imágenes de bordes en validación,
    # puedes aplicar EdgeDetectionTransform directamente aquí (sin RandomEdgeDetection)
    # EdgeDetectionTransform(method='canny', low_threshold=50, high_threshold=150),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset_no_transform = datasets.ImageFolder(data_dir)
class_names = full_dataset_no_transform.classes
print(f"Clases detectadas por ImageFolder: {class_names}")

if len(class_names) != 3:
    print(f"Error: Se esperaban 3 clases (Ectomicorriza, Endomicorriza, Ectendomicorriza), pero ImageFolder detectó {len(class_names)}: {class_names}")
    print("Asegúrate de que las carpetas de las tres clases existen directamente en el data_dir y no hay otras carpetas no deseadas.")
    exit()

train_size = int(0.8 * len(full_dataset_no_transform))
val_size = len(full_dataset_no_transform) - train_size

indices = list(range(len(full_dataset_no_transform)))
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transforms), train_indices)
val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transforms), val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if device == "cuda:0" else 0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if device == "cuda:0" else 0)

# --- 3. Definición del modelo (MobileNetV2 pre-entrenado) ---
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

# --- 4. Definición de la función de pérdida y el optimizador ---
class_counts = [212, 431, 964]
total = sum(class_counts)

class_weights = [total / c for c in class_counts]
class_weights = [w / sum(class_weights) for w in class_weights]
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# --- 5. Entrenamiento del modelo ---
print(f"\nIniciando el entrenamiento del modelo por {NUM_EPOCHS} épocas...")
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Época {epoch+1}/{NUM_EPOCHS} - "
          f"Pérdida de Entrenamiento: {epoch_train_loss:.4f}, Precisión de Entrenamiento: {epoch_train_acc:.4f} - "
          f"Pérdida de Validación: {epoch_val_loss:.4f}, Precisión de Validación: {epoch_val_acc:.4f}")

print("Entrenamiento del modelo completado.")

# --- 6. Visualización de resultados ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Precisión de Entrenamiento')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Precisión de Validación')
plt.title('Precisión de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Pérdida de Entrenamiento')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Pérdida de Validación')
plt.title('Pérdida de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.show()

# --- 7. Guardar el modelo entrenado ---
model_save_path = 'modelo_micorrizas_3clases_pytorch.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Modelo entrenado guardado como '{model_save_path}' en el directorio actual.")