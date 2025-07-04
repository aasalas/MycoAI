import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights # Importar para usar los pesos recomendados
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# --- 1. Configuración de rutas y parámetros ---
# La ruta base que contiene tus carpetas de clases (Ectomicorriza, Endomicorriza, Ectendomicorriza)
# ASEGÚRATE de que esta ruta sea EXACTAMENTE la correcta en tu sistema y que esté "limpia"
# (solo conteniendo las tres carpetas de clases).
data_dir = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'

# Nombres de las carpetas de tus tres clases
class_folder_names = ['Ectomicorriza', 'Endomicorriza', 'Ecten- domicorriza']

# Verificar que los directorios de las clases existen y contienen archivos
for folder_name in class_folder_names:
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        print(f"Advertencia: El directorio '{folder_path}' no existe o está vacío.")
        print("Asegúrate de que la ruta es correcta, las imágenes están allí, y que has ejecutado 'rearrange_folders.py'.")
        exit() # Salir si no se encuentran los datos

print(f"Número de archivos en Ectomicorriza: {len(os.listdir(os.path.join(data_dir, 'Ectomicorriza')))}")
print(f"Número de archivos en Endomicorriza: {len(os.listdir(os.path.join(data_dir, 'Endomicorriza')))}")
print(f"Número de archivos en Ectendomicorriza: {len(os.listdir(os.path.join(data_dir, 'Ecten- domicorriza')))}")


# Dimensiones de la imagen para MobileNetV2
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 400 # Puedes ajustar el número de épocas
LEARNING_RATE = 0.001

# Fijar semillas para reproducibilidad (opcional, pero buena práctica)
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # Descomentar para reproducibilidad total si usas CUDA
    # torch.backends.cudnn.benchmark = False    # Descomentar para reproducibilidad total si usas CUDA

set_seed(42)

# Detectar si CUDA está disponible y usar la GPU si lo está, de lo contrario usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# --- 2. Preparación del conjunto de datos ---
# Definir las transformaciones para las imágenes
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Convierte PIL Image a Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), # Convierte PIL Image a Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar el conjunto de datos completo (sin transformaciones iniciales, solo para obtener índices y clases)
# Esto asegura que random_split opera sobre el mismo conjunto de datos base
full_dataset_no_transform = datasets.ImageFolder(data_dir)
class_names = full_dataset_no_transform.classes
print(f"Clases detectadas por ImageFolder: {class_names}")

# Verificar que se detectaron las 3 clases esperadas
if len(class_names) != 3:
    print(f"Error: Se esperaban 3 clases (Ectomicorriza, Endomicorriza, Ectendomicorriza), pero ImageFolder detectó {len(class_names)}: {class_names}")
    print("Asegúrate de que las carpetas de las tres clases existen directamente en el data_dir y no hay otras carpetas no deseadas.")
    exit()

# Dividir los índices del conjunto de datos en entrenamiento y validación
train_size = int(0.8 * len(full_dataset_no_transform))
val_size = len(full_dataset_no_transform) - train_size

# Generar índices aleatorios para la división. Esto asegura que la división sea reproducible
# si la semilla está fijada.
indices = list(range(len(full_dataset_no_transform)))
random.shuffle(indices) # Mezclar los índices para una división aleatoria

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Ahora, crea las instancias de ImageFolder con las transformaciones apropiadas
# y luego usa los índices para crear los Subsets. Cada Subset ahora referenciará
# una ImageFolder con su propia pipeline de transformaciones.
train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transforms), train_indices)
val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transforms), val_indices)


# Crear DataLoaders para cargar los datos en lotes
# num_workers puede ser 0 para depuración o si tienes problemas con múltiples procesos en Windows.
# Para GPU (cuda), se pueden usar más workers si el sistema lo soporta.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if device == "cuda:0" else 0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if device == "cuda:0" else 0)

# --- 3. Definición del modelo (MobileNetV2 pre-entrenado) ---
# Usar `weights` en lugar de `pretrained` para evitar la advertencia de deprecación
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names)) # Número de salidas se ajusta a las clases detectadas

model = model.to(device)

# --- 4. Definición de la función de pérdida y el optimizador ---
# --- Calcular pesos para clases desbalanceadas ---
class_counts = [212, 431, 964]  # Reemplaza estos números con el conteo real si cambia
total = sum(class_counts)

# Pesos inversamente proporcionales a la frecuencia de cada clase
class_weights = [total / c for c in class_counts]
class_weights = [w / sum(class_weights) for w in class_weights]  # Normaliza
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# --- Función de pérdida con pesos ---
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





