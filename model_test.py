import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuración de rutas y parámetros ---
# La ruta base que contiene tus carpetas de clases (Ectomicorriza y Endomicorriza)
# ASEGÚRATE de que esta ruta sea la misma que usaste en model_train.py
data_dir = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'

# Ruta al modelo entrenado guardado por model_train.py
model_path = 'modelo_micorrizas_3clases_pytorch2.pth'

IMG_SIZE = 224
BATCH_SIZE = 32

# Detectar si CUDA está disponible y usar la GPU si lo está, de lo contrario usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Verificar si el modelo existe
if not os.path.exists(model_path):
    print(f"Error: No se encontró el archivo del modelo en '{model_path}'.")
    print("Asegúrate de haber ejecutado 'model_train.py' primero para entrenar y guardar el modelo.")
    exit()

# --- 2. Preparación del conjunto de datos de prueba ---
# Las transformaciones para el conjunto de prueba son las mismas que para la validación
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar el conjunto de datos completo para poder dividirlo en validación
full_dataset = datasets.ImageFolder(data_dir)
class_names = full_dataset.classes

# Re-crear la división de entrenamiento/validación para obtener el mismo conjunto de validación
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Asignar las transformaciones al conjunto de prueba
test_dataset.dataset.transform = test_transforms

# Crear DataLoader para el conjunto de prueba
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if device == "cuda:0" else 0)

print(f"Clases detectadas para prueba: {class_names}")
print(f"Número de imágenes en el conjunto de prueba: {len(test_dataset)}")

# --- 3. Cargar y configurar el modelo ---
# Cargar MobileNetV2 pre-entrenado (necesitamos la arquitectura para cargar los pesos)
model = models.mobilenet_v2(pretrained=False) # No necesitamos pre-entrenado ya que cargaremos nuestros pesos

# Reemplazar la capa clasificadora para nuestro número de clases (2)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Cargar los pesos guardados
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Poner el modelo en modo evaluación
model = model.to(device)

print(f"Modelo cargado exitosamente desde '{model_path}'.")

# --- 4. Evaluar el modelo ---
print("\nEvaluando el modelo en el conjunto de prueba...")
correct = 0
total = 0
all_predictions = []
all_true_labels = []
all_confidences = [] # Para almacenar la confianza de la clase predicha

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1) # Obtener probabilidades
        confidences, predicted = torch.max(probabilities.data, 1) # Obtener confianza y clase predicha
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

accuracy = correct / total
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")






# --- 5. Generar la matriz de confusión ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear matriz de confusión
cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Graficar la matriz de confusión
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Matriz de Confusión")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
