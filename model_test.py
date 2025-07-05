import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataloader import MycoDataLoader
from model import MycoModel

# --- 1. Configuración ---
data_dir = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'
model_path = 'MycoModel.pth'

IMG_SIZE = 224
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

if not os.path.exists(model_path):
    print(f"Error: No se encontró el archivo del modelo en '{model_path}'.")
    exit()

# --- 2. Preparar los datos (sólo validación) ---
# Usamos MycoDataLoader para dividir como en entrenamiento
loader = MycoDataLoader(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, device=device)
class_names = loader.class_names
test_loader = loader.val_loader

print(f"Clases detectadas: {class_names}")
print(f"Imágenes en conjunto de validación: {len(test_loader.dataset)}")

# --- 3. Cargar modelo ---
model_builder = MycoModel(num_classes=len(class_names), device=device)
model_builder.load(model_path)
model = model_builder.model
model.eval()

# --- 4. Evaluar modelo ---
print("\nEvaluando el modelo en el conjunto de validación...")
correct = 0
total = 0
all_predictions = []
all_true_labels = []
all_confidences = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_predictions.extend(preds.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())

accuracy = correct / total
print(f"Precisión en validación: {accuracy:.4f}")

# --- 5. Matriz de confusión ---
cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Matriz de Confusión")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
