import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataloader import MycoDataLoader
from model import MycoModel

# --- Parámetros ---
DATA_DIR = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'
SAVE_PATH = 'MycoModel.pth'
CLASS_COUNTS = [212, 431, 964]
IMG_SIZE = 224
BATCH_SIZE = 50
NUM_EPOCHS = 2
LEARNING_RATE = 0.001


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- Cargar datos ---
loader = MycoDataLoader(DATA_DIR, IMG_SIZE, BATCH_SIZE, device=device)
train_loader = loader.train_loader
val_loader = loader.val_loader
class_names = loader.class_names

# --- Inicializar modelo ---
model_builder = MycoModel(num_classes=len(class_names), device=device)
model = model_builder.model

# --- Pérdida ponderada ---
total = sum(CLASS_COUNTS)
weights = [total / c for c in CLASS_COUNTS]
weights = [w / sum(weights) for w in weights]
weights_tensor = torch.FloatTensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# --- Entrenamiento ---
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print(f"\nEntrenando {NUM_EPOCHS} épocas...")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(train_loss / total)
    train_accuracies.append(correct / total)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_losses.append(val_loss / total)
    val_accuracies.append(correct / total)

    print(f"Época {epoch+1}/{NUM_EPOCHS} - "
          f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f}, Acc: {val_accuracies[-1]:.4f}")

# --- Visualización ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.legend(), plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(), plt.title("Loss")

plt.tight_layout()
plt.show()

# --- Guardar modelo ---
model_builder.save(SAVE_PATH)
print(f"Modelo guardado como + {SAVE_PATH}")
