import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from dataloader import MycoDataLoader
from model import MycoModel

# --- Parámetros ---
DATA_DIR = r'C:\Users\Lenovo Yoga\Desktop\MycoAI\Micorrizas-DataSet'
SAVE_PATH = 'MycoModel.pth' # Nombre para guardar el modelo entrenado
IMG_SIZE = 224  # Quizas si aumentamos la resolución mejora la precisión
BATCH_SIZE = 20
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
ARCHITECTURE = "mobilenet"  # mobilenet o resnet50

# Inicializa wandb
wandb.login(key="604cb8bc212df5c53f97526f8520c686e12d8588") #CUENTA DE AARON
wandb.init(
    project=f"MycoAI-Classifier",  # Cambia esto por tu proyecto real
    name=f"{ARCHITECTURE}_img{IMG_SIZE}_{wandb.util.generate_id()[:4]}",
    config={
        "architecture": ARCHITECTURE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMG_SIZE,
        "optimizer": "Adam",
        "loss": "CrossEntropyWeighted"
    }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- Cargar datos ---
loader = MycoDataLoader(DATA_DIR, IMG_SIZE, BATCH_SIZE, device=device)
train_loader = loader.train_loader
val_loader = loader.val_loader
class_names = loader.class_names

# --- Inicializar modelo ---
model_builder = MycoModel(num_classes=len(class_names), device=device, architecture=ARCHITECTURE)
model = model_builder.model

weights_tensor = loader.get_class_weights()
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE) if ARCHITECTURE == "mobilenet" else optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_losses.append(val_loss / total)
    val_accuracies.append(correct / total)

    print(f"Época {epoch+1}/{NUM_EPOCHS} - "
          f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f}, Acc: {val_accuracies[-1]:.4f}")

    # Log de métricas básicas
    wandb.log({
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_acc": train_accuracies[-1],
        "val_acc": val_accuracies[-1],
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })




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
