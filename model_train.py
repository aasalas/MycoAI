import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


from dataloader import MycoDataLoader
from model import MycoModel

sweep_config = {
    'method': 'grid',  # Método de búsqueda: 'random' (aleatorio), 'grid' (malla), 'bayes' (bayesiano)
    'metric': {          # Métrica a optimizar durante el sweep
        'name': 'f1_macro', # La recompensa media por episodio es una métrica estándar de SB3
        'goal': 'maximize'             # Queremos maximizar esta métrica
    },
    'parameters': {      # Definición de los hiperparámetros y sus valores/rangos
        'img_size': {
            'values': [256, 320] # Valores específicos a probar para el tamaño de imagen
        },
        'batch_size': {
            'values': [2, 4] # Valores específicos a probar para
        },
        'learning_rate': {
            'values': [0.00001, 0.0001] # Valores específicos a probar para
        },
        'epoch_num': {
            'values': [3, 5] # Valores específicos a probar para
        },
        'arch': {
            'values': ["mobilenet","resnet50"] # Valores específicos a probar para
        }
    }
}
# --- Parámetros ---
DATA_DIR = r'/content/drive/MyDrive/preprocessing/dataset_bc_segmented'
SAVE_PATH = 'MycoModel.pth' # Nombre para guardar el modelo entrenado

best_f1 = 0.0  # Para guardar solo el mejor modelo

# Inicializa wandb

def train():
    global best_f1
    wandb.login(key="ce64ccc4ab0dc13513de5af31b6086e12830f6e9") #CUENTA DE ALVARO
    wandb.init(project=f"MycoAI-Classifier")
    config = wandb.config
    IMG_SIZE = (config.img_size,config.img_size)
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    NUM_EPOCHS = config.epoch_num
    ARCHITECTURE = config.arch
    wandb.run.name =f"{ARCHITECTURE}_img{IMG_SIZE[0]}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_epochs{NUM_EPOCHS}_{wandb.util.generate_id()[:4]}"
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

        f1_macro = f1_score(all_labels, all_preds, average='macro')

        wandb.log({
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
            "train_acc": train_accuracies[-1],
            "val_acc": val_accuracies[-1],
            "f1_macro": f1_macro, 
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })

        # --- Guardar el mejor modelo basado en F1 Macro ---
        if f1_macro > best_f1:
            best_f1 = f1_macro
            model_builder.save(SAVE_PATH)
            print(f"✅ Nuevo mejor modelo guardado con F1 Macro: {best_f1:.4f}")








if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="MycoAI-Classifier")
    wandb.agent(sweep_id, function=train, count=27)