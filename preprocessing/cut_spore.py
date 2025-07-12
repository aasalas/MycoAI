# segment_all.py (modificado para recortar esporas a mayor resolución)

import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
# Asegúrate de que unet.py esté accesible para la importación
from unet import UNet

# -------------- CONFIGURACIÓN --------------------
MODEL_PATH = "unet_model.pth"
INPUT_DIR = "./dataset_bc"
OUTPUT_DIR = "./dataset_bc_segmented_high_res_crops" # Nuevo directorio de salida
MODEL_INPUT_SIZE = (256, 256) # Tamaño de entrada para el modelo UNet
OUTPUT_IMAGE_SIZE = (640, 640) # Tamaño de la imagen de salida y de la máscara redimensionada
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valores de normalización del entrenamiento (CRÍTICO: deben ser los mismos que usaste para entrenar)
MEAN = [0.7590, 0.6850, 0.5364]
STD = [0.2145, 0.2137, 0.2792]

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar modelo
model = UNet().to(DEVICE)
model.load(MODEL_PATH, map_location=DEVICE)
model.eval()

print(f"Iniciando segmentación de esporas para recorte a {OUTPUT_IMAGE_SIZE[0]}x{OUTPUT_IMAGE_SIZE[1]}.")
print(f"Las imágenes de entrada al modelo son redimensionadas a {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}.")
print(f"Resultados guardados en: {OUTPUT_DIR}")

# Buscar imágenes en todas las subcarpetas
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)

            # Construir ruta de salida manteniendo estructura
            rel_dir = os.path.relpath(root, INPUT_DIR)
            save_subdir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(save_subdir, exist_ok=True)
            save_path = os.path.join(save_subdir, file) # El nombre del archivo sigue siendo el mismo

            # 1. Cargar la imagen original a su RESOLUCIÓN NATIVA (o a OUTPUT_IMAGE_SIZE si es preferible)
            # Para la salida de alta calidad, queremos empezar con la mejor resolución posible.
            # Aquí la cargamos y la redimensionamos directamente a OUTPUT_IMAGE_SIZE
            high_res_img_pil = Image.open(input_path).convert('RGB').resize(OUTPUT_IMAGE_SIZE, Image.LANCZOS)
            # Image.LANCZOS es un filtro de alta calidad para redimensionar (mejor que el default)

            # 2. Preparar la imagen para la entrada del MODELO (redimensionar a MODEL_INPUT_SIZE)
            img_for_model_pil = high_res_img_pil.resize(MODEL_INPUT_SIZE) # Redimensiona para el modelo
            tensor = TF.to_tensor(img_for_model_pil)
            tensor = TF.normalize(tensor, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)

            # 3. Realizar la predicción con el modelo
            with torch.no_grad():
                output = model(tensor)
                # Aplicar sigmoid si tu UNet no lo hace en la capa final para obtener probabilidades
                # output_probs = torch.sigmoid(output) 
                
                # Obtener máscara binaria a la resolución del modelo (MODEL_INPUT_SIZE)
                pred_mask_low_res = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) # Esto es 0 o 1

            # 4. Redimensionar la máscara predicha a la RESOLUCIÓN DE SALIDA deseada
            # Usar INTERPOLACIÓN NEAREST para máscaras binarias para evitar valores intermedios (grises)
            pred_mask_pil = Image.fromarray(pred_mask_low_res * 255).convert('L') # Convertir a PIL para redimensionar
            pred_mask_high_res = np.array(pred_mask_pil.resize(OUTPUT_IMAGE_SIZE, Image.NEAREST)) > 0 # Redimensionar y binarizar de nuevo

            # 5. Aplicar la máscara de alta resolución a la imagen original de alta resolución
            high_res_img_np = np.array(high_res_img_pil)

            # Recortar la espora usando la máscara de alta resolución
            # Multiplica cada canal RGB por la máscara binaria (extendida a 3 canales)
            cropped_img_np = high_res_img_np * np.stack([pred_mask_high_res] * 3, axis=-1) 

            # Opcional: Si quieres un fondo blanco en lugar de negro:
            # background = np.full(high_res_img_np.shape, 255, dtype=np.uint8) # Fondo blanco
            # cropped_img_np = np.where(np.stack([pred_mask_high_res] * 3, axis=-1), high_res_img_np, background)
            
            # Convertir el array NumPy resultante de nuevo a PIL Image
            cropped_img_pil = Image.fromarray(cropped_img_np)

            # 6. Guardar la imagen recortada de alta calidad
            # PIL guarda con la calidad predeterminada de PNG (sin pérdida) o JPEG (comprimido)
            # Puedes especificar 'quality' para JPEG si lo guardas como '.jpg'
            # Por ejemplo: cropped_img_pil.save(save_path.replace('.png', '.jpg'), quality=95)
            cropped_img_pil.save(save_path) # Guarda como PNG o el formato original del archivo

print(f"✅ Segmentación y recorte completados. Esporas recortadas de {OUTPUT_IMAGE_SIZE[0]}x{OUTPUT_IMAGE_SIZE[1]} guardadas en: {OUTPUT_DIR}")