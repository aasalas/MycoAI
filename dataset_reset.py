import os
import shutil
import glob # Para encontrar todos los archivos de imagen

def rearrange_micorriza_folders(base_dataset_dir, class_folders=['Ectomicorriza', 'Endomicorriza', 'Ecten- domicorriza']):
    """
    Reorganiza las carpetas de micorrizas, moviendo todas las imágenes de subcarpetas
    profundas a sus respectivas carpetas de clase de nivel superior.

    Args:
        base_dataset_dir (str): La ruta base del dataset (ej. 'C:\\Users\\...\\Micorrizas-DataSet').
        class_folders (list): Lista de nombres de las carpetas de clases principales.
    """

    # Extensiones de imagen soportadas
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    print(f"Iniciando la reorganización en: {base_dataset_dir}")

    for class_folder_name in class_folders:
        class_folder_path = os.path.join(base_dataset_dir, class_folder_name)

        if not os.path.isdir(class_folder_path):
            print(f"Advertencia: La carpeta de clase '{class_folder_path}' no existe. Saltando.")
            continue

        print(f"\nProcesando la clase: '{class_folder_name}'")
        
        # Lista para almacenar las rutas de las subcarpetas que pueden quedar vacías
        subfolders_to_clean = []

        # Recorrer todos los subdirectorios y archivos dentro de la carpeta de clase
        for dirpath, dirnames, filenames in os.walk(class_folder_path):
            # Asegurarse de no procesar la carpeta de clase raíz en esta parte
            if dirpath == class_folder_path:
                continue

            # Añadir esta carpeta a la lista para posible limpieza si es una subcarpeta
            subfolders_to_clean.append(dirpath)

            for filename in filenames:
                if filename.lower().endswith(allowed_extensions):
                    old_file_path = os.path.join(dirpath, filename)

                    # Crear un nombre de archivo único para evitar colisiones
                    # Usa la ruta relativa desde la carpeta de la clase para renombrar
                    relative_path = os.path.relpath(dirpath, class_folder_path)
                    if relative_path == ".": # Si el archivo ya está directamente en la carpeta de clase
                        new_file_name = filename
                    else:
                        # Reemplaza separadores de ruta por guiones bajos para el nuevo nombre
                        unique_prefix = relative_path.replace(os.sep, '_')
                        new_file_name = f"{unique_prefix}_{filename}"
                    
                    new_file_path = os.path.join(class_folder_path, new_file_name)

                    # Si el archivo con el nuevo nombre ya existe (poco probable con el prefijo, pero posible)
                    # Añade un contador para asegurar unicidad
                    counter = 1
                    original_new_file_path = new_file_path
                    while os.path.exists(new_file_path):
                        name, ext = os.path.splitext(original_new_file_path)
                        new_file_path = f"{name}_{counter}{ext}"
                        counter += 1

                    try:
                        shutil.move(old_file_path, new_file_path)
                        # print(f"Movido: '{old_file_path}' -> '{new_file_path}'")
                    except Exception as e:
                        print(f"Error al mover '{old_file_path}': {e}")
            
        # Limpiar subcarpetas vacías, de las más anidadas a las menos
        # Ordenar en orden inverso de longitud para eliminar de las más profundas a las menos
        subfolders_to_clean.sort(key=len, reverse=True)
        print(f"Intentando limpiar {len(subfolders_to_clean)} subcarpetas vacías en '{class_folder_name}'...")
        for folder_to_remove in subfolders_to_clean:
            try:
                # Comprobar si la carpeta está realmente vacía antes de eliminar
                if not os.listdir(folder_to_remove):
                    os.rmdir(folder_to_remove)
                    # print(f"Eliminado carpeta vacía: '{folder_to_remove}'")
            except OSError as e:
                # Esto puede ocurrir si la carpeta no está vacía o hay problemas de permisos
                # print(f"No se pudo eliminar '{folder_to_remove}': {e}")
                pass # Ignorar errores si la carpeta no está vacía o hay permisos

    print("\nReorganización completada. Por favor, verifica tu estructura de carpetas.")
    print(f"Ahora, la ruta para 'data_dir' en tus scripts de PyTorch debería ser: {base_dataset_dir}")
    print("Asegúrate de que no queden carpetas o archivos no deseados (como '.git') en este directorio raíz.")


# --- Configuración principal ---
if __name__ == "__main__":
    # Define la ruta principal de tu dataset
    # ASEGÚRATE de que esta ruta sea la CORRECTA para ti
    dataset_root = r'C:\Users\Lenovo Yoga\Desktop\DATASET\Micorrizas-DataSet'
    
    # Llama a la función para reorganizar las carpetas
    rearrange_micorriza_folders(dataset_root)