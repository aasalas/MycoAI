# Clasificador de Esporas Micorrízicas con ResNet50

![ResNet50](https://wisdomml.in/wp-content/uploads/2023/03/resnet_bannner.png)

Este proyecto implementa un **clasificador de esporas micorrízicas** utilizando una arquitectura **ResNet50** preentrenada. El objetivo es automatizar la identificación de especies micorrízicas mediante técnicas de visión por computadora y aprendizaje profundo. 

Se utilizó la arquitectura **ResNet50** de PyTorch, ajustada mediante *fine-tuning* para adaptarse al problema de clasificación binaria (dos clases). La red fue entrenada con técnicas de **data augmentation** para mejorar la generalización, dada la baja cantidad de datos.



---

## 🧬 Dataset

El conjunto de datos fue extraído del catálogo oficial del [Canadian Collection of Arbuscular Mycorrhizal Fungi (CCAMF)](https://agriculture.canada.ca/en/science/collections/canadian-collection-arbuscular-mycorrhizal-fungi-ccamf/catalogue-arbuscular-mycorrhizal-fungi-strains-available-glomeromycetes-vitro-collection#fn*).  
Contiene imágenes de dos especies pertenecientes a la familia **Glomeromycota**:

- **Rhizophagus irregularis** – 10 imágenes
- **Rhizophagus intraradices** – 8 imágenes

### 🧫 Ejemplos

| *Rhizophagus irregularis* | *Rhizophagus intraradices* |
|---------------------------|-----------------------------|
| ![Irregularis](https://agriculture.canada.ca/sites/default/files/legacy/pack/img/M_DAOM197198_M5-2.jpg) | ![Intraradices](https://agriculture.canada.ca/sites/default/files/media/images/2024-06/DAOMC-300074-2.jpg) |

---

## 🧩 Segmentación de Esporas

También desarrollamos un **segmentador de esporas micorrízicas** como paso de preprocesamiento para mejorar la clasificación.

Para entrenar este segmentador basado en la arquitectura **U-Net**, generamos manualmente las máscaras de aproximadamente **200 imágenes** de esporas de diversas especies. Este proceso permite aislar las esporas del fondo, facilitando la extracción de características relevantes para el clasificador.

📂 El desarrollo completo del modelo de segmentación y las herramientas utilizadas se encuentra en el siguiente repositorio:  
🔗 [Repositorio de segmentación – Competencia](https://github.com/achaconrosales/Competencia/tree/main)



