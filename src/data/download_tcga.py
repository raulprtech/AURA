import os
import pandas as pd
from tcia_utils import nbia
import json

# Configuración
DATASET_NAME = "TCGA-KIRC"
DOWNLOAD_PATH = "data/raw/TCGA-KIRC/images"

# Crear directorio si no existe
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

print(f"Iniciando descarga de la colección {DATASET_NAME}...")

# 1. Obtener la lista de series (scans) disponibles
try:
    series_data = nbia.getSeries(
        collection=DATASET_NAME,
        modality="CT"
    )
except Exception as e:
    print(f"Error al conectar con TCIA: {e}")
    exit(1)

# Diagnóstico: Imprimir el tipo de dato recibido
print(f"Tipo de datos recibido de getSeries: {type(series_data)}")
if isinstance(series_data, list) and len(series_data) > 0:
    print(f"Primer elemento: {series_data[0]} (Tipo: {type(series_data[0])})")

# 2. Procesar los UIDs de las series
series_uids = []

if isinstance(series_data, list):
    if len(series_data) == 0:
        print("No se encontraron series para los criterios dados.")
        exit()
        
    first_item = series_data[0]
    
    # CASO A: Es una lista de diccionarios (Formato esperado habitual)
    if isinstance(first_item, dict):
        print("Formato detectado: Lista de Diccionarios")
        # Convertir a DataFrame para facilitar el manejo y visualización
        df = pd.DataFrame(series_data)
        print(f"Se encontraron {len(df)} series de CT.")
        if 'SeriesInstanceUID' in df.columns:
            series_uids = df['SeriesInstanceUID'].tolist()
        else:
            print("Error: La columna 'SeriesInstanceUID' no se encuentra en los datos.")
            print("Columnas disponibles:", df.columns)
            exit(1)
            
    # CASO B: Es una lista de strings (Formato simple)
    elif isinstance(first_item, str):
        print("Formato detectado: Lista de Strings (UIDs directos)")
        series_uids = series_data
        print(f"Se encontraron {len(series_uids)} series de CT.")
        
    else:
        print(f"Formato de datos desconocido: {type(first_item)}")
        exit(1)
else:
    print("La respuesta de la API no es una lista.")
    exit(1)

# 3. Descargar las imágenes
print(f"\nPreparando descarga de {len(series_uids)} series...")

BATCH_SIZE = 5 # Reducido para mejor control inicial
total_batches = (len(series_uids) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(total_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, len(series_uids))
    batch_uids = series_uids[start_idx : end_idx]
    
    print(f"Descargando lote {i+1}/{total_batches} (Series {start_idx+1} a {end_idx})...")
    
    try:
        # La función downloadSeries espera una lista de strings (UIDs)
        # Nota: input_type="list" es el valor por defecto, pero ser explícito ayuda
        nbia.downloadSeries(batch_uids, input_type="list", path=DOWNLOAD_PATH)
    except Exception as e:
        print(f" Error descargando lote {i+1}: {e}")
        # Opcional: Guardar los UIDs fallidos en un log
        with open("failed_downloads.log", "a") as f:
            for uid in batch_uids:
                f.write(f"{uid}\n")
        continue

print("\n¡Proceso de descarga finalizado!")