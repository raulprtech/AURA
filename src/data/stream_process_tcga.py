import os
import shutil
import pandas as pd
from tcia_utils import nbia
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import time

# --- CONFIGURACIÓN ---
DATASET_NAME = "TCGA-KIRC"
# Directorios temporales y finales
TEMP_DICOM_DIR = Path("data/temp_dicom")  # Aquí descargamos temporalmente
OUTPUT_NIFTI_DIR = Path("data/raw/nnUNet_raw/Dataset102_TCGA/imagesTr")
# Archivo de registro para saber qué ya procesamos
PROCESSED_LOG = Path("data/processed_series.log")

# Asegurar directorios
TEMP_DICOM_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_NIFTI_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------

def load_processed_series():
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_series_as_processed(series_uid):
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{series_uid}\n")

def convert_dicom_series(series_dir, output_path):
    """Convierte una serie DICOM a NIfTI usando SimpleITK."""
    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_names:
            return False
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_path))
        return True
    except Exception as e:
        print(f"⚠️ Error convirtiendo {series_dir}: {e}")
        return False

def get_patient_id(series_dir):
    """Intenta extraer el PatientID de los archivos DICOM."""
    try:
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files: return None
        # Leemos solo los tags del primer archivo
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(dcm_files[0]))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        return reader.GetMetaData("0010|0020").strip()
    except:
        return None

def process_batch(series_uids):
    """Descarga, convierte y borra un lote de series."""
    
    # 1. Descargar Lote
    print(f"⬇️ Descargando lote de {len(series_uids)} series...")
    try:
        # Descargamos en la carpeta temporal
        nbia.downloadSeries(series_uids, input_type="list", path=TEMP_DICOM_DIR)
    except Exception as e:
        print(f"❌ Error en descarga de lote: {e}")
        return

    # 2. Convertir Lote
    print("🔄 Convirtiendo a NIfTI...")
    # Buscamos las carpetas de series descargadas (TCIA crea una estructura anidada)
    # Buscamos recursivamente cualquier carpeta que tenga .dcm
    downloaded_series_dirs = set()
    for f in TEMP_DICOM_DIR.rglob("*.dcm"):
        downloaded_series_dirs.add(f.parent)
    
    for series_dir in downloaded_series_dirs:
        # Identificar Paciente
        patient_id = get_patient_id(series_dir)
        if not patient_id:
            patient_id = "UNKNOWN" # Fallback
            
        # Definir nombre de salida (formato nnU-Net: ID_0000.nii.gz)
        # Usamos el SeriesInstanceUID como parte del nombre para evitar colisiones si un paciente tiene varios scans
        # Pero nnU-Net espera ID_0000. Si hay múltiples, hay que decidir.
        # Por ahora, guardamos como ID_SERIESUID_0000.nii.gz para no sobreescribir.
        # Luego filtraremos.
        series_uid_name = series_dir.name
        output_filename = f"{patient_id}_{series_uid_name}_0000.nii.gz"
        output_path = OUTPUT_NIFTI_DIR / output_filename
        
        if convert_dicom_series(series_dir, output_path):
            print(f"✅ Convertido: {output_filename}")
        else:
            print(f"❌ Falló conversión: {series_dir}")

    # 3. Limpiar (Borrar DICOMs)
    print("🧹 Limpiando archivos temporales...")
    # Borramos todo el contenido de TEMP_DICOM_DIR
    for item in TEMP_DICOM_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
            
    # Marcar como procesados en el log
    for uid in series_uids:
        mark_series_as_processed(uid)

def main():
    print(f"🚀 Iniciando Flujo de Procesamiento {DATASET_NAME}...")
    
    # 1. Obtener lista completa de series
    try:
        series_data = nbia.getSeries(collection=DATASET_NAME, modality="CT")
        # Manejo robusto de la respuesta (como hicimos antes)
        if isinstance(series_data, list) and len(series_data) > 0:
            if isinstance(series_data[0], dict):
                all_series_uids = [item['SeriesInstanceUID'] for item in series_data]
            elif isinstance(series_data[0], str):
                all_series_uids = series_data
            else:
                raise ValueError("Formato desconocido de TCIA")
        else:
            print("No se encontraron series.")
            return
    except Exception as e:
        print(f"Error conectando a TCIA: {e}")
        return

    # 2. Filtrar ya procesados
    processed = load_processed_series()
    pending_uids = [uid for uid in all_series_uids if uid not in processed]
    
    print(f"Total series: {len(all_series_uids)}")
    print(f"Ya procesadas: {len(processed)}")
    print(f"Pendientes: {len(pending_uids)}")
    
    if not pending_uids:
        print("¡Todo está al día!")
        return

    # 3. Procesar por Lotes (Batch)
    BATCH_SIZE = 5 # Tamaño pequeño para controlar el espacio en disco
    total_batches = (len(pending_uids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(total_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(pending_uids))
        batch = pending_uids[start:end]
        
        print(f"\n--- Procesando Lote {i+1}/{total_batches} ---")
        process_batch(batch)
        
        # Pausa breve para no saturar
        time.sleep(1)

    print("\n🎉 ¡Misión cumplida! Todos los datos han sido procesados.")

if __name__ == "__main__":
    main()