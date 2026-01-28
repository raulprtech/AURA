import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

# Configuración
RAW_DICOM_DIR = Path("data/raw/TCGA-KIRC/images")
OUTPUT_NIFTI_DIR = Path("data/raw/nnUNet_raw/Dataset102_TCGA/imagesTr")
# Mapeo de IDs (opcional, si quieres renombrar TCGA-B0-5083 a kirc_001)
# ID_MAPPING_FILE = "data/raw/TCGA-KIRC/id_mapping.csv" 

def convert_dicom_series(series_dir, output_path):
    """Lee una serie DICOM y la escribe como NIfTI comprimido."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    reader.SetFileNames(dicom_names)
    
    try:
        image = reader.Execute()
        # Escribir imagen. nnU-Net espera _0000.nii.gz para el canal 0
        sitk.WriteImage(image, str(output_path))
        return True
    except Exception as e:
        print(f"Error convirtiendo {series_dir}: {e}")
        return False

def main():
    # Crear carpeta de salida si no existe
    OUTPUT_NIFTI_DIR.mkdir(parents=True, exist_ok=True)
    
    # Encontrar todas las carpetas de series (asumiendo estructura de TCIA)
    # La estructura típica es: PatientID / StudyUID / SeriesUID / *.dcm
    print("Buscando series DICOM...")
    # Buscamos recursivamente cualquier directorio que contenga archivos .dcm
    series_dirs = set()
    for file in RAW_DICOM_DIR.rglob("*.dcm"):
        series_dirs.add(file.parent)
    
    print(f"Se encontraron {len(series_dirs)} series para convertir.")
    
    successful_conversions = 0
    
    for series_dir in tqdm(series_dirs, desc="Convirtiendo"):
        # Extraer ID del paciente de la ruta (esto depende de cómo TCIA guardó los datos)
        # Asumimos que el nombre de la carpeta abuela es el ID del paciente, o lo extraemos del DICOM
        # Para ser robustos, leemos el primer DICOM para sacar el PatientID real
        try:
            first_dcm = next(series_dir.glob("*.dcm"))
            tags = sitk.ReadImage(str(first_dcm))
            patient_id = tags.GetMetaData("0010|0020").strip() # Tag Patient ID
        except:
            patient_id = series_dir.parent.parent.name # Fallback a nombre de carpeta
            
        # Nombre de salida compatible con nnU-Net: CASO_MODALIDAD.nii.gz
        # Ejemplo: TCGA_B0_5083_0000.nii.gz
        output_filename = f"{patient_id}_0000.nii.gz"
        output_path = OUTPUT_NIFTI_DIR / output_filename
        
        # Evitar re-convertir si ya existe
        if output_path.exists():
            continue
            
        if convert_dicom_series(series_dir, output_path):
            successful_conversions += 1

    print(f"\nProceso finalizado. {successful_conversions} volúmenes convertidos correctamente.")

if __name__ == "__main__":
    main()