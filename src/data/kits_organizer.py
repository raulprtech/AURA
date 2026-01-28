import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Ajusta la ruta a donde tengas DESCOMPRIMIDO tu dataset KiTS23
# Según tu imagen, parece que tienes una carpeta llamada "kits23" y dentro las carpetas de casos.
RAW_KITS_DIR = Path("scripts/kits23/dataset") 

# Ruta de destino para nnU-Net (Dataset101_KiTS23)
NNUNET_RAW_DIR = Path("data/raw/nnUNet_raw/Dataset101_KiTS23")
# ---------------------

def create_dataset_json(output_dir, num_training_cases):
    """
    Crea el archivo dataset.json requerido por nnU-Net.
    """
    json_dict = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "kidney": 1,
            "tumor": 2,
            "cyst": 3
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": "KiTS23",
        "reference": "KiTS23 Challenge",
        "release": "1.0",
        "description": "Kidney Tumor Segmentation Challenge 2023",
        "tensorImageSize": "3D", 
        "modality": { 
            "0": "CT" 
        } 
    }
    
    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f"✅ Archivo dataset.json creado en: {json_path}")

def main():
    # 1. Verificar directorios
    if not RAW_KITS_DIR.exists():
        print(f"❌ Error: No se encuentra el directorio de origen: {RAW_KITS_DIR}")
        return

    # 2. Crear estructura de destino
    imagesTr_dir = NNUNET_RAW_DIR / "imagesTr"
    labelsTr_dir = NNUNET_RAW_DIR / "labelsTr"
    
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Estructura nnU-Net creada en: {NNUNET_RAW_DIR}")

    # 3. Buscar casos de forma recursiva
    print(f"🔍 Buscando carpetas 'case_*' en {RAW_KITS_DIR}...")
    
    # Buscamos recursivamente cualquier carpeta que empiece con "case_"
    # Esto encontrará data/raw/kits23/case_00000 o data/raw/kits23/dataset/case_00000, etc.
    case_folders = sorted(list(RAW_KITS_DIR.rglob("case_*")))
    
    # Filtramos para asegurarnos de que son directorios y no archivos
    case_folders = [f for f in case_folders if f.is_dir()]
    
    if not case_folders:
        print("⚠️ No se encontraron carpetas 'case_*'. Verifica la descarga.")
        return
        
    print(f"✅ Se encontraron {len(case_folders)} casos potenciales.")
    
    processed_count = 0
    
    # 4. Procesar y copiar archivos
    for case_dir in tqdm(case_folders, desc="Procesando casos"):
        case_id = case_dir.name # "case_00000"
        
        # Archivos origen esperados
        src_image = case_dir / "imaging.nii.gz"
        src_label = case_dir / "segmentation.nii.gz"
        
        # Verificar existencia
        if not src_image.exists():
            # A veces los archivos se llaman diferente, probamos variantes
            candidates = list(case_dir.glob("imaging*.nii*"))
            if candidates:
                src_image = candidates[0]
            else:
                continue # Saltamos si no hay imagen
                
        if not src_label.exists():
            # Si no hay label, lo saltamos (podría ser un caso de test sin label pública)
            continue

        # Nombres destino (Formato nnU-Net)
        # Imagen: case_00000_0000.nii.gz (IMPORTANTE: el _0000 al final)
        dst_image_name = f"{case_id}_0000.nii.gz"
        # Label: case_00000.nii.gz
        dst_label_name = f"{case_id}.nii.gz"
        
        dst_image_path = imagesTr_dir / dst_image_name
        dst_label_path = labelsTr_dir / dst_label_name
        
        # Copiar archivos
        try:
            # Usamos copy2 para preservar metadatos
            if not dst_image_path.exists():
                shutil.copy2(src_image, dst_image_path)
            if not dst_label_path.exists():
                shutil.copy2(src_label, dst_label_path)
            processed_count += 1
        except Exception as e:
            print(f"Error copiando caso {case_id}: {e}")

    print(f"\n✅ Procesamiento completado. {processed_count} casos organizados correctamente en {NNUNET_RAW_DIR}.")
    
    # 5. Generar dataset.json
    if processed_count > 0:
        create_dataset_json(NNUNET_RAW_DIR, processed_count)
    else:
        print("❌ No se generó dataset.json porque no se procesaron casos.")

if __name__ == "__main__":
    main()