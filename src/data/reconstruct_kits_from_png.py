import os
import shutil
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import re

# --- CONFIGURACIÓN ---
# Ajusta la ruta a donde tengas DESCOMPRIMIDO tu dataset KiTS23 (con PNGs)
RAW_KITS_DIR = Path("data/raw/kits23") 

# Ruta de destino para nnU-Net (Dataset101_KiTS23)
NNUNET_RAW_DIR = Path("data/raw/nnUNet_raw/Dataset101_KiTS23")
# ---------------------

def create_dataset_json(output_dir, num_training_cases):
    """Crea el archivo dataset.json requerido por nnU-Net."""
    json_dict = {
        "channel_names": { "0": "CT" },
        "labels": { "background": 0, "kidney": 1, "tumor": 2, "cyst": 3 },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": "KiTS23",
        "reference": "KiTS23 Challenge",
        "tensorImageSize": "3D", 
        "modality": { "0": "CT" } 
    }
    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        import json
        json.dump(json_dict, f, indent=4)
    print(f"✅ Archivo dataset.json creado en: {json_path}")

def get_slice_index(filename):
    """Extrae el número de slice del nombre del archivo.
       Ejemplo: 'slice_case_00000_90.png' -> 90
    """
    # Buscamos el último número en el nombre del archivo antes de la extensión
    match = re.search(r'_(\d+)(?:_mask)?\.(png|jpg|jpeg)$', filename.name)
    if match:
        return int(match.group(1))
    return -1

def reconstruct_volume(image_files, mask_files=None):
    """Lee una lista de archivos de imagen ordenados y crea un volumen 3D."""
    # Ordenar por índice de slice
    image_files.sort(key=get_slice_index)
    
    slices = []
    for img_path in image_files:
        img = Image.open(img_path).convert('L') # Convertir a escala de grises (L)
        slices.append(np.array(img))
        
    # Apilar en un volumen 3D (H, W, D) -> (X, Y, Z)
    # nnU-Net generalmente espera (X, Y, Z), donde Z son los cortes.
    # np.stack axis=0 crea (Z, H, W). axis=-1 crea (H, W, Z).
    # La convención médica suele ser (X, Y, Z). Probemos axis=-1 primero.
    volume = np.stack(slices, axis=-1) 
    
    mask_volume = None
    if mask_files:
        mask_files.sort(key=get_slice_index)
        mask_slices = []
        for mask_path in mask_files:
            mask = Image.open(mask_path).convert('L') # Asumiendo máscaras en escala de grises
            # Mapear valores si es necesario (ej. 255 -> 1)
            # KiTS23 suele usar 0, 1, 2, 3 directamente.
            mask_arr = np.array(mask)
            mask_slices.append(mask_arr)
        
        if len(mask_slices) != len(slices):
            print(f"⚠️ Advertencia: Número de slices de imagen ({len(slices)}) y máscara ({len(mask_slices)}) no coinciden.")
            return None, None
            
        mask_volume = np.stack(mask_slices, axis=-1)
        
    return volume, mask_volume

def main():
    # 1. Crear estructura de destino
    imagesTr_dir = NNUNET_RAW_DIR / "imagesTr"
    labelsTr_dir = NNUNET_RAW_DIR / "labelsTr"
    
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Estructura nnU-Net creada en: {NNUNET_RAW_DIR}")

    # 2. Buscar carpetas de casos
    # Buscamos carpetas que contengan "JPEGImages" y "Annotations"
    print(f"🔍 Buscando casos en {RAW_KITS_DIR}...")
    potential_cases = sorted(list(RAW_KITS_DIR.rglob("case_*")))
    valid_cases = []
    
    for case_dir in potential_cases:
        if not case_dir.is_dir(): continue
        # Verificamos si tiene las subcarpetas de PNGs
        if (case_dir / "JPEGImages").exists() and (case_dir / "Annotations").exists():
            valid_cases.append(case_dir)
            
    if not valid_cases:
        print("⚠️ No se encontraron carpetas de casos con estructura PNG (JPEGImages/Annotations).")
        return

    print(f"✅ Se encontraron {len(valid_cases)} casos para reconstruir.")
    
    processed_count = 0
    
    # 3. Procesar cada caso
    for case_dir in tqdm(valid_cases, desc="Reconstruyendo volúmenes"):
        case_id = case_dir.name # "case_00000"
        
        img_dir = case_dir / "JPEGImages"
        mask_dir = case_dir / "Annotations"
        
        img_files = list(img_dir.glob("*.png"))
        mask_files = list(mask_dir.glob("*.png"))
        
        if not img_files:
            continue

        # Reconstruir 3D
        vol, mask = reconstruct_volume(img_files, mask_files)
        
        if vol is None:
            continue
            
        # Crear objetos NIfTI
        # IMPORTANTE: Al reconstruir desde PNGs, perdemos la información espacial original (spacing, origin, direction).
        # Usaremos una matriz identidad por defecto. nnU-Net remuestreará esto después, 
        # pero idealmente deberíamos tener el spacing original.
        affine = np.eye(4) 
        nifti_img = nib.Nifti1Image(vol, affine)
        
        # Guardar Imagen
        dst_image_name = f"{case_id}_0000.nii.gz"
        nib.save(nifti_img, imagesTr_dir / dst_image_name)
        
        # Guardar Máscara
        if mask is not None:
            nifti_mask = nib.Nifti1Image(mask, affine)
            dst_label_name = f"{case_id}.nii.gz"
            nib.save(nifti_mask, labelsTr_dir / dst_label_name)
            
        processed_count += 1

    print(f"\n✅ Reconstrucción completada. {processed_count} volúmenes creados.")
    
    if processed_count > 0:
        create_dataset_json(NNUNET_RAW_DIR, processed_count)

if __name__ == "__main__":
    main()