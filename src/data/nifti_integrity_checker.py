import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import numpy as np

TARGET_DIR = Path("data/raw/nnUNet_raw/Dataset101_KiTS23/imagesTr")

def check_files():
    files = list(TARGET_DIR.glob("*.nii.gz"))
    print(f"Verificando {len(files)} archivos en {TARGET_DIR}...")
    
    issues = []
    
    for f in tqdm(files):
        try:
            img = nib.load(f)
            header = img.header
            data_shape = header.get_data_shape()
            
            # Chequeos básicos
            if len(data_shape) != 3:
                issues.append(f"{f.name}: Dimensiones incorrectas {data_shape}")
            
            if np.any(np.array(data_shape) == 0):
                issues.append(f"{f.name}: Dimensión cero detectada")
                
        except Exception as e:
            issues.append(f"{f.name}: Error al leer - {e}")
            
    if issues:
        print("\n⚠️ SE ENCONTRARON PROBLEMAS:")
        for i in issues:
            print(i)
    else:
        print("\n✅ Todos los archivos parecen válidos (encabezados legibles y 3D).")

if __name__ == "__main__":
    check_files()