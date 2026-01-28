import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Ajusta la ruta a donde tengas el repositorio oficial de KiTS23
# Usamos rutas relativas desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
RAW_KITS_DIR = (PROJECT_ROOT / "scripts/kits23/dataset").resolve()
NNUNET_RAW_DIR = (PROJECT_ROOT / "data/raw/nnUNet_raw/Dataset101_KiTS23").resolve()

print(f"Ruta base: {PROJECT_ROOT}")
print(f"Origen KiTS: {RAW_KITS_DIR}")
print(f"Destino nnU-Net: {NNUNET_RAW_DIR}")
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
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset.json", 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f"✅ dataset.json creado.")

def main():
    if not RAW_KITS_DIR.exists():
        print(f"❌ Error: No existe {RAW_KITS_DIR}")
        return

    imagesTr = NNUNET_RAW_DIR / "imagesTr"
    labelsTr = NNUNET_RAW_DIR / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    print("🔍 Buscando casos...")
    # Busqueda robusta
    case_folders = sorted(list(RAW_KITS_DIR.glob("case_*")))
    if not case_folders:
        case_folders = sorted(list(RAW_KITS_DIR.rglob("case_*")))
        case_folders = [f for f in case_folders if f.is_dir()]

    print(f"✅ Se encontraron {len(case_folders)} casos.")
    
    processed_count = 0
    
    # ⚠️ IMPORTANTE: Si tu disco ya está lleno, borra manualmente
    # la carpeta 'nnUNet_raw' antes de correr esto, o fallará al intentar copiar el primero.

    for case_dir in tqdm(case_folders, desc="Moviendo y limpiando"):
        case_id = case_dir.name 
        
        src_img = case_dir / "imaging.nii.gz"
        src_seg = case_dir / "segmentation.nii.gz"
        
        if not src_img.exists() or not src_seg.exists():
            continue 

        dst_img = imagesTr / f"{case_id}_0000.nii.gz"
        dst_seg = labelsTr / f"{case_id}.nii.gz"
        
        try:
            # 1. PROCESAR IMAGEN
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img) # Copiar
            
            # Verificar integridad básica (existencia y tamaño > 0)
            if dst_img.exists() and dst_img.stat().st_size > 0:
                os.remove(src_img) # 🗑️ BORRAR ORIGEN
            else:
                raise Exception(f"Fallo al copiar imagen {case_id}")

            # 2. PROCESAR ETIQUETA
            if not dst_seg.exists():
                shutil.copy2(src_seg, dst_seg) # Copiar

            if dst_seg.exists() and dst_seg.stat().st_size > 0:
                os.remove(src_seg) # 🗑️ BORRAR ORIGEN
            else:
                raise Exception(f"Fallo al copiar etiqueta {case_id}")

            # 3. LIMPIEZA FINAL DEL CASO
            # Si ya borramos las imágenes, borramos la carpeta del caso vacía
            # Ignoramos errores aquí por si queda algún archivo basura (ej. .DS_Store)
            shutil.rmtree(case_dir, ignore_errors=True) 

            processed_count += 1
            
        except OSError as e:
            # Si nos quedamos sin espacio justo a la mitad, paramos para no corromper datos
            if e.errno == 28: 
                print(f"\n⛔ ALTO: Espacio lleno en {case_id}. Libera espacio y reanuda.")
                break
            print(f"Error en {case_id}: {e}")
        except Exception as e:
            print(f"Error genérico en {case_id}: {e}")

    print(f"\n✨ Completado. {processed_count} casos movidos.")
    
    if processed_count > 0:
        create_dataset_json(NNUNET_RAW_DIR, processed_count)

if __name__ == "__main__":
    main()