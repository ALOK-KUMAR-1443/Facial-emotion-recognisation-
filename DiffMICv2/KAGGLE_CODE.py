# ============================================================================
# FINAL KAGGLE CODE FOR DiffMICv2 (Train 80% + Val 20% + Test)
# Copy each cell into your Kaggle notebook
# ============================================================================

# ----------------------------------------------------------------------------
# CELL 1: Clone repository
# ----------------------------------------------------------------------------
!git clone https://github.com/ALOK-KUMAR-1443/Facial-emotion-recognisation-.git
%cd Facial-emotion-recognisation-/DiffMICv2

# ----------------------------------------------------------------------------
# CELL 2: Verify dataset
# ----------------------------------------------------------------------------
import os
DATASET_PATH = '/kaggle/input/your-dataset-name/DATASET'  # ⚠️ UPDATE THIS
print("Dataset folders:", os.listdir('/kaggle/input'))
if os.path.exists(DATASET_PATH):
    print("✓ Dataset found!")
    print("Train classes:", os.listdir(f'{DATASET_PATH}/train'))
    print("Test classes:", os.listdir(f'{DATASET_PATH}/test'))
else:
    print("❌ Update DATASET_PATH in this cell")

# ----------------------------------------------------------------------------
# CELL 3: Install dependencies
# ----------------------------------------------------------------------------
!pip install -q pytorch-lightning einops timm pyyaml opencv-python-headless albumentations scikit-learn

# ----------------------------------------------------------------------------
# CELL 4: Clone EfficientSAM
# ----------------------------------------------------------------------------
!git clone https://github.com/yformer/EfficientSAM.git

# ----------------------------------------------------------------------------
# CELL 5: Update config (dataset path + 80/20 split)
# ----------------------------------------------------------------------------
import yaml, os
DATASET_PATH = '/kaggle/input/your-dataset-name/DATASET'  # ⚠️ UPDATE THIS

with open('configs/placental.yml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['dataroot'] = DATASET_PATH
config['data']['val_split'] = 0.2  # 20% validation
config['training']['batch_size'] = 8  # Optimize for GPU
config['training']['n_epochs'] = 50  # Reduce epochs

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
with open('configs/placental.yml', 'w') as f:
    yaml.dump(config, f)

print(f"✓ Dataset: {DATASET_PATH}")
print("✓ Train/Val split: 80%/20%")
print(f"✓ Batch: {config['training']['batch_size']}, Epochs: {config['training']['n_epochs']}")

# ----------------------------------------------------------------------------
# CELL 6: Check GPU
# ----------------------------------------------------------------------------
import torch
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
# CELL 7: 🚀 TRAIN MODEL (auto 80/20 split)
# ----------------------------------------------------------------------------
print("="*70)
print("TRAINING: 80% train, 20% validation")
print("="*70)
!python diffuser_trainer.py
print("\n✓ Training complete!")

# ----------------------------------------------------------------------------
# CELL 8: View results
# ----------------------------------------------------------------------------
import torch, glob, os
checkpoints = glob.glob('/kaggle/working/checkpoints/*.ckpt')
if checkpoints:
    ckpt = torch.load(checkpoints[-1], map_location='cpu')
    print("✓ Training Complete!")
    if 'callback_metrics' in ckpt:
        for k, v in ckpt['callback_metrics'].items():
            print(f"  {k}: {v}")
else:
    print("❌ No checkpoint found")
    print("Troubleshooting:")
    print("1. Check for OOM errors in Cell 7")
    print("2. Reduce batch_size to 4 in Cell 5")
    print("3. Enable GPU in Kaggle settings")
