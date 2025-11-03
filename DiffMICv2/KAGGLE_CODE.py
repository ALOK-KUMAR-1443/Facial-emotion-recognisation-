# ============================================================================
# FINAL KAGGLE CODE FOR DiffMICv2 (Train 80% + Val 20% + Test)
# Copy each cell into your Kaggle notebook
# ============================================================================

# ----------------------------------------------------------------------------
# CELL 1: Clone repository
# ----------------------------------------------------------------------------
!git clone https://github.com/ALOK-KUMAR-1443/Facial-emotion-recognisation-.git
import os
os.chdir('Facial-emotion-recognisation-/DiffMICv2')
print(f"✓ Current directory: {os.getcwd()}")

# ----------------------------------------------------------------------------
# CELL 2: Verify dataset
# ----------------------------------------------------------------------------
import os
DATASET_PATH = '/kaggle/input/raf-db-dataset/DATASET'
print("Dataset folders:", os.listdir('/kaggle/input'))
if os.path.exists(DATASET_PATH):
    print("✓ Dataset found!")
    print("Train classes:", os.listdir(f'{DATASET_PATH}/train'))
    print("Test classes:", os.listdir(f'{DATASET_PATH}/test'))
else:
    print("❌ Dataset not found. Please check path.")

# ----------------------------------------------------------------------------
# CELL 3: Install required dependencies
# ----------------------------------------------------------------------------
print("Installing required packages (Kaggle-compatible)...")

# Install only missing/outdated packages
!pip install peft>=0.17.0 diffusers>=0.20.0 accelerate einops ema-pytorch easydict -q

# Verify versions
import numpy as np
import torch
print(f"\n✓ NumPy version: {np.__version__}")
print(f"✓ PyTorch version: {torch.__version__}")
print("✓ All dependencies installed (Kaggle-compatible)")

# ----------------------------------------------------------------------------
# CELL 4: Clone EfficientSAM
# ----------------------------------------------------------------------------
!git clone https://github.com/yformer/EfficientSAM.git

# ----------------------------------------------------------------------------
# CELL 5: Update config (dataset path + 80/20 split)
# ----------------------------------------------------------------------------
import yaml, os
DATASET_PATH = '/kaggle/input/raf-db-dataset/DATASET'

with open('configs/placental.yml', 'r') as f:
    config = yaml.safe_load(f)

# Update for folder-based dataset (Kaggle)
config['data']['dataroot'] = DATASET_PATH
config['data']['use_folder_structure'] = True  # Enable folder structure
config['data']['val_split'] = 0.2  # 20% validation
config['training']['batch_size'] = 8  # Optimize for GPU
config['training']['n_epochs'] = 50  # Reduce epochs
config['data']['num_workers'] = 2  # Reduce workers for Kaggle

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
with open('configs/placental.yml', 'w') as f:
    yaml.dump(config, f)

print(f"✓ Dataset: {DATASET_PATH}")
print("✓ Using folder structure (Kaggle mode)")
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
# CELL 8: Training complete
# ----------------------------------------------------------------------------
print("="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)
print("\n📊 Check the output above (Cell 7) for:")
print("  - Training loss per epoch")
print("  - Validation metrics (accuracy, F1, precision, recall)")
print("  - Final model performance")
print("\n💡 Tips:")
print("  - Training metrics are logged during execution")
print("  - Best model is automatically saved")
print("  - Scroll up to see detailed epoch-by-epoch results")
print("\n" + "="*70)
