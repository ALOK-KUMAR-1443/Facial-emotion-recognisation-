# ============================================================================
# FINAL KAGGLE CODE FOR DiffMICv2 (Train 80% + Val 20% + Test)
# Copy each cell into your Kaggle notebook
# ============================================================================

# ----------------------------------------------------------------------------
# CELL 1: Setup and Clone repository
# ----------------------------------------------------------------------------
# Clean previous runs
!rm -rf Facial-emotion-recognisation-

# Clone repository
!git clone https://github.com/ALOK-KUMAR-1443/Facial-emotion-recognisation-.git

# Navigate to correct directory (absolute path to avoid nesting)
import os
os.chdir('/kaggle/working/Facial-emotion-recognisation-/DiffMICv2')
print(f"✓ Current directory: {os.getcwd()}")

# Verify we're in the right place
if os.path.exists('diffuser_trainer.py'):
    print("✓ Found diffuser_trainer.py")
else:
    print("❌ Error: diffuser_trainer.py not found!")

# ----------------------------------------------------------------------------
# CELL 2: Verify dataset
# ----------------------------------------------------------------------------
import os
DATASET_PATH = '/kaggle/input/raf-db-dataset/DATASET'
print("Dataset folders:", os.listdir('/kaggle/input'))
if os.path.exists(DATASET_PATH):
    print("✓ Dataset found!")
    train_classes = sorted(os.listdir(f'{DATASET_PATH}/train'))
    test_classes = sorted(os.listdir(f'{DATASET_PATH}/test'))
    print(f"Train classes: {train_classes}")
    print(f"Test classes: {test_classes}")
    
    # Count images
    train_total = sum([len(os.listdir(f'{DATASET_PATH}/train/{c}')) for c in train_classes])
    test_total = sum([len(os.listdir(f'{DATASET_PATH}/test/{c}')) for c in test_classes])
    print(f"\nTotal train images: {train_total}")
    print(f"Total test images: {test_total}")
    print(f"After 80/20 split: Train={int(train_total*0.8)}, Val={int(train_total*0.2)}")
else:
    print("❌ Dataset not found. Please check path.")

# ----------------------------------------------------------------------------
# CELL 3: Fix ALL package compatibility issues
# ----------------------------------------------------------------------------
print("="*70)
print("FIXING PACKAGE COMPATIBILITY (takes ~2 minutes)...")
print("="*70)

# Step 1: Fix scipy/scikit-learn compatibility
print("\n[1/3] Fixing scipy and scikit-learn...")
!pip uninstall -y scipy scikit-learn --quiet
!pip install scipy==1.13.1 scikit-learn==1.4.2 --quiet

# Step 2: Upgrade peft (for diffusers)
print("[2/3] Upgrading peft to 0.17.0...")
!pip uninstall -y peft --quiet
!pip install peft==0.17.0 --quiet

# Step 3: Install other dependencies
print("[3/3] Installing remaining packages...")
!pip install -q diffusers>=0.20.0 accelerate einops ema-pytorch easydict

# Verify versions
print("\n" + "="*70)
print("✅ PACKAGE VERSIONS (verified working):")
print("="*70)
import numpy as np
import scipy
import sklearn
import torch
import peft
import diffusers

print(f"  NumPy: {np.__version__}")
print(f"  scipy: {scipy.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  PyTorch: {torch.__version__}")
print(f"  peft: {peft.__version__}")
print(f"  diffusers: {diffusers.__version__}")
print("="*70)
print("✅ All packages installed successfully!")

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
# CELL 7: 🚀 TRAIN MODEL (with epoch-by-epoch display)
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print("Mode: 80% Train / 20% Validation")
print("\nYou will see:")
print("  ✓ Epoch X/50, Batch Y: Loss = Z")
print("  ✓ Train Loss, Train Acc (after each epoch)")
print("  ✓ Val Loss, Val Acc (after each epoch)")
print("  ✓ '✅ New best model saved' (when improving)")
print("="*70 + "\n")

# Run training
!python diffuser_trainer.py

print("\n" + "="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)

# ----------------------------------------------------------------------------
# CELL 8: View training results and metrics
# ----------------------------------------------------------------------------
import os
import pandas as pd

print("\n" + "="*70)
print("📊 TRAINING RESULTS SUMMARY")
print("="*70)

# Find latest training run
if os.path.exists('lightning_logs'):
    versions = sorted(os.listdir('lightning_logs'))
    
    if versions:
        latest = versions[-1]
        print(f"\n✓ Latest training run: {latest}")
        
        # Try to load metrics
        metrics_file = f'lightning_logs/{latest}/metrics.csv'
        if os.path.exists(metrics_file):
            print(f"✓ Found metrics file: {metrics_file}\n")
            
            try:
                df = pd.read_csv(metrics_file)
                
                # Show validation results per epoch
                val_df = df[df['val_loss'].notna()].copy()
                
                if len(val_df) > 0:
                    print("="*70)
                    print("EPOCH-BY-EPOCH VALIDATION RESULTS:")
                    print("="*70)
                    
                    for idx, row in val_df.iterrows():
                        epoch = int(row.get('epoch', idx + 1))
                        train_loss = row.get('train_loss', 0)
                        train_acc = row.get('train_acc', 0)
                        val_loss = row.get('val_loss', 0)
                        val_acc = row.get('val_acc', 0)
                        
                        if train_loss > 0:
                            print(f"\nEpoch {epoch}:")
                            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
                            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
                    
                    # Best model
                    best_idx = val_df['val_acc'].idxmax()
                    best_acc = val_df.loc[best_idx, 'val_acc']
                    best_epoch = int(val_df.loc[best_idx, 'epoch']) if 'epoch' in val_df.columns else best_idx + 1
                    
                    print("\n" + "="*70)
                    print(f"🏆 BEST VALIDATION ACCURACY: {best_acc*100:.2f}% (Epoch {best_epoch})")
                    print("="*70)
                else:
                    print("⚠ No validation metrics recorded")
                    
            except Exception as e:
                print(f"⚠ Could not parse metrics: {e}")
        else:
            print("⚠ Metrics file not found")
        
        # Show checkpoints
        ckpt_dir = f'lightning_logs/{latest}/checkpoints'
        if os.path.exists(ckpt_dir):
            ckpts = os.listdir(ckpt_dir)
            if ckpts:
                print(f"\n✓ Found {len(ckpts)} checkpoint(s):")
                for ckpt in ckpts:
                    size = os.path.getsize(f'{ckpt_dir}/{ckpt}') / (1024*1024)
                    print(f"  • {ckpt} ({size:.1f} MB)")
    else:
        print("⚠ No training logs found")
else:
    print("⚠ No lightning_logs directory found")

print("\n" + "="*70)
print("✅ ALL DONE!")
print("="*70)
print("\n💡 Scroll up to Cell 7 to see detailed training progress")
print("="*70)
