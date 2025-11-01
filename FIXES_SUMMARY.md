# DiffMICv2 Notebook - Fixes Applied

## Problem 1: scipy/numpy Compatibility Error

**Error Message:**

```
ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, ...)
```

**Solution:**

```python
!pip uninstall -y numpy scipy -qq
!pip install numpy==1.24.3 -qq
!pip install scipy==1.11.4 -qq
```

- Removed `seaborn` import (not needed for this project)
- Fixed version conflicts between numpy and scipy

## Problem 2: Dataset Configuration

**Your Dataset Structure:**

```
/kaggle/input/data/
├── images_001/
├── images_002/
├── ...
├── images_012/
├── train_val_list.txt
├── test_list.txt
└── [other CSV/PDF files]
```

**Solution Implemented:**

1. Set `DATA_ROOT = Path('/kaggle/input/data')`
2. Read image paths from `train_val_list.txt` and `test_list.txt`
3. Split train_val into 80% train / 20% validation using sklearn
4. Keep test set separate for final evaluation

**Code:**

```python
# Load official splits
train_val_images = load_image_list(DATA_ROOT / 'train_val_list.txt')
test_images = load_image_list(DATA_ROOT / 'test_list.txt')

# 80-20 split for train/val
train_images, val_images = train_test_split(
    train_val_images,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
```

## Problem 3: Custom Dataset Class

**Created `ChestXrayDataset` class that:**

- Uses predefined image lists (not scanning directories)
- Loads images from: `DATA_ROOT / relative_path`
- Example: `/kaggle/input/data/images_001/12345678_001.png`
- Handles grayscale/RGB conversion
- Applies data augmentation for training
- Normalizes to [-1, 1] range for diffusion model

## Data Split Summary

Following official splits with 80-20 train/val ratio:

- **Training Set**: ~80% of train_val_list.txt
- **Validation Set**: ~20% of train_val_list.txt
- **Test Set**: 100% of test_list.txt (kept separate)

This ensures:
✅ Fair comparison using official test set
✅ Proper validation during training
✅ No data leakage between splits
✅ Reproducible results (fixed random seed)

## Next Steps

The notebook now includes:

1. ✅ Fixed dependency installation
2. ✅ Proper dataset loading with official splits
3. ✅ Custom dataset class for ChestX-ray
4. ✅ Data loaders with 80-20 train/val split
5. 🔄 Complete model architecture (UNet + Diffusion) - CONTINUE RUNNING
6. 🔄 Training loop with checkpointing - CONTINUE RUNNING
7. 🔄 Evaluation and testing - CONTINUE RUNNING

**To use this notebook:**

1. Upload to Kaggle
2. Add your ChestX-ray dataset
3. Enable GPU (P100 or T4)
4. Run all cells sequentially
5. Monitor training progress
6. Evaluate on test set

**The notebook will output:**

- Trained model checkpoints
- Training history plots
- Generated images
- Evaluation metrics (PSNR, SSIM, etc.)
- Test set results
