# DiffMICv2 - Modified for Kaggle with 80/20 Train/Val Split

## 🔧 Changes Made to Codebase

### 1. **Modified `utils.py`**

- Updated `get_dataset()` function to support train/validation split
- Added automatic 80/20 split with stratification
- Accepts `split` parameter: 'train', 'val', or 'test'

### 2. **Modified `diffuser_trainer.py`**

- Updated `train_dataloader()` to use train split (80%)
- Updated `val_dataloader()` to use validation split (20%)
- Automatically splits data based on `val_split` config parameter

## 🚀 Quick Start for Kaggle

### Step 1: Copy the Kaggle Code

Copy each cell from `KAGGLE_CODE.py` into your Kaggle notebook

### Step 2: Update Dataset Path

In Cell 2 and Cell 5, update:

```python
DATASET_PATH = '/kaggle/input/your-dataset-name/DATASET'
```

### Step 3: Run All Cells

Run cells 1-8 sequentially

## 📊 What the Code Does

| Cell | Description                                   |
| ---- | --------------------------------------------- |
| 1    | Clone GitHub repository                       |
| 2    | Verify dataset is mounted                     |
| 3    | Install dependencies                          |
| 4    | Clone EfficientSAM                            |
| 5    | **Configure 80/20 split + optimize settings** |
| 6    | Check GPU availability                        |
| 7    | **Train model with validation**               |
| 8    | View training results                         |

## ✨ Key Features

✅ **Automatic 80/20 Split** - No manual folder splitting needed
✅ **Stratified Split** - Maintains class distribution
✅ **Memory Optimized** - Batch size 8 for Kaggle GPU
✅ **Time Optimized** - 50 epochs default
✅ **Simple** - Only 8 cells to run

## 📁 Dataset Structure Required

```
DATASET/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## ⚙️ Configuration

The training will automatically:

- Use 80% of train folder for training
- Use 20% of train folder for validation
- Use test folder for final testing

## 🔧 Troubleshooting

**Out of Memory Error:**

- Reduce `batch_size` to 4 in Cell 5

**Dataset Not Found:**

- Update `DATASET_PATH` in Cell 2 and Cell 5

**No GPU:**

- Enable GPU: Settings → Accelerator → GPU T4 x2

## 📝 Original vs Modified

**Before:**

- Manual folder splitting required
- No built-in validation split
- Complex setup

**After:**

- Automatic 80/20 split in code
- Built-in validation support
- Simple 8-cell setup

## 🎯 Training Flow

1. Load full train dataset
2. Split into 80% train + 20% validation (stratified)
3. Train on 80% with validation monitoring
4. Save best checkpoints
5. View results

---

**Repository:** https://github.com/ALOK-KUMAR-1443/Facial-emotion-recognisation-
**License:** See LICENSE.md
