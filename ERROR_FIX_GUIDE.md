# 🔧 Fix for scipy/numpy ValueError

## The Error

```
ValueError: All ufuncs must have type `numpy.ufunc`.
Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)
```

## Root Cause

This error occurs when there's a version mismatch between `numpy` and `scipy`. The `albumentations` library depends on `scipy`, which then fails to load due to incompatible numpy ufunc types.

## Solution Applied ✅

### Step 1: Install Cell (Replace Cell 1)

```python
# CRITICAL FIX: Uninstall conflicting versions first
!pip uninstall -y numpy scipy albumentations -qq

# Install compatible versions in correct order
!pip install --no-cache-dir numpy==1.24.3 -q
!pip install --no-cache-dir scipy==1.11.4 -q
!pip install --no-cache-dir albumentations==1.3.1 -q

# Install remaining packages
!pip install -q pandas matplotlib scikit-image==0.21.0 scikit-learn PyYAML opencv-python-headless einops tqdm
```

### Step 2: Import Cell (Replace Cell 2)

```python
# IMPORTANT: Remove seaborn import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # REMOVED - causes scipy conflict
from tqdm.auto import tqdm
# ... rest of imports
import albumentations as A  # Now this will work!
```

## Why This Works

1. **Uninstall first**: Removes conflicting cached versions
2. **Install numpy 1.24.3**: This version is compatible with scipy 1.11.4
3. **Install scipy 1.11.4**: Contains properly compiled ufuncs for numpy 1.24.3
4. **Install albumentations 1.3.1**: Works with the above versions
5. **Remove seaborn**: Not needed and can trigger additional scipy imports

## After Making Changes

**CRITICAL STEP:**

1. Go to **Runtime > Restart Runtime** (or Kernel > Restart)
2. Then run all cells from the beginning
3. The error should be gone!

## Verification

After running the import cell, you should see:

```
✓ Using device: cuda
✓ PyTorch version: 2.x.x
✓ NumPy version: 1.24.3
✓ Albumentations imported successfully!
```

## Alternative Solution (If Above Doesn't Work)

If you still get errors, try this order:

```bash
# In Kaggle, add a new code cell at the top:
!pip install --upgrade pip setuptools wheel
!pip install --force-reinstall --no-cache-dir numpy==1.24.3
!pip install --force-reinstall --no-cache-dir scipy==1.11.4
!pip install --force-reinstall --no-cache-dir albumentations==1.3.1

# Then restart runtime and run normally
```

## For Local Development

If running locally (not on Kaggle), you can also try:

```bash
conda create -n diffmic python=3.10
conda activate diffmic
pip install numpy==1.24.3 scipy==1.11.4
pip install albumentations==1.3.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Changes Made to Your Notebook ✅

1. ✅ **Cell 1**: Fixed installation order with compatible versions
2. ✅ **Cell 2**: Removed `seaborn` import
3. ✅ Added version verification prints
4. ✅ All other cells remain the same

**Now your notebook should run without the scipy error!**
