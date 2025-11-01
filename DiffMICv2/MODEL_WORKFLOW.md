# DiffMICv2 Model Training Workflow

## 🔄 Complete Training Pipeline

### **Phase 1: Setup & Data Loading**
```
1. Clone Repository
   └─> Load project files from GitHub

2. Install Dependencies
   └─> pytorch-lightning, diffusers, einops, etc.

3. Configure Dataset
   ├─> Set dataroot path
   ├─> Enable folder structure mode
   └─> Set val_split = 0.2 (20%)

4. Load Data
   ├─> Read train folder (all classes)
   ├─> Read test folder (all classes)
   └─> Auto-detect number of classes
```

### **Phase 2: Data Splitting (80/20)**
```
5. Train/Val Split
   ├─> Get all train samples
   ├─> Extract labels
   ├─> Stratified split:
   │   ├─> 80% → Training set
   │   └─> 20% → Validation set
   └─> Maintain class distribution
```

### **Phase 3: Model Initialization**
```
6. Initialize DiffMIC-v2 Components:
   
   a) Auxiliary Classifier (DCG)
      ├─> ResNet18 backbone
      ├─> Global and local predictions
      ├─> Load pretrained weights
      └─> Freeze (eval mode)
   
   b) Diffusion Model (ConditionalModel)
      ├─> U-Net architecture
      ├─> Timestep encoding
      ├─> Conditional guidance
      └─> Trainable parameters
   
   c) Diffusion Sampler (SR3Sampler)
      ├─> DDIM scheduler
      ├─> 1000 train timesteps
      └─> 100 test timesteps
```

### **Phase 4: Training Loop**
```
7. For each epoch (1 to n_epochs):
   
   A. Training Phase:
      ├─> For each batch in train_loader:
      │   │
      │   ├─> Load images (x_batch) & labels (y_batch)
      │   │
      │   ├─> Auxiliary Model Forward (frozen):
      │   │   ├─> Extract global features
      │   │   ├─> Extract local patches
      │   │   ├─> Generate attention maps
      │   │   └─> Predict y0_aux (global & local)
      │   │
      │   ├─> Diffusion Forward Process:
      │   │   ├─> Create label map from y_batch
      │   │   ├─> Add noise at random timestep
      │   │   ├─> Generate noisy_y
      │   │   └─> Create guided prob map (y0_cond)
      │   │
      │   ├─> Diffusion Model Forward:
      │   │   ├─> Input: x_batch, noisy_y, timestep
      │   │   ├─> Conditions: y0_cond, patches, attentions
      │   │   └─> Predict: noise_pred
      │   │
      │   ├─> Compute Loss:
      │   │   ├─> Focal loss with prior weights
      │   │   └─> MSE between noise_pred and noise_gt
      │   │
      │   └─> Backward & Optimize:
      │       ├─> loss.backward()
      │       ├─> optimizer.step()
      │       └─> Log train_loss
      │
      └─> Update learning rate (CosineAnnealingLR)
   
   B. Validation Phase (every 5 epochs):
      ├─> For each batch in val_loader:
      │   │
      │   ├─> Auxiliary Model Forward
      │   ├─> Generate y0_cond
      │   ├─> Sample from noise (yT)
      │   │
      │   ├─> Diffusion Reverse Process:
      │   │   ├─> Start with random noise
      │   │   ├─> Iteratively denoise (100 steps)
      │   │   ├─> Use DDIM scheduler
      │   │   └─> Generate final prediction
      │   │
      │   ├─> Average predictions over patches
      │   └─> Store gt & pred
      │
      ├─> Compute Metrics:
      │   ├─> Accuracy
      │   ├─> F1 Score
      │   ├─> Precision & Recall
      │   ├─> AUC (one-vs-one)
      │   └─> Cohen's Kappa
      │
      └─> Save Best Checkpoint:
          └─> Monitor F1 score (save_top_k=1)
```

### **Phase 5: Testing**
```
8. Load Best Checkpoint
   └─> Highest F1 score model

9. Test on Test Set:
   ├─> Similar to validation
   ├─> Use test_loader
   └─> Report final metrics

10. Generate Results:
    ├─> Confusion matrix
    ├─> Class-wise metrics
    └─> Visualizations
```

## 📊 Data Flow Architecture

```
Input Image (224x224x3)
         ↓
    ┌────────────────────────┐
    │  Auxiliary Classifier  │
    │      (Frozen DCG)       │
    └────────────────────────┘
         ↓              ↓
    Global Pred    Local Pred
         ↓              ↓
    ┌─────────────────────────┐
    │  Guided Probability Map │
    │      (y0_cond)          │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │   Noisy Label (y_t)     │
    │  (forward diffusion)    │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │  Conditional U-Net      │
    │  (Diffusion Model)      │
    │  Inputs: x, y_t, t      │
    │  Conditions: y0_cond    │
    └─────────────────────────┘
         ↓
    Noise Prediction (ε_θ)
         ↓
    ┌─────────────────────────┐
    │  Focal Loss + MSE       │
    └─────────────────────────┘
         ↓
    Backpropagation & Update
```

## 🎯 Key Components

### **1. Dual-Conditional Guidance**
- **Global Path**: Overall image classification
- **Local Path**: Patch-level predictions
- **Guided Map**: Interpolates between global and local
- **Purpose**: Multi-granularity attention

### **2. Heterologous Diffusion**
- **Forward Process**: Add noise to labels (not images)
- **Reverse Process**: Denoise labels to get predictions
- **Advantage**: Works in latent space, more efficient

### **3. Attention Mechanism**
- **Patches**: Local image regions
- **Attention Maps**: Spatial importance weights
- **Integration**: Guide diffusion process

### **4. Loss Function**
```python
Focal Loss = (1 + α(1-p)^γ) * MSE(noise_pred, noise_gt)
where:
  p = softmax probability
  α = 10 (focus on hard samples)
  γ = 1 (modulation factor)
```

## 📈 Training Monitoring

**Logged Metrics:**
- `train_loss`: Training loss per batch
- `accuracy`: Validation accuracy
- `f1`: F1 score (used for checkpoint selection)
- `precision`: Precision score
- `recall`: Recall score
- `auc`: AUC one-vs-one
- `kappa`: Cohen's Kappa

**Checkpoints:**
- Saved every epoch
- Best model: Highest F1 score
- Last model: Most recent

## ⚙️ Hyperparameters

**Training:**
- Batch size: 8 (Kaggle optimized)
- Epochs: 50 (adjustable)
- Learning rate: 0.001
- Scheduler: CosineAnnealingLR
- Validation frequency: Every 5 epochs

**Diffusion:**
- Train timesteps: 1000
- Test timesteps: 100
- Beta schedule: Linear
- Beta range: [0.0001, 0.02]

**Data:**
- Image size: 224x224
- Normalization: ImageNet stats
- Augmentation: Flip, rotation (train only)

## 🔄 Kaggle-Specific Workflow

```
Cell 1: Clone repo
   ↓
Cell 2: Verify dataset (check train/test folders)
   ↓
Cell 3: Install dependencies
   ↓
Cell 4: Clone EfficientSAM
   ↓
Cell 5: Configure (enable folder structure, set val_split=0.2)
   ↓
Cell 6: Check GPU
   ↓
Cell 7: Run training
   ├─> Auto 80/20 split
   ├─> Train with validation
   └─> Save checkpoints
   ↓
Cell 8: View results
   └─> Load checkpoint, display metrics
```

## ✅ Validation Checks

**Before Training:**
- ✓ Dataset path exists
- ✓ Train/test folders present
- ✓ GPU available
- ✓ Dependencies installed

**During Training:**
- ✓ Loss decreasing
- ✓ Validation F1 improving
- ✓ No OOM errors
- ✓ Checkpoints saving

**After Training:**
- ✓ Best checkpoint exists
- ✓ Metrics computed
- ✓ Results logged

## 🎓 Model Architecture Summary

**DiffMIC-v2 = Auxiliary Classifier + Diffusion Model**

1. **Auxiliary Classifier (DCG)**:
   - Provides prior knowledge
   - Frozen during training
   - Guides diffusion process

2. **Diffusion Model**:
   - Main trainable component
   - Refines predictions
   - Handles uncertainty

3. **Integration**:
   - Dual-conditional guidance
   - Heterologous diffusion
   - Attention-based fusion

## 📝 Output Files

```
/kaggle/working/
├── checkpoints/
│   ├── best_model.ckpt (highest F1)
│   └── last.ckpt (latest)
├── logs/
│   └── placental/
│       └── version_X/
│           ├── events.out.tfevents.*
│           └── hparams.yaml
└── outputs/
    └── (saved results)
```

---

**Total Pipeline**: Input Images → Auxiliary Features → Diffusion Process → Final Predictions → Evaluation Metrics
