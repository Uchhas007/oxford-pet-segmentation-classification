# Oxford-IIIT Pet Segmentation & Classification

<div align="center">

Joint **foreground segmentation** and **fine-grained breed classification** on the Oxford-IIIT Pet Dataset using **Base U-Net** and **Attention U-Net**, trained end-to-end with a shared encoder and dual output heads.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-1.3%2B-yellow)](https://albumentations.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Augmentation Pipeline](#augmentation-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Inference](#inference)
- [Bonus Experiments](#bonus-experiments)
- [References](#references)

---

## Overview

This project tackles two computer vision tasks simultaneously on pet images:

| Task | Output | Head |
|---|---|---|
| Foreground Segmentation | Binary pixel mask — pet vs. background | U-Net / Attention U-Net decoder |
| Breed Classification | 37-class label (12 cat breeds + 25 dog breeds) | GAP → FC layers on bottleneck |

Both tasks share a **single encoder**, trained jointly through a combined segmentation and classification loss. This multi-task setup means the encoder learns features that are simultaneously spatially precise (for masking) and semantically discriminative (for breed identity).

---

## Dataset

**Oxford-IIIT Pet Dataset** — Parkhi et al., CVPR 2012

- **Download:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Total images:** 7,349 RGB JPEGs
- **Classes:** 37 breeds (~200 images per class, near-balanced)
- **Species split:** 12 cat breeds · 25 dog breeds
- **Annotations:** Per-pixel trimap PNG masks

### Trimap Mask Labels

| Label | Meaning | Preprocessing |
|---|---|---|
| 1 | Foreground (pet body) | Kept as foreground |
| 2 | Background | Kept as background |
| 3 | Boundary / uncertain | **Merged into foreground** |

Boundary pixels are reassigned to foreground before training, converting the task into a clean binary segmentation problem. This is the recommended approach for standard foreground segmentation benchmarking on this dataset.

### Dataset Split

| Split | Images | Proportion |
|---|---|---|
| Train | ~5,880 | 75% |
| Validation | ~1,100 | 15% |
| Test | ~735 | 10% |

---

## Models
## 📦 Pretrained Models

The pretrained models for this project are available on Kaggle.  
You can download them from the link below:

🔗 https://www.kaggle.com/datasets/uchhas007/oxford-pet-segmentation-trained-models
### Base U-Net

An encoder–decoder architecture with skip connections, adapted for RGB input at 128×128 resolution with a binary segmentation output and an attached classifier head.

```
Input (3×128×128)
    │
    ├─ Encoder Block 1 ── Conv3×3 → BN → ReLU (×2) ── 128×128×64  ──┐ skip
    │       MaxPool 2×2                                                │
    ├─ Encoder Block 2 ── Conv3×3 → BN → ReLU (×2) ── 64×64×128   ──┤ skip
    │       MaxPool 2×2                                                │
    ├─ Encoder Block 3 ── Conv3×3 → BN → ReLU (×2) ── 32×32×256   ──┤ skip
    │       MaxPool 2×2                                                │
    ├─ Encoder Block 4 ── Conv3×3 → BN → ReLU (×2) ── 16×16×512   ──┤ skip
    │       MaxPool 2×2                                                │
    │                                                                  │
    ├─ Bottleneck ──────── Conv3×3 → BN → ReLU (×2) ── 8×8×1024      │
    │       │                                                          │
    │       └─ Classifier Head ─────────────────────────── 37 logits  │
    │                                                                  │
    ├─ Decoder Block 1 ── Upsample + Concat skip ──── 16×16×512   ←──┘
    ├─ Decoder Block 2 ── Upsample + Concat skip ──── 32×32×256
    ├─ Decoder Block 3 ── Upsample + Concat skip ──── 64×64×128
    ├─ Decoder Block 4 ── Upsample + Concat skip ──── 128×128×64
    │
    └─ Output Conv 1×1 → Sigmoid ─────────────────── 128×128×1 (mask)
```

### Attention U-Net

Identical to Base U-Net but introduces **Attention Gates** at every skip connection in the decoder. Each gate learns a soft spatial mask that suppresses background-heavy encoder features before they are concatenated into the decoder, reducing the distraction caused by complex or colour-matched backgrounds.

**Attention Gate mechanism (Oktay et al., 2018):**

```
skip (x)  ──→  W_x (1×1 Conv) ──┐
                                 ├─→ ReLU → W_ψ (1×1 Conv) → Sigmoid → α
gate (g)  ──→  W_g (1×1 Conv) ──┘

attended output = α ⊙ x
```

The attended features replace the raw skip connection features in the decoder.

### Classifier Head

Attached to the bottleneck of both models:

```
Bottleneck features (8×8×1024)
    → Global Average Pooling       # 1024-dim vector, spatial invariant
    → Dropout(p=0.4)
    → Linear(1024 → 512) + ReLU
    → Dropout(p=0.3)
    → Linear(512 → 37)             # raw logits → Softmax at inference
```

---

## Project Structure

```
oxford-pet-segmentation-classification/
│
├── oxford_iiit_pet_dataset.ipynb   # Main notebook (Colab-ready, resumable)
├── README.md
│
└── checkpoints/                    # Auto-saved during training (Google Drive)
    ├── unet_best.pth               # Best U-Net checkpoint by val mIoU
    ├── unet_last.pth               # Latest U-Net epoch (for resuming)
    ├── attn_unet_best.pth          # Best Attention U-Net checkpoint
    └── attn_unet_last.pth          # Latest Attention U-Net epoch
```

---

## Installation

```bash
pip install torch torchvision albumentations opencv-python \
            matplotlib scikit-learn Pillow tqdm numpy
```

Or with the full Google Colab setup (runs the install cell inside the notebook automatically):

1. Open `oxford_iiit_pet_dataset.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run the **For Google Colab** setup cell — it mounts Drive, installs dependencies, and downloads the dataset
3. All checkpoints are saved to `MyDrive/cse428/checkpoints/` and resume automatically on re-run

---

## Usage

### Training

```python
# Base U-Net
unet_model = UNet(in_ch=3, num_seg_classes=2, num_cls_classes=37,
                  features=(64, 128, 256, 512))

unet_history = train_model(
    model=unet_model, model_name='unet',
    train_loader=train_loader, val_loader=val_loader,
    num_epochs=50, lr=1e-4,
    checkpoint_dir=CHECKPOINT_DIR, resume=True
)

# Attention U-Net
attn_model = AttentionUNet(in_ch=3, num_seg_classes=2, num_cls_classes=37,
                           features=(64, 128, 256, 512))

attn_history = train_model(
    model=attn_model, model_name='attn_unet',
    train_loader=train_loader, val_loader=val_loader,
    num_epochs=50, lr=1e-4,
    checkpoint_dir=CHECKPOINT_DIR, resume=True
)
```

Training supports **automatic mixed precision (AMP)** via `torch.cuda.amp` for ~1.5× speed and halved VRAM on NVIDIA T4/V100/A100.

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Image size | 128 × 128 | Memory-efficient; captures sufficient boundary detail |
| Batch size | 12 | Stable BN statistics within Colab T4 VRAM budget |
| Epochs | 50 | Dataset saturates within this range; early stopping prevents overfitting |
| Optimizer | Adam | Adaptive moments handle multi-task gradient scale differences |
| Learning rate | 1e-4 | Conservative initial rate; scheduler refines further |
| Weight decay | 1e-5 | Light L2 regularisation on all parameters |
| LR scheduler | ReduceLROnPlateau | Halves LR after 5 stagnant val-loss epochs (factor=0.5) |
| Early stopping | patience=10 | Saves best checkpoint and halts when val mIoU stops improving |
| Seg loss weight | 1.0 | Segmentation is the primary task |
| Cls loss weight | 0.5 | Classification signal balances without overpowering segmentation |
| Dropout (head) | 0.4 · 0.3 | Strong regularisation for the small per-class sample count |

### Loss Functions

```
L_seg  = 0.5 × BCE(pred_mask, gt_mask) + 0.5 × DiceLoss(pred_mask, gt_mask)
L_cls  = CrossEntropyLoss(pred_logits, gt_label)
L_total = 1.0 × L_seg + 0.5 × L_cls
```

BCE gives pixel-level probability calibration; Dice loss directly optimises the overlap metric and handles class imbalance. Their combination consistently outperforms either alone.

---

## Augmentation Pipeline

Applied jointly to image **and** mask to maintain spatial alignment (via [Albumentations](https://albumentations.ai/)):

```python
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

Colour jitter and noise are applied to the **image only**; geometric transforms are applied to both image and mask together.

---

## Evaluation Metrics

### Segmentation

| Metric | Formula | Notes |
|---|---|---|
| **mIoU** | TP / (TP + FP + FN), averaged over both classes | Primary metric — robust to class imbalance |
| **Dice** | 2·TP / (2·TP + FP + FN) | Matches training loss objective |
| **Pixel Accuracy** | (TP + TN) / all pixels | Supplementary — can be misleading on imbalanced masks |

### Classification

| Metric | Notes |
|---|---|
| **Top-1 Accuracy** | Primary — reliable given the near-balanced 37-class distribution |
| **Precision** | Weighted average across 37 classes |
| **Recall** | Weighted average across 37 classes |
| **F1 Score** | Harmonic mean of weighted precision and recall |

All metrics are reported separately for **train / validation / test** splits.

---

## Inference

### Single-image prediction

```python
# Set your index and model of choice
CUSTOM_INDEX = 42          # any integer index into the test set
MODEL_CHOICE = 'unet'      # 'unet' or 'attn_unet'
```

The output renders three panels side by side:

```
Original class  – "Abyssinian" (cat)
Predicted class – "Abyssinian" (cat)
IoU             – 87%

┌──────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐
│ Original     │  │ Image + True Mask     │  │ Image + Predicted Mask    │
│ Image        │  │ Overlay (green, α=0.5)│  │ Overlay (green, α=0.5)   │
└──────────────┘  └──────────────────────┘  └───────────────────────────┘
```

### Standalone inference from saved model

```python
predictor = PetPredictor(
    model_path='path/to/unet_best.pth',
    model_type='standard'   # 'standard' for UNet, 'attention' for AttentionUNet
)
predictor.predict(image_path='path/to/any_image.jpg')
```

---

## Bonus Experiments

### 1 — Augmentation Ablation

Trained identical U-Net models with and without the augmentation pipeline. Augmentation improves generalisation substantially, especially for classification given the small per-class sample count.

| Configuration | Val mIoU | Val Cls Accuracy |
|---|---|---|
| No augmentation | baseline | baseline |
| With augmentation | +↑ | +↑ |

### 2 — Classifier Backbone Comparison

The segmentation encoder–decoder is frozen and only the classifier head is replaced with pretrained alternatives:

| Head | Parameters |
|---|---|
| GAP + FC (default) | ~525K |
| MobileNetV3-style | ~350K |
| EfficientNet-B0 | ~680K |
| DenseNet-121 | ~820K |

Comparison curves plotted for val mIoU and val classification accuracy across heads.

### 3 — Three-Class Segmentation

The trimap boundary label is **not collapsed** into foreground. Models are retrained with a 3-class output head (foreground / background / boundary) using multi-class cross-entropy. Both U-Net and Attention U-Net are evaluated; metrics include per-class IoU for the boundary class separately.

### 4 — Learning Rate Sensitivity

U-Net trained for 10 epochs at four candidate learning rates to identify the optimal starting point:

```
Candidates: [1e-3, 5e-4, 1e-4, 5e-5]
```

Results summarised as a table and validation curve plot.

---

## References

```
[1] Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012).
    Cats and Dogs. CVPR 2012. IEEE.

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015. arXiv:1505.04597

[3] Oktay, O. et al. (2018).
    Attention U-Net: Learning Where to Look for the Pancreas.
    MIDL 2018. arXiv:1804.03999

[4] Buslaev, A. et al. (2020).
    Albumentations: Fast and Flexible Image Augmentations.
    Information 2020. https://albumentations.ai
```

---

<div align="center">
Made with PyTorch · Trained on Google Colab
</div>
