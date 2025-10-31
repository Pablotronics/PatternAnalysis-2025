Quick Start
Run the project in 3 simple steps
# 1. Decompress the data
python decompress_data.py


# 2. Train the CAN model
python Project2/train.py --data_root "C:\Users\User\Desktop\Final project comp3710"


# 3. Predict and visualise results
python Project2/predict.py --data_root "C:\Users\User\Desktop\Final project comp3710\keras_slices_data" --save_pngs --num_samples 9

For first-time setup, follow the environment instructions in Section 2 below.

# Project 3 — 2D Context-Aware Network (CAN) for Prostate Segmentation

## 1. Project Overview

This project implements a **2D Context-Aware Network (CAN)** for medical image segmentation as part of *COMP3710: Pattern Analysis*. The dataset used is the **HipMRI Study on Prostate Cancer**, consisting of 2D axial MRI slices. The model aims to accurately segment the prostate region, targeting a **minimum Dice similarity coefficient of 0.75** on the test set.

### 1.1 Objective

The goal is to train and evaluate a deep learning model that performs robust segmentation on MRI slices of the prostate. The CAN model enhances the baseline U-Net architecture by incorporating *context modules* that expand the receptive field and integrate multi-scale feature aggregation, improving segmentation around prostate boundaries.

### 1.2 Algorithm Summary

* **Baseline:** U-Net (Project 2)
* **Improved Model:** 2D Context-Aware Network (CAN)

  * Adds parallel atrous (dilated) convolutions to capture multi-scale context.
  * Includes context aggregation and residual skip pathways.
  * Enhances sensitivity to structural variations while maintaining efficient computation.

![CAN Architecture Example](https://miro.medium.com/v2/resize\:fit:1400/format\:webp/1*FpqP3IjA4TtFIS4qN3RjWA.png)
*Figure: Simplified CAN architecture showing multi-scale context modules.*

---

## 2. Dependencies and Environment Setup

A reproducible environment is provided using **Conda**.

### 2.1 Setup Instructions

```bash
# Clone the repository
https://github.com/Pablotronics/PatternAnalysis-2025.git
cd PatternAnalysis-2025

# Create and activate Conda environment (GPU)
conda env create -f environment-gpu.yml
conda activate comp3710-gpu

# (Alternative CPU-only environment)
conda env create -f environment-cpu.yml
conda activate comp3710-cpu
```

### 2.2 Key Dependencies

| Library      | Version | Purpose                 |
| ------------ | ------- | ----------------------- |
| Python       | 3.11    | Base language           |
| PyTorch      | 2.4+    | Deep learning framework |
| Torchvision  | 0.19+   | Data transformations    |
| CUDA Toolkit | 12.1    | GPU acceleration        |
| NumPy        | 1.26    | Numerical processing    |
| Matplotlib   | 3.9     | Visualisation           |
| NiBabel      | 5.2     | NIfTI file handling     |
| tqdm         | 4.66    | Progress bars           |

To confirm your setup:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Data Pre-processing

### 3.1 Dataset Structure

```
Final project comp3710/
├── keras_slices_data/
│   ├── keras_slices_train/
│   ├── keras_slices_validate/
│   ├── keras_slices_test/
│   ├── keras_slices_seg_train/
│   ├── keras_slices_seg_validate/
│   └── keras_slices_seg_test/
```

Each folder contains compressed `.nii.gz` slices, which are decompressed into `.nii` files using the provided script:

```bash
python decompress_data.py
```

### 3.2 Preprocessing Pipeline

* Normalisation (z-score → min-max scaling)
* Resizing to 256×256 pixels
* Conversion to binary mask (`prostate = 3` → 1, background = 0)
* Train/validation/test split using provided folder separation

---

## 4. Model Training

Run the training script from the **recognition** folder:

```bash
python Project2/train.py --data_root "C:\Users\User\Desktop\Final project comp3710"
```

### Training Parameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 20               |
| Batch Size    | 4                |
| Learning Rate | 5e-4             |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |

During training, model checkpoints are saved in:

```
models/CAN_models/unet2d_can_hipmri_best_<timestamp>.pth
```

The best checkpoint is selected automatically based on validation Dice score.

---

## 5. Model Evaluation and Prediction

Run the trained model on the test set:

```bash
python Project2/predict.py --data_root "C:\Users\User\Desktop\Final project comp3710\keras_slices_data" --save_pngs --num_samples 9
```

This produces:

* **Dice Coefficient Output** in the terminal
* **Prediction Samples** saved as PNGs in `./preds/`
* Optional NIfTI outputs in `./preds_nii/`

You can interactively visualise predictions:

```python
show_samples(model, test_ds, device, k=3)
```

A scrollable window will display side-by-side comparison of input MRI, ground truth, and predicted segmentation.

---

## 6. Example Outputs

**Quantitative Results:**

```
TEST Dice (prostate): 0.9186
```

**Qualitative Example:**

| Input MRI                    | Ground Truth              | CAN Prediction              |
| ---------------------------- | ------------------------- | --------------------------- |
| ![](./docs/input_sample.png) | ![](./docs/gt_sample.png) | ![](./docs/pred_sample.png) |

---

## 7. Reproducibility Notes

* Random seeds can be fixed via `torch.manual_seed(42)`.
* Deterministic training ensures consistent results across runs.
* Compatible with CUDA and CPU; code automatically falls back to CPU if GPU unavailable.

---

## 8. References

* [1] **Yu et al. (2015)** — *Multi-Scale Context Aggregation by Dilated Convolutions*. [arXiv:1511.07122](https://arxiv.org/abs/1511.07122)
* [2] **Ronneberger et al. (2015)** — *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

---

## 9. Author and Acknowledgements

Developed by **Pablotronics** for *COMP3710 (Pattern Analysis)* at **The University of Queensland**, 2025.

Dataset and starter code provided by *Dr. Shekhar S. Chandra*.
