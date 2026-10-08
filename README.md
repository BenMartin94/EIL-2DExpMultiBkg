# Electromagnetic Imaging with Multi-Background Learning

Deep learning framework for electromagnetic (EM) field reconstruction and imaging with uncertainty quantification. This project implements multiple neural network architectures (U-Net, Evidential Networks, Bayesian CNNs) for reconstructing permittivity distributions from scattered field measurements in both 2D and 3D.

## Overview

This repository contains implementations for EM inverse scattering problems with multiple approaches to uncertainty quantification:

- **Multi-Background Training**: Train models on multiple background permittivity distributions to improve generalization
- **Evidential Deep Learning**: Predict epistemic and aleatoric uncertainty using Normal-Inverse-Gamma distributions
- **Bayesian CNNs**: Monte Carlo dropout for uncertainty estimation
- **3D Reconstruction**: Transformer + U-Net architecture for volumetric imaging

## Project Structure

```
.
├── src/                        # Core library modules
│   ├── data_loader.py         # Data loading and dataset utilities
│   ├── Unet.py                # Standard and Bayesian U-Net implementations
│   ├── evidential.py          # Evidential neural network
│   ├── Models_3d.py           # 3D reconstruction models (Transformer + U-Net)
│   ├── MultiBkgDataset.py     # Multi-background dataset for 2D
│   ├── MultiBkgDataset_3d.py  # Multi-background dataset for 3D
│   ├── rendering.py           # Volume rendering utilities
│   ├── prepare_data_3d.py     # 3D data preprocessing
│   └── uncertainty_cal_eval.py # Uncertainty calibration metrics
│
├── train.py                   # Train standard U-Net with multi-background
├── train_evidential.py        # Train evidential network
├── train_bcnn.py              # Train Bayesian CNN
├── train_3d.py                # Train 3D reconstruction model
├── eval.py                    # Evaluation and uncertainty analysis
├── paper_plots.py             # Generate publication figures
├── main.py                    # Data loading utilities
│
├── data/                      # Training data
├── 3d_dataset/                # 3D experimental data
├── checkpoints/               # Saved model checkpoints
├── lightning_logs/            # TensorBoard logs
└── figures/                   # Output visualizations
```

## Installation

This project uses Python 3.10+ and PyTorch Lightning for training.

```bash
# Clone the repository
git clone https://github.com/BenMartin94/EIL-2DExpMultiBkg.git
cd EIL-2DExpMultiBkg

# Install dependencies (using uv or pip)
uv sync  # if using uv
# or
pip install -e .
```

### Dependencies

- PyTorch & PyTorch Lightning
- NumPy, SciPy, Matplotlib
- OpenCV (for edge detection in masking)
- scikit-image
- TensorBoard
- torchinfo

## Usage

### Training

**Standard U-Net with Multi-Background:**
```bash
python train.py --data_file all_data.mat --num_backgrounds 25 --experiment_tag mbg_25bkgs
```

**Evidential Network:**
```bash
python train_evidential.py --data_file all_data.mat --experiment_tag evidential_experiment
```

**Bayesian CNN:**
```bash
python train_bcnn.py --data_file all_data.mat --experiment_tag bcnn_experiment
```

**3D Reconstruction:**
```bash
python train_3d.py --background_file 3d_dataset/targetless_fielddata_72Rx.mat \
                   --target_file 3d_dataset/field_data_with2000Target_72Rx.mat \
                   --num_backgrounds 10
```

### Evaluation

Evaluate trained models and compute uncertainty calibration metrics:

```bash
python eval.py
```

The evaluation script computes:
- Mean Squared Error (MSE)
- Error-Uncertainty Correlation
- Weighted Expected Calibration Error (WEI-ECE)
- Confidence Interval Coverage

### Key Arguments

Common training arguments:
- `--data_file`: Path to .mat file containing training data
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--max_epochs`: Maximum training epochs (default: 50)
- `--num_backgrounds`: Number of background permittivity samples (for multi-background training)
- `--experiment_tag`: Name for logging and checkpoint saving
- `--gpu`: GPU ID to use (default: 0)

## Data Format

Input data should be MATLAB `.mat` files with the following structure:

**2D Data (`all_data.mat`):**
- `grids`: Permittivity distributions (N, 100, 100) complex
- `uncal_s_pars`: Uncalibrated S-parameters (N, 24, 24) complex
- `cal_e_fields`: Calibrated E-fields (N, 24, 24) complex  
- `synth_fields`: Synthesized scattered fields (N, 24, 24) complex

**3D Data:**
- Background file: Contains measurement chamber without targets
- Target file: Contains measurements with embedded targets
- Data includes receiver positions and complex field measurements

## Models

### U-Net Architecture
- Encoder-decoder with skip connections
- Handles complex-valued data (real/imaginary channels)
- Outputs: mean prediction (standard) or mean + variance (Bayesian)

### Evidential Network
- Based on U-Net backbone
- Outputs 4 parameters per channel: (γ, ν, α, β)
- Models epistemic and aleatoric uncertainty via Normal-Inverse-Gamma distribution
- Loss includes NLL and regularization terms

### 3D Transformer + U-Net
- Transformer encoder processes frequency domain data
- U-Net decoder reconstructs 3D volumetric permittivity
- Produces (56×56×56) voxel outputs from (24×24) receiver array

## Uncertainty Quantification

Three approaches to uncertainty estimation:

1. **Multi-Background Ensemble**: Train on diverse backgrounds, variance across predictions
2. **Evidential**: Predictive uncertainty from evidential parameters
3. **Bayesian (MC Dropout)**: Epistemic uncertainty from dropout sampling

Calibration metrics:
- **WEI-ECE**: Weighted expected calibration error
- **Error-Std Correlation**: How well uncertainty correlates with prediction error
- **CI Coverage**: Percentage of targets within confidence intervals

## Experiments

The project includes several experimental scenarios:

1. Synthetic test data (clean)
2. Noisy measurements (10%, 20%, 40%, 100% noise)
3. Calibrated E-field measurements
4. Multiple background variations (1, 2, 3, 4, 5, 6, 25, 50 backgrounds)
5. 3D volumetric reconstruction

## Citation

If you use this code in your research, please cite:

B. Martin, K. Narendra, H. Janz, C. Gilmore and I. Jeffrey, "Multi-background Data Augmentation for Microwave Imaging Uncertainty Quantification," in IEEE Transactions on Antennas and Propagation, doi: 10.1109/TAP.2026.3737816.
keywords: {Modeling;Training;Uncertainty;Scattering;Pixel;Deep learning;Measurement;Inverse problems;Microwave imaging;Calibration;Microwave Imaging;Uncertainty Quantification;Deep Learning;Inhomogeneous Backgrounds},


