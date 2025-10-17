"""
Evaluation script for uncertainty calibration analysis.

Supports both multi-background (mbkg) and evidential models.
Configuration is hardcoded in main() - update EXPERIMENT_NAMES to evaluate your models.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import load_data, FieldsDataset
from train import LitUNet
from train_evidential import LitEvidentialUNet
from uncertainty_cal_eval import calibration_curve, expected_calibration_error, confidence_interval_coverage, error_std_correlation


def find_latest_checkpoint(experiment_name):
    """Find the latest checkpoint for an experiment.
    
    Searches in both lightning_logs/ and checkpoints/ directories.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        str: Path to the latest checkpoint
        
    Raises:
        FileNotFoundError: If no checkpoints found
    """
    # Search patterns for different checkpoint locations
    search_patterns = [
        f"lightning_logs/{experiment_name}/version_*/checkpoints/*.ckpt",
        f"checkpoints/{experiment_name}/*.ckpt",
        f"lightning_logs/{experiment_name}/checkpoints/*.ckpt",
    ]
    
    all_checkpoints = []
    for pattern in search_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for experiment '{experiment_name}'\n"
            f"Searched in:\n  " + "\n  ".join(search_patterns)
        )
    
    # Get the most recently modified checkpoint
    latest_ckpt = max(all_checkpoints, key=os.path.getmtime)
    return latest_ckpt


def load_models(checkpoint_paths, model_types=None):
    """Load models from checkpoints. Auto-detects type if model_types=None."""
    models, detected_types = [], []
    
    for idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"Loading: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        experiment_tag = checkpoint.get('hyper_parameters', {}).get('experiment_tag', 'experiment')
        
        # Determine model type
        if model_types and idx < len(model_types):
            model_type = model_types[idx]
        else:
            model_type = 'mbkg' if 'bkgs' in checkpoint['state_dict'] else 'evidential'
        
        # Load appropriate model
        if model_type == 'mbkg':
            model = LitUNet.load_from_checkpoint(
                ckpt_path, strict=False,
                bkgs=checkpoint['state_dict']['bkgs'],
                bkg_sct_fields=checkpoint['state_dict']['bkg_sct_fields'],
                experiment_tag=experiment_tag
            )
            print(f"  ✓ Multi-background model ({model.bkgs.shape[0]} backgrounds)")
        else:
            model = LitEvidentialUNet.load_from_checkpoint(ckpt_path, strict=False, experiment_tag=experiment_tag)
            print(f"  ✓ Evidential model")
        
        models.append(model)
        detected_types.append(model_type)
    
    return models, detected_types


def build_data_loaders(file_path, batch_size=8, num_workers=2, seed=42):
    """Build calibrated and uncalibrated data loaders."""
    print(f"Loading data from: {file_path}")
    synth_fields, cal_e_fields, grids, uncal_spars = load_data(file_path)
    
    cal_loader = DataLoader(
        FieldsDataset(x_np=cal_e_fields, y_np=grids),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, generator=torch.Generator().manual_seed(seed)
    )
    uncal_loader = DataLoader(
        FieldsDataset(x_np=uncal_spars, y_np=grids),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, generator=torch.Generator().manual_seed(seed + 100)
    )
    
    print(f"Calibrated: {len(cal_loader.dataset)} samples")
    print(f"S param: {len(uncal_loader.dataset)} samples")
    return cal_loader, uncal_loader


def evaluate_model(model, model_type, loader, num_samples=None, device='cuda'):
    """Run model inference and collect predictions with uncertainties."""
    model.to(device).eval()
    means, stds, targets = [], [], []
    remaining = float('inf') if num_samples is None else num_samples
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if remaining <= 0:
                break
            
            take = min(x.shape[0], remaining) if remaining != float('inf') else x.shape[0]
            x, y = x[:take].to(device), y[:take].to(device)
            
            # Get predictions and uncertainty based on model type
            if model_type == 'mbkg':
                bg_grid = model.bkgs.unsqueeze(0).expand(take, -1, -1, -1, -1)
                bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(take, -1, -1, -1, -1)
                _, mean, std = model.reconstruction(x, bg_sct, bg_grid)
            else:  # evidential
                gamma, v, alpha, beta = model(x)
                mean = gamma
                std = torch.sqrt(beta / (alpha - 1 + 1e-10) + beta / (v * (alpha - 1) + 1e-10))
            
            means.append(mean.cpu().numpy())
            stds.append(std.cpu().numpy())
            targets.append(y.cpu().numpy())
            
            if remaining != float('inf'):
                remaining -= take
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {sum(m.shape[0] for m in means)} samples...")
    
    return np.concatenate(means), np.concatenate(stds), np.concatenate(targets)


def plot_calibration_curve(results_dict, output_path, title='Calibration Curve'):
    """Plot calibration curves for one or more models."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (label, (mean, std, targets)), color in zip(results_dict.items(), colors):
        expected, observed = calibration_curve(mean, std, targets)
        ece = expected_calibration_error(mean, std, targets)
        ax.plot(expected, observed, 'o-', label=f'{label} (ECE={ece:.4f})',
                linewidth=2, markersize=5, color=color)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('Expected Confidence Level', fontsize=13)
    ax.set_ylabel('Observed Coverage', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_metrics(mean, std, targets, label):
    """Print comprehensive calibration metrics."""
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    
    ece = expected_calibration_error(mean, std, targets)
    print(f"Expected Calibration Error (ECE): {ece:.4f} (lower is better)")
    
    print(f"\nConfidence Interval Coverage:")
    for conf in [0.68, 0.95, 0.99]:
        cov, _, _ = confidence_interval_coverage(mean, std, targets, conf)
        print(f"  {conf*100:.0f}% CI: {cov:.4f} (expected: {conf:.4f})")
    
    corr, p_val = error_std_correlation(mean, std, targets)
    print(f"\nError-Uncertainty Correlation: {corr:.4f} (p={p_val:.4e})")
    
    errors = np.abs(mean - targets)
    print(f"\nPrediction Quality:")
    print(f"  MAE: {np.mean(errors):.4e}")
    print(f"  MSE: {np.mean(errors**2):.4e}")
    print(f"  Mean Uncertainty: {np.mean(std):.4e}")
    print('='*70)


def main():
    # ========================================================================
    # Configuration - Just provide experiment names!
    # ========================================================================
    EXPERIMENT_NAMES = [
        "mbg_train_synth_test_cal_exp_25bkgs",
        "evidential_experiment",
    ]
    MODEL_TYPES = None  # Auto-detect, or specify ['mbkg', 'evidential']
    DATA_FILE = "all_data.mat"
    OUTPUT_DIR = "figures/calibration_eval"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    SEED = 42
    NUM_SAMPLES = None  # None = use all
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========================================================================
    # Find latest checkpoints for each experiment
    # ========================================================================
    print("\n" + "="*70 + "\nFINDING CHECKPOINTS\n" + "="*70)
    checkpoint_paths = []
    for exp_name in EXPERIMENT_NAMES:
        try:
            ckpt_path = find_latest_checkpoint(exp_name)
            checkpoint_paths.append(ckpt_path)
            print(f"✓ {exp_name}:")
            print(f"  {ckpt_path}")
        except FileNotFoundError as e:
            print(f"✗ {exp_name}: {e}")
            raise
    
    # ========================================================================
    # Load models and data
    # ========================================================================
    print("\n" + "="*70 + "\nLOADING MODELS\n" + "="*70)
    models, model_types = load_models(checkpoint_paths, MODEL_TYPES)
    
    print("\n" + "="*70 + "\nBUILDING DATA LOADERS\n" + "="*70)
    cal_loader, uncal_loader = build_data_loaders(DATA_FILE, BATCH_SIZE, NUM_WORKERS, SEED)
    
    # ========================================================================
    # Evaluate models
    # ========================================================================
    print("\n" + "="*70 + "\nEVALUATING MODELS\n" + "="*70)
    
    all_cal_results = {}
    all_uncal_results = {}
    
    for idx, (model, model_type) in enumerate(zip(models, model_types)):
        model_name = f"Model{idx+1}_{model_type}"
        print(f"\n--- {model_name} ---")
        
        # Calibrated data
        print("  Evaluating on calibrated data...")
        mean_cal, std_cal, targets_cal = evaluate_model(model, model_type, cal_loader, NUM_SAMPLES, DEVICE)
        all_cal_results[model_name] = (mean_cal, std_cal, targets_cal)
        
        # Individual plot
        plot_calibration_curve(
            {model_name: (mean_cal, std_cal, targets_cal)},
            os.path.join(OUTPUT_DIR, f"{model_name}_calibrated.pdf"),
            f"{model_name} - Calibrated Data"
        )
        print_metrics(mean_cal, std_cal, targets_cal, f"{model_name} - Calibrated")
        
        # Uncalibrated data
        print("  Evaluating on uncalibrated data...")
        mean_uncal, std_uncal, targets_uncal = evaluate_model(model, model_type, uncal_loader, NUM_SAMPLES, DEVICE)
        all_uncal_results[model_name] = (mean_uncal, std_uncal, targets_uncal)
        
        # Individual plot
        plot_calibration_curve(
            {model_name: (mean_uncal, std_uncal, targets_uncal)},
            os.path.join(OUTPUT_DIR, f"{model_name}_uncalibrated.pdf"),
            f"{model_name} - Uncalibrated Data"
        )
        print_metrics(mean_uncal, std_uncal, targets_uncal, f"{model_name} - Uncalibrated")
    
    # ========================================================================
    # Generate comparison plots
    # ========================================================================
    if len(models) > 1:
        print("\n" + "="*70 + "\nGENERATING COMPARISON PLOTS\n" + "="*70)
        plot_calibration_curve(all_cal_results, os.path.join(OUTPUT_DIR, "comparison_calibrated.pdf"),
                              "Calibration Curves - Calibrated Data")
        plot_calibration_curve(all_uncal_results, os.path.join(OUTPUT_DIR, "comparison_uncalibrated.pdf"),
                              "Calibration Curves - Uncalibrated Data")
    
    # Compare cal vs uncal for each model
    for idx in range(len(models)):
        model_name = f"Model{idx+1}_{model_types[idx]}"
        plot_calibration_curve(
            {"Calibrated": all_cal_results[model_name], "Uncalibrated": all_uncal_results[model_name]},
            os.path.join(OUTPUT_DIR, f"{model_name}_cal_vs_uncal.pdf"),
            f"{model_name}: Calibrated vs Uncalibrated"
        )
    
    print("\n" + "="*70 + f"\nDONE! Results saved to: {OUTPUT_DIR}\n" + "="*70)


if __name__ == "__main__":
    main()
