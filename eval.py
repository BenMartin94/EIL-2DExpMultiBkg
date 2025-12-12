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
from train_bcnn import LitBCNNUNet
from uncertainty_cal_eval import error_std_correlation, wei_ece, evidential_to_student_t


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
        
        
        # Load appropriate model
        if model_type == 'mbkg':
            model = LitUNet.load_from_checkpoint(
                ckpt_path, strict=False,
                bkgs=checkpoint['state_dict']['bkgs'],
                bkg_sct_fields=checkpoint['state_dict']['bkg_sct_fields'],
                experiment_tag=experiment_tag
            )
            print(f"  ✓ Multi-background model ({model.bkgs.shape[0]} backgrounds)")
        elif model_type == 'bcnn':
            model = LitBCNNUNet.load_from_checkpoint(ckpt_path, strict=False, experiment_tag=experiment_tag)
            print(f"  ✓ BCNN model")
        else:
            model = LitEvidentialUNet.load_from_checkpoint(ckpt_path, strict=False, experiment_tag=experiment_tag)
            print(f"  ✓ Evidential model")
        
        models.append(model)
        detected_types.append(model_type)
    
    return models, detected_types


def build_data_loaders(file_path, batch_size=8, num_workers=2, seed=42, num_synthetic_test_samples=25, val_split=0.05, num_backgrounds=25):
    """Build all data loaders matching train.py pattern."""
    print(f"Loading data from: {file_path}")
    
    # Load data with train/test split already done
    (synth_fields, cal_e_fields, grids, uncal_spars,
     synth_fields_test, cal_e_fields_test, grids_test, uncal_spars_test) = load_data(
        file_path, seed=seed, num_synthetic_test_samples=num_synthetic_test_samples
    )
    
    # Import MultiBkgDataset for training dataset
    from MultiBkgDataset import MultiBkgDataset
    from torch.utils.data import random_split
    
    # Create datasets
    synth_dataset = MultiBkgDataset(fields=synth_fields, grids=grids, n_backgrounds=num_backgrounds)
    synth_test_dataset = FieldsDataset(x_np=synth_fields_test, y_np=grids_test)
    exp_dataset = FieldsDataset(x_np=cal_e_fields_test, y_np=grids_test)
    sparam_dataset = FieldsDataset(x_np=uncal_spars_test, y_np=grids_test)

    # Split train/val dataset
    n_total = len(synth_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(synth_dataset, [n_train, n_val], generator=g)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size//2, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Deterministic shuffling for test loaders by using explicit generators
    test_gen = torch.Generator().manual_seed(seed + 100)
    sparam_gen = torch.Generator().manual_seed(seed + 200)
    
    test_loader = DataLoader(
        exp_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=test_gen,
    )
    synth_test_loader = DataLoader(
        synth_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    sparam_test_loader = DataLoader(
        sparam_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=sparam_gen,
    )
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Synth test samples: {len(synth_test_dataset)}")
    print(f"Calibrated test samples: {len(exp_dataset)}")
    print(f"S-param test samples: {len(sparam_dataset)}")
    
    return train_loader, val_loader, synth_test_loader, test_loader, sparam_test_loader


def evaluate_model(model, model_type, loader, num_samples=None, device='cuda'):
    """Run model inference and collect predictions with uncertainties."""
    model.to(device).eval()
    
    # Enable dropout for BCNN MC sampling
    if model_type == 'bcnn':
        model.enable_dropout()
    
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
            elif model_type == 'bcnn':
                # MC Dropout sampling
                MC_count = 15
                predictions = []
                variances = []
                for k in range(MC_count):
                    output = model(x)  # (take, 4, 100, 100)
                    predictions.append(output[:, :2, :, :])  # mean channels
                    variances.append(output[:, 2:, :, :])    # variance channels
                
                # Stack predictions
                predictions = torch.stack(predictions)  # (MC_count, take, 2, 100, 100)
                variances = torch.stack(variances)      # (MC_count, take, 2, 100, 100)
                
                # Compute mean and total uncertainty
                mean = torch.mean(predictions, dim=0)  # (take, 2, 100, 100)
                aleatoric = torch.mean(torch.exp(variances), dim=0)  # mean of predicted variances
                epistemic = torch.mean((predictions - mean.unsqueeze(0))**2, dim=0)  # variance of predictions
                std = torch.sqrt(aleatoric + epistemic)  # total uncertainty
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


def compute_signal_strength(data_loader, device='cpu'):
    """
    Compute signal strength statistics from field data.
    
    Converts field data from real/imaginary representation to magnitude and phase,
    then computes mean signal strength across all samples.
    
    Args:
        data_loader: DataLoader providing field data with shape (N, 2, H, W)
                     where channel 0 is real part and channel 1 is imaginary part
        device: Device to perform computation on
    
    Returns:
        dict: Dictionary containing signal strength statistics:
            - 'mean_magnitude': Mean magnitude across all samples
            - 'std_magnitude': Standard deviation of magnitudes
            - 'max_magnitude': Maximum magnitude observed
            - 'mean_phase': Mean phase in radians
            - 'std_phase': Standard deviation of phases in radians
    """
    print("Computing signal strength statistics...")
    
    all_magnitudes = []
    all_phases = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            # Move to device
            x = x.to(device)
            y = y.to(device)
            
            # Process input fields (x has shape: batch_size, 2, H, W)
            # Channel 0: real part, Channel 1: imaginary part
            real_part = x[:, 0, :, :]  # (batch_size, H, W)
            imag_part = x[:, 1, :, :]  # (batch_size, H, W)
            
            # Compute magnitude and phase
            magnitude = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.atan2(imag_part, real_part)
            
            # Store statistics
            all_magnitudes.append(magnitude.cpu().numpy().flatten())
            all_phases.append(phase.cpu().numpy().flatten())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {batch_idx + 1} batches...")
    
    # Concatenate all batches
    all_magnitudes = np.concatenate(all_magnitudes)
    all_phases = np.concatenate(all_phases)
    
    # Compute statistics
    stats = {
        'mean_magnitude': np.mean(all_magnitudes),
        'std_magnitude': np.std(all_magnitudes),
        'max_magnitude': np.max(all_magnitudes),
        'median_magnitude': np.median(all_magnitudes),
        'mean_phase': np.mean(all_phases),
        'std_phase': np.std(all_phases),
    }
    
    print(f"\nSignal Strength Statistics:")
    print(f"  Mean Magnitude: {stats['mean_magnitude']:.4e}")
    print(f"  Std Magnitude:  {stats['std_magnitude']:.4e}")
    print(f"  Max Magnitude:  {stats['max_magnitude']:.4e}")
    print(f"  Median Magnitude: {stats['median_magnitude']:.4e}")
    print(f"  Mean Phase:     {stats['mean_phase']:.4f} rad")
    print(f"  Std Phase:      {stats['std_phase']:.4f} rad")
    
    return stats


def run_experiment(models, model_types, model_names, test_loader, experiment_name, output_dir, 
                   max_figures=10, device='cuda', noise_std=0):
    """
    Run an experiment: evaluate models on test data and save visualization figures.
    
    Args:
        models: List of PyTorch Lightning models to evaluate
        model_types: List of model type strings ('mbkg', 'bcnn', 'evidential')
        model_names: List of display names for each model
        test_loader: DataLoader providing test (x, y) pairs
        experiment_name: Name of experiment (used for folder naming)
        output_dir: Base output directory for saving figures
        max_figures: Maximum number of sample figures to save
        device: Device to run inference on
        noise_std: Standard deviation of noise to inject into inputs

    Returns:
        results_dict: Dictionary mapping model names to (mean, std, targets) tuples
    """
    print(f"\n{'='*70}\n{experiment_name}\n{'='*70}")
    
    # Create experiment output directory
    exp_output_dir = os.path.join(output_dir, experiment_name.lower().replace(' ', '_'))
    os.makedirs(exp_output_dir, exist_ok=True)
    
    results_dict = {}
    all_model_results = []  # Store results for all models
    
    for model, model_type, model_name in zip(models, model_types, model_names):
        print(f"\n--- {model_name} ---")
        
        model.to(device).eval()
        
        # Enable dropout for BCNN MC sampling
        if model_type == 'bcnn':
            model.enable_dropout()
        
        all_inputs = []
        all_means = []
        all_stds = []
        all_targets = []
        all_bcnn_preds = []
        all_bcnn_vars = []
        all_evidential_params = []  # Store (gamma, v, alpha, beta) for evidential models
        
        # Run inference on test data
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)

                noise = torch.randn_like(x) * noise_std
                x = x + noise

                # Get predictions and uncertainty based on model type
                if model_type == 'mbkg':
                    B = x.shape[0]
                    bg_grid = model.bkgs.unsqueeze(0).expand(B, -1, -1, -1, -1)
                    bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(B, -1, -1, -1, -1)
                    _, mean, std = model.reconstruction(x, bg_sct, bg_grid)
                elif model_type == 'bcnn':
                    # MC Dropout sampling
                    MC_count = 15
                    predictions = []
                    variances = []
                    for k in range(MC_count):
                        output = model(x)  # (B, 4, 100, 100)
                        predictions.append(output[:, :2, :, :])  # mean channels
                        variances.append(output[:, 2:, :, :])    # variance channels
                    
                    # Stack predictions
                    predictions = torch.stack(predictions)  # (MC_count, B, 2, 100, 100)
                    variances = torch.stack(variances)      # (MC_count, B, 2, 100, 100)
                    
                    # Compute mean and total uncertainty
                    mean = torch.mean(predictions, dim=0)  # (B, 2, 100, 100)
                    aleatoric = torch.mean(torch.exp(variances), dim=0)  # mean of predicted variances
                    epistemic = torch.mean((predictions - mean.unsqueeze(0))**2, dim=0)  # variance of predictions
                    std = torch.sqrt(aleatoric + epistemic)  # total uncertainty
                else:  # evidential
                    gamma, v, alpha, beta = model(x)
                    mean = gamma
                    std = torch.sqrt(beta / (alpha - 1 + 1e-10) + beta / (v * (alpha - 1) + 1e-10))
                    # Store evidential parameters for Student-t calibration
                    all_evidential_params.append((gamma.cpu(), v.cpu(), alpha.cpu(), beta.cpu()))
                
                all_inputs.append(x.cpu())
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())
                all_targets.append(y.cpu())
                if model_type == 'bcnn':
                    all_bcnn_preds.append(predictions.cpu())
                    all_bcnn_vars.append(variances.cpu())
        
        # Concatenate all batches
        all_inputs = torch.cat(all_inputs, dim=0)
        all_means = torch.cat(all_means, dim=0)
        all_stds = torch.cat(all_stds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"  Processed {len(all_inputs)} samples")
        
        # Convert to numpy for statistics computation
        means_np = all_means.numpy()
        stds_np = all_stds.numpy()
        targets_np = all_targets.numpy()
        bcnn_preds_np = None
        bcnn_vars_np = None
        if model_type == 'bcnn':
            bcnn_preds_np = torch.cat(all_bcnn_preds, dim=1).numpy()
            bcnn_vars_np = torch.cat(all_bcnn_vars, dim=1).numpy()
            bcnn_stds_np = np.sqrt(np.exp(bcnn_vars_np))
            print(f"  BCNN MC Predictions shape: {bcnn_preds_np.shape}")
        
        # Compute statistics for this model (for summary at end)
        mse_per_sample = np.mean((means_np - targets_np)**2, axis=(1, 2, 3))
        overall_mse = np.mean((means_np - targets_np)**2)
        mae = np.mean(np.abs(means_np - targets_np))
        correlation, p_value = error_std_correlation(means_np, stds_np, targets_np)
        
        # Compute calibration error based on model type
        if model_type == 'bcnn':
            expected_calibration_error_value, creds, accs = wei_ece(bcnn_preds_np, bcnn_stds_np, targets_np)
        elif model_type == 'evidential':
            # Use Student-t distribution for evidential models
            gamma_list, v_list, alpha_list, beta_list = zip(*all_evidential_params)
            gamma_np = torch.cat(gamma_list, dim=0).numpy()
            v_np = torch.cat(v_list, dim=0).numpy()
            alpha_np = torch.cat(alpha_list, dim=0).numpy()
            beta_np = torch.cat(beta_list, dim=0).numpy()
            
            # Convert to Student-t parameters
            mean_t, scale_t, nu_t = evidential_to_student_t(gamma_np, v_np, alpha_np, beta_np)
            expected_calibration_error_value, creds, accs = wei_ece(
                mean_t, scale_t, targets_np, distribution='student_t', nu=nu_t
            )
        else:
            expected_calibration_error_value, creds, accs = wei_ece(means_np, stds_np, targets_np)
        
        # Save results for later analysis
        results_dict[model_name] = (
            means_np,
            stds_np,
            targets_np
        )
        
        # Store results for combined visualization
        all_model_results.append({
            'name': model_name,
            'inputs': all_inputs,
            'means': all_means,
            'stds': all_stds,
            'targets': all_targets,
            'mse_per_sample': mse_per_sample,
            'overall_mse': overall_mse,
            'mae': mae,
            'correlation': correlation,
            'expected_calibration_error': expected_calibration_error_value,
            'calibration_creds': creds,
            'calibration_accs': accs
        })
    # Create combined figures: one figure per sample, all models in rows
    num_models = len(models)
    num_figs_to_save = min(max_figures, len(all_model_results[0]['inputs']))
    print(f"\n  Saving {num_figs_to_save} combined visualization figures to {exp_output_dir}")
    
    # Font size configuration for paper-ready plots
    TITLE_FONTSIZE = 28  # Column titles
    YLABEL_FONTSIZE = 28  # Model names on left
    
    for sample_idx in range(num_figs_to_save):
        # Create figure: rows = models, columns = GT, Pred, Std, Abs Error
        fig, axes = plt.subplots(num_models, 4, figsize=(16, 4 * num_models))
        fig.suptitle(f"{experiment_name} - Sample {sample_idx}", fontsize=32)
        
        # Handle case of single model
        if num_models == 1:
            axes = axes.reshape(1, -1)
        
        # First pass: collect all data and compute global min/max for color scales
        all_targets = []
        all_preds = []
        all_stds = []
        all_errs = []
        
        for model_result in all_model_results:
            target = model_result['targets'][sample_idx, 0]  # Real channel only
            pred = model_result['means'][sample_idx, 0]      # Real channel only
            std = model_result['stds'][sample_idx, 0]        # Real channel only
            abs_err = torch.abs(pred - target)
            
            all_targets.append(target)
            all_preds.append(pred)
            all_stds.append(std)
            all_errs.append(abs_err)
        
        # Compute global min/max for columns 0-1 (GT and Pred share same scale)
        gt_pred_min = min(torch.min(t).item() for t in all_targets + all_preds)
        gt_pred_max = max(torch.max(t).item() for t in all_targets + all_preds)
        
        # Compute global min/max for column 3 (Abs Error)
        err_min = min(torch.min(e).item() for e in all_errs)
        err_max = max(torch.max(e).item() for e in all_errs)
        
        # Column titles (only displayed on top row)
        column_titles = ["Ground Truth", "Prediction", "Uncertainty", "Absolute Error"]
        
        def show(ax, tensor2d, cmap='viridis', vmin=None, vmax=None, show_ylabel=False, ylabel_text="", show_title=False, title_text=""):
            im = ax.imshow(tensor2d, cmap=cmap, vmin=vmin, vmax=vmax)
            if show_title:
                ax.set_title(title_text, fontsize=TITLE_FONTSIZE)
            if show_ylabel:
                ax.set_ylabel(ylabel_text, fontsize=YLABEL_FONTSIZE, rotation=90, labelpad=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Map model types to display acronyms
        acronym_map = {'mbkg': 'MBM', 'evidential': 'EDLS', 'bcnn': 'BCNN'}
        
        # Second pass: plot with synced color scales
        for model_idx, model_result in enumerate(all_model_results):
            model_name = model_result['name']
            model_type = model_types[model_idx]
            display_name = acronym_map.get(model_type, model_name)
            is_top_row = (model_idx == 0)
            
            # Column 0: Ground Truth (synced with Pred) - show ylabel for model name
            show(axes[model_idx, 0], all_targets[model_idx], 
                 cmap='viridis', vmin=gt_pred_min, vmax=gt_pred_max,
                 show_ylabel=True, ylabel_text=display_name,
                 show_title=is_top_row, title_text=column_titles[0])
            
            # Column 1: Prediction (synced with GT)
            show(axes[model_idx, 1], all_preds[model_idx], 
                 cmap='viridis', vmin=gt_pred_min, vmax=gt_pred_max,
                 show_title=is_top_row, title_text=column_titles[1])
            
            # Column 2: Uncertainty (independent scale per model)
            show(axes[model_idx, 2], all_stds[model_idx], 
                 cmap='plasma', vmax=1, # clipping for plots.
                 show_title=is_top_row, title_text=column_titles[2])
            
            # Column 3: Absolute Error (synced across models)
            show(axes[model_idx, 3], all_errs[model_idx], 
                 cmap='magma', vmin=err_min, vmax=err_max,
                 show_title=is_top_row, title_text=column_titles[3])
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_output_dir, f'sample_{sample_idx:03d}.pdf'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Print comparative summary of all models
    print(f"\n{'='*70}")
    print(f"COMPARATIVE SUMMARY - {experiment_name}")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'MSE':<15} {'MAE':<15} {'Corr':<10} {'ECE':<10}")
    print(f"{'-'*70}")
    for model_result in all_model_results:
        model_name = model_result['name']
        mse = model_result['overall_mse']
        mae = model_result['mae']
        corr = model_result['correlation']
        ece = model_result['expected_calibration_error']
        print(f"{model_name:<30} {mse:<15.6e} {mae:<15.6e} {corr:<10.4f} {ece:<10.4f}")
    print(f"{'='*70}\n")
    
    # Create combined calibration plot for all models
    print(f"\n  Creating combined WEI-ECE calibration plot...")
    
    # Font settings
    TITLE_FONTSIZE = 28
    AXIS_LABEL_FONTSIZE = 20
    LEGEND_FONTSIZE = 20
    TICK_FONTSIZE = 18
    
    plt.figure(figsize=(10, 8))
    for model_result, model_type in zip(all_model_results, model_types):
        model_name = model_result['name']
        creds = model_result['calibration_creds']
        accs = model_result['calibration_accs']
        ece = model_result['expected_calibration_error']
        
        # Map model type to display label
        label_map = {'mbkg': 'MBM', 'evidential': 'EDLS', 'bcnn': 'BCNN'}
        display_label = label_map.get(model_type, model_name)
        
        # Sort by credibility for proper visualization
        # Filter out bins with zero credibility (empty bins)
        mask = creds > 0
        creds_filtered = creds[mask]
        accs_filtered = accs[mask]
        
        # Sort by credibility
        sort_idx = np.argsort(creds_filtered)
        creds_sorted = creds_filtered[sort_idx]
        accs_sorted = accs_filtered[sort_idx]
        
        plt.plot(creds_sorted, accs_sorted, 'o-', linewidth=2, markersize=6, 
                label=f"{display_label} (ECE={ece:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    plt.xlabel('Confidence', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Accuracy (Observed)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'Calibration Curves - {experiment_name}', fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    calibration_plot_path = os.path.join(exp_output_dir, 'wei_ece_calibration_combined.png')
    plt.savefig(calibration_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved calibration plot to: {calibration_plot_path}")
    
    print(f"\n✓ Experiment complete: {experiment_name}")
    print(f"  Figures saved to: {exp_output_dir}")
    
    return results_dict


def main():
    # ========================================================================
    # Configuration - Just provide experiment names!
    # ========================================================================
    EXPERIMENT_NAMES = [
        "mbg_train_synth_test_cal_exp_25bkgs",
        "evidential_experiment",
        "bcnn_experiment_800"
    ]
    MODEL_TYPES = [
        "mbkg",
        "evidential",
        "bcnn"]
    DATA_FILE = "all_data.mat"
    OUTPUT_DIR = "figures/calibration_eval"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    SEED = 42
    NUM_SAMPLES = None  # None = use all
    DEVICE = "cpu"#'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    train_loader, val_loader, synth_test_loader, cal_loader, uncal_loader = build_data_loaders(
        DATA_FILE, BATCH_SIZE, NUM_WORKERS, SEED
    )

    # Compute signal strength for scaling noise
    signal_details = compute_signal_strength(synth_test_loader)
    signal_std = signal_details['mean_magnitude']

    # ========================================================================
    # Evaluate models - Repeat experiments each for different data
    # ========================================================================

    # ========================================================================
    # Experiment 1 - Synth test set
    # ========================================================================
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        synth_test_loader,
        experiment_name="Experiment 1 - Synth Test Set",
        output_dir="figures",
        max_figures=10,
        device=DEVICE
    )

    # ========================================================================
    # Experiment 2 - 10% noise injected into synthetic fields
    # ========================================================================
    noise_std = 0.1 * signal_std
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        synth_test_loader,
        experiment_name="Experiment 2 - 10% Noise",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
        noise_std=noise_std
    )

    # ========================================================================
    # Experiment 3 - 20% noise injected into synthetic fields
    # ========================================================================
    noise_std = 0.2 * signal_std
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        synth_test_loader,
        experiment_name="Experiment 3 - 20% Noise",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
        noise_std=noise_std
    )

    # ========================================================================
    # Experiment 4 - 40% noise injected into synthetic fields
    # ========================================================================
    noise_std = 0.4 * signal_std
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        synth_test_loader,
        experiment_name="Experiment 4 - 40% Noise",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
        noise_std=noise_std
    )

    # ========================================================================
    # Experiment 4.5 - 100% noise injected into synthetic fields
    # ========================================================================
    noise_std = 1.0 * signal_std
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        synth_test_loader,
        experiment_name="Experiment 5 - 100% Noise",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
        noise_std=noise_std
    )

    # ========================================================================
    # Experiment 5 - Calibrated E-field test set
    # ========================================================================
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        cal_loader,
        experiment_name="Experiment 6 - Calibrated E-field Test Set",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
    )
    # ========================================================================
    # Experiment 5 - Calibrated E-field test set with 100% noise
    # ========================================================================
    
    results_dict = run_experiment(
        models, model_types, EXPERIMENT_NAMES,
        cal_loader,
        experiment_name="Experiment 7 - Calibrated E-field Test Set - 100% Noise",
        output_dir="figures",
        max_figures=10,
        device=DEVICE,
        noise_std=1.0 * signal_std
    )
    

if __name__ == "__main__":
    main()
