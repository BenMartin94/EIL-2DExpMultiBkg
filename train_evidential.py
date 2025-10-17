"""
Training script for evidential neural network for EM field reconstruction.
This script trains a network that outputs evidential parameters (gamma, v, alpha, beta)
for uncertainty quantification in addition to predictions.
"""
import os
# Set environment variable to use only GPU 0 (RTX 2080 Ti) before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from evidential import EvidentialUnet, evidential_NLL, reg_loss_1, reg_loss_2
from data_loader import load_data, FieldsDataset
from uncertainty_cal_eval import error_std_correlation


class LitEvidentialUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        lr: float = 1e-3,
        reg_coef_1: float = 0.01,
        reg_coef_2: float = 0.01,
        mse_coef: float = 1.0,
        experiment_tag: str = "evidential_experiment",
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Note: EvidentialUnet outputs 4 channels per output channel (gamma, v, alpha, beta)
        self.model = EvidentialUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        )
        
        self.experiment_tag = experiment_tag
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the evidential network.
        
        Returns:
            gamma: Predicted mean (B, 2, H, W)
            v: Variance scaling factor (B, 2, H, W)
            alpha: Shape parameter (B, 2, H, W)
            beta: Scale parameter (B, 2, H, W)
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        
        # Get evidential parameters
        gamma, v, alpha, beta = self(x)
        
        # Compute MSE loss (always included to anchor predictions)
        mse_loss = torch.mean((gamma - y) ** 2)
        
        # Compute evidential losses for each channel (real and imaginary)
        nll_loss = 0
        reg_loss = 0
        
        for c in range(y.shape[1]):  # Loop over channels (real, imag)
            # NLL loss - measures quality of uncertainty estimates
            nll_loss += evidential_NLL(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :],
                beta[:, c:c+1, :, :]
            )
            
            # Regularization loss - CRITICAL to prevent blob!
            # Only use reg_loss_1 (not both) for stability
            reg_loss_1_val = reg_loss_1(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :]
            )
            reg_loss_2_val = reg_loss_2(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :],
                beta[:, c:c+1, :, :]
            )
            reg_loss += self.hparams.reg_coef_1 * reg_loss_1_val + self.hparams.reg_coef_2 * reg_loss_2_val

        # Total loss: MSE (for good predictions) + NLL (for good uncertainty) + Reg (to prevent blob)
        total_loss = self.hparams.mse_coef * mse_loss + nll_loss + reg_loss
        
        # Log training metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_nll", nll_loss, on_step=True, on_epoch=True)
        self.log("train_reg", reg_loss, on_step=True, on_epoch=True)

        # Additional metrics for TensorBoard
        if batch_idx % 50 == 0:  # Log every 50 batches to avoid overhead
            with torch.no_grad():
                # Log learning rate
                self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
                
                # Log prediction error
                mae = torch.mean(torch.abs(gamma - y))
                self.log("train_mae", mae, on_step=True)
                
                # Log evidential parameter statistics (CRITICAL for debugging blob!)
                self.log("train_v_mean", torch.mean(v), on_step=True)
                self.log("train_v_std", torch.std(v), on_step=True)
                self.log("train_alpha_mean", torch.mean(alpha), on_step=True)
                self.log("train_beta_mean", torch.mean(beta), on_step=True)
                
                # Log uncertainty statistics
                # Epistemic uncertainty = beta / (alpha - 1)
                epistemic = beta / (alpha - 1 + 1e-10)
                # Aleatoric uncertainty = beta / (v * (alpha - 1))
                aleatoric = beta / (v * (alpha - 1) + 1e-10)
                
                self.log("train_epistemic_unc", torch.mean(epistemic), on_step=True)
                self.log("train_aleatoric_unc", torch.mean(aleatoric), on_step=True)
                self.log("train_total_unc", torch.mean(epistemic + aleatoric), on_step=True)
                
                # Check if parameters have variation (if std is very low, we have a blob!)
                self.log("train_gamma_std", torch.std(gamma), on_step=True)
                
                # Log weight magnitudes (L2 norm) for each layer
                for name, param in self.named_parameters():
                    if param.requires_grad and 'weight' in name:
                        weight_norm = param.data.norm(2).item()
                        # Simplify the name for logging (remove 'model.' prefix if present)
                        log_name = name.replace('model.', '')
                        self.log(f"weight_norm/{log_name}", weight_norm, on_step=True)
                
        return total_loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        
        # Get evidential parameters
        gamma, v, alpha, beta = self(x)
        
        # Compute MSE loss (same as training)
        mse_loss = torch.mean((gamma - y) ** 2)
        
        # Compute evidential losses for each channel (real and imaginary)
        nll_loss = 0
        reg_loss = 0
        
        for c in range(y.shape[1]):  # Loop over channels (real, imag)
            # NLL loss
            nll_loss += evidential_NLL(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :],
                beta[:, c:c+1, :, :]
            )
            
            # Regularization loss (only reg_loss_1, same as training)
            reg_loss_1_val = reg_loss_1(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :]
            )
            reg_loss_2_val = reg_loss_2(
                y[:, c:c+1, :, :],
                gamma[:, c:c+1, :, :],
                v[:, c:c+1, :, :],
                alpha[:, c:c+1, :, :],
                beta[:, c:c+1, :, :]
            )
            reg_loss += self.hparams.reg_coef_1 * reg_loss_1_val + self.hparams.reg_coef_2 * reg_loss_2_val

        # Total loss: Same formula as training step
        total_loss = self.hparams.mse_coef * mse_loss + nll_loss + reg_loss
        
        # Log validation metrics
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_nll", nll_loss, on_step=False, on_epoch=True)
        self.log("val_reg", reg_loss, on_step=False, on_epoch=True)
        
        # Additional validation metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(gamma - y))
            self.log("val_mae", mae, on_step=False, on_epoch=True)
            
            # Log per-channel metrics
            for c in range(y.shape[1]):
                channel_mae = torch.mean(torch.abs(gamma[:, c] - y[:, c]))
                self.log(f"val_mae_channel_{c}", channel_mae, on_step=False, on_epoch=True)
            
            # Log uncertainty statistics (with numerical stability)
            epistemic_unc = beta / (alpha - 1 + 1e-10)
            aleatoric_unc = beta / (v * (alpha - 1) + 1e-10)

            var = (beta/(alpha-1))*(1+v)/v
            
            self.log("val_epistemic_unc", torch.mean(epistemic_unc), on_step=False, on_epoch=True)
            self.log("val_aleatoric_unc", torch.mean(aleatoric_unc), on_step=False, on_epoch=True)
            
            # Log parameter statistics for debugging
            self.log("val_gamma_std", torch.std(gamma), on_step=False, on_epoch=True)
            self.log("val_v_mean", torch.mean(v), on_step=False, on_epoch=True)
            self.log("val_alpha_mean", torch.mean(alpha), on_step=False, on_epoch=True)
            self.log("val_beta_mean", torch.mean(beta), on_step=False, on_epoch=True)
        
        # Save local figures every 5 epochs
        if batch_idx == 0 and (self.current_epoch + 1) % 5 == 0:
            self._save_validation_figures(x, y, gamma, epistemic_unc, aleatoric_unc)
            #self._save_weight_histograms()
            
        return total_loss
    
    def _save_validation_figures(self, x, y, gamma, epistemic_unc, aleatoric_unc):
        """Save validation figures to disk"""
        n_examples = min(4, x.size(0))
        
        # Create local figures
        fig, axes = plt.subplots(n_examples, 5, figsize=(15, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n_examples):
            # Show real part
            im0 = axes[i, 0].imshow(x[i, 0].cpu(), cmap="viridis")
            axes[i, 0].set_title("Input (real)")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            im1 = axes[i, 1].imshow(y[i, 0].cpu(), cmap="viridis")
            axes[i, 1].set_title("Target (real)")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[i, 2].imshow(gamma[i, 0].detach().cpu(), cmap="viridis")
            axes[i, 2].set_title("Pred (real)")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            im3 = axes[i, 3].imshow(epistemic_unc[i, 0].detach().cpu(), cmap="plasma")
            axes[i, 3].set_title("Epistemic Unc (real)")
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            im4 = axes[i, 4].imshow(aleatoric_unc[i, 0].detach().cpu(), cmap="plasma")
            axes[i, 4].set_title("Aleatoric Unc (real)")
            plt.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
            
            for j in range(5):
                axes[i, j].axis("off")
                
        plt.tight_layout()
        fig_dir = f"figures/{self.experiment_tag}/epoch_{self.current_epoch}"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "val_examples.pdf")
        plt.savefig(fig_path)
        plt.close(fig)
    
    def _save_weight_histograms(self):
        """Save histograms of model weights"""
        fig_dir = f"figures/{self.experiment_tag}/epoch_{self.current_epoch}"
        os.makedirs(fig_dir, exist_ok=True)
        
        # Collect all weight tensors
        weight_data = {}
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Simplify name for display
                clean_name = name.replace('model.', '').replace('.weight', '')
                weight_data[clean_name] = param.data.cpu().flatten().numpy()
        
        # Determine grid size for subplots
        n_weights = len(weight_data)
        if n_weights == 0:
            return
        
        n_cols = 3
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot histogram for each weight tensor
        for idx, (name, weights) in enumerate(weight_data.items()):
            ax = axes[idx]
            ax.hist(weights, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\n(mean={weights.mean():.4f}, std={weights.std():.4f})')
            ax.set_xlabel('Weight value')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_weights, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Weight Histograms - Epoch {self.current_epoch}', fontsize=16, y=0.995)
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "weight_histograms.pdf")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def build_loaders(
    file_path: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    test_field_type: str = "cal"
):
    """
    Build data loaders for training and validation.
    
    Args:
        file_path: Path to data file
        batch_size: Batch size for training
        val_split: Validation split fraction
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        test_field_type: Type of field data to use for test set ("synth", "cal", or "uncal")
    
    Returns:
        train_loader, val_loader, test_loader
    """
    synth_fields, cal_e_fields, grids, uncal_spars = load_data(file_path)
    
    # Always use synth fields for training/validation
    train_val_dataset = FieldsDataset(x_np=synth_fields, y_np=grids)
    
    # Select test field data
    if test_field_type == "synth":
        test_fields = synth_fields
    elif test_field_type == "cal":
        test_fields = cal_e_fields
    elif test_field_type == "uncal":
        test_fields = uncal_spars
    else:
        raise ValueError(f"Unknown test_field_type: {test_field_type}. Choose from 'synth', 'cal', or 'uncal'")
    
    # Create test dataset
    test_dataset = FieldsDataset(x_np=test_fields, y_np=grids)
    
    # Split train/val dataset
    n_total = len(train_val_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_val_dataset, [n_train, n_val], generator=g)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size // 2, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Use test dataset with deterministic shuffle
    test_gen = torch.Generator().manual_seed(seed + 100)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, generator=test_gen
    )
    
    return train_loader, val_loader, test_loader


def test(
    model: LitEvidentialUNet,
    test_loader,
    num_cases: int = 10,
    output_dir: str = "figures/test_evidential",
    device: str | None = None,
):
    """
    Run testing on a handful of test samples.
    
    Args:
        model: Trained LitEvidentialUNet instance
        test_loader: DataLoader providing test (x,y) pairs
        num_cases: Number of individual samples to visualize
        output_dir: Directory to save figures
        device: Optional device override
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    collected = 0
    
    # Lists to collect all predictions for correlation analysis
    all_predictions = []
    all_targets = []
    all_epistemic = []
    all_aleatoric = []
    all_total_unc = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            B = x.shape[0]
            take = min(B, num_cases - collected)
            if take <= 0:
                break
            x = x.to(device)[:take]
            y = y.to(device)[:take]

            # Get evidential parameters
            gamma, v, alpha, beta = model(x)
            
            # Compute uncertainties
            epistemic_unc = beta / (alpha - 1)
            aleatoric_unc = beta / (v * (alpha - 1))
            total_unc = epistemic_unc + aleatoric_unc
            
            # Collect for correlation analysis
            all_predictions.append(gamma.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_epistemic.append(epistemic_unc.cpu().numpy())
            all_aleatoric.append(aleatoric_unc.cpu().numpy())
            all_total_unc.append(total_unc.cpu().numpy())

            # Compute metrics
            mae = torch.mean(torch.abs(gamma - y)).item()
            mse = torch.mean((gamma - y) ** 2).item()
            print(f"Samples {collected}-{collected+take} MAE={mae:.4e} MSE={mse:.4e}")

            for i in range(take):
                idx_global = collected + i
                
                # Create visualization with 2 rows (real, imag) and 6 columns
                fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                
                def show(ax, tensor2d, title, cmap='viridis'):
                    im = ax.imshow(tensor2d, cmap=cmap)
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Row 0 (Real): Input | GT | Pred | Abs Err | Epistemic | Aleatoric
                show(axes[0, 0], x[i, 0].cpu(), 'Input (Real)')
                show(axes[0, 1], y[i, 0].cpu(), 'GT (Real)')
                show(axes[0, 2], gamma[i, 0].cpu(), 'Pred (Real)')
                show(axes[0, 3], torch.abs(gamma[i, 0] - y[i, 0]).cpu(), 'Abs Err (Real)', cmap='magma')
                show(axes[0, 4], epistemic_unc[i, 0].cpu(), 'Epistemic (Real)', cmap='plasma')
                show(axes[0, 5], aleatoric_unc[i, 0].cpu(), 'Aleatoric (Real)', cmap='plasma')
                
                # Row 1 (Imag): Input | GT | Pred | Abs Err | Epistemic | Aleatoric
                show(axes[1, 0], x[i, 1].cpu(), 'Input (Imag)')
                show(axes[1, 1], y[i, 1].cpu(), 'GT (Imag)')
                show(axes[1, 2], gamma[i, 1].cpu(), 'Pred (Imag)')
                show(axes[1, 3], torch.abs(gamma[i, 1] - y[i, 1]).cpu(), 'Abs Err (Imag)', cmap='magma')
                show(axes[1, 4], epistemic_unc[i, 1].cpu(), 'Epistemic (Imag)', cmap='plasma')
                show(axes[1, 5], aleatoric_unc[i, 1].cpu(), 'Aleatoric (Imag)', cmap='plasma')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx_global}.pdf"), dpi=150)
                plt.close(fig)
                
            collected += take
            if collected >= num_cases:
                break
    
    # Compute correlation coefficients
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_epistemic = np.concatenate(all_epistemic, axis=0)
    all_aleatoric = np.concatenate(all_aleatoric, axis=0)
    all_total_unc = np.concatenate(all_total_unc, axis=0)
    
    print("\n" + "="*60)
    print("UNCERTAINTY CORRELATION ANALYSIS")
    print("="*60)
    
    # Epistemic uncertainty correlation
    corr_epistemic, p_epistemic = error_std_correlation(all_predictions, all_epistemic, all_targets)
    print(f"Epistemic Uncertainty Correlation: {corr_epistemic:.4f} (p={p_epistemic:.4e})")
    
    # Aleatoric uncertainty correlation
    corr_aleatoric, p_aleatoric = error_std_correlation(all_predictions, all_aleatoric, all_targets)
    print(f"Aleatoric Uncertainty Correlation: {corr_aleatoric:.4f} (p={p_aleatoric:.4e})")
    
    # Total uncertainty correlation
    corr_total, p_total = error_std_correlation(all_predictions, all_total_unc, all_targets)
    print(f"Total Uncertainty Correlation:     {corr_total:.4f} (p={p_total:.4e})")
    
    print("="*60)
    print(f"Interpretation: Values close to +1 indicate good uncertainty estimates.")
    print(f"High correlation means high uncertainty corresponds to high error.")
    print("="*60 + "\n")
                
    print(f"Saved {collected} test figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Evidential UNet for EM field reconstruction")
    parser.add_argument("--data", type=str, default="all_data.mat", help="Path to .mat file")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--reg-coef-1", type=float, default=0.1, help="Regularization coefficient for reg_loss_1")
    parser.add_argument("--reg-coef-2", type=float, default=0.01, help="Regularization coefficient for reg_loss_2")
    parser.add_argument("--mse-coef", type=float, default=0.0, help="MSE loss coefficient (always included in training)")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run test on a checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint for testing. If None, finds latest.")
    parser.add_argument("--experiment-tag", type=str, default="evidential_experiment", help="Tag for experiment (used in logging)")
    parser.add_argument("--test-field-type", type=str, default="cal", choices=["synth", "cal", "uncal"], 
                        help="Type of field data to use for test set (training always uses synth)")

    args = parser.parse_args()
    
    EXPERIMENT_TAG = args.experiment_tag

    # Build data loaders
    train_loader, val_loader, test_loader = build_loaders(
        file_path=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        test_field_type=args.test_field_type
    )

    if not args.test_only:
        # --- Training Phase ---
        model = LitEvidentialUNet(
            in_channels=2,
            out_channels=2,
            base_channels=args.base_channels,
            lr=args.lr,
            reg_coef_1=args.reg_coef_1,
            reg_coef_2=args.reg_coef_2,
            mse_coef=args.mse_coef,
            experiment_tag=EXPERIMENT_TAG
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/{EXPERIMENT_TAG}',
            filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="cuda",
            devices=1,
            log_every_n_steps=10,
            fast_dev_run=args.fast_dev_run,
            deterministic=True,
            callbacks=[checkpoint_callback],
        )

        print("--- Starting Training ---")
        trainer.fit(model, train_loader, val_loader)
        print("--- Training Finished ---")
        
        # Use the path to the best checkpoint saved during training
        ckpt_path = checkpoint_callback.best_model_path
        print(f"Best checkpoint from training: {ckpt_path}")

    else:
        # --- Test-Only Phase ---
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            print("No checkpoint path provided, finding the latest...")
            # Find the most recently modified checkpoint file
            list_of_files = glob.glob(f'checkpoints/{EXPERIMENT_TAG}/*.ckpt')
            if not list_of_files:
                raise FileNotFoundError("No checkpoints found to test.")
            ckpt_path = max(list_of_files, key=os.path.getctime)
        
        print(f"Loading model from checkpoint: {ckpt_path}")

    # --- Testing Phase ---
    print(f"--- Starting Test Phase on {ckpt_path} ---")
    
    # Load model from checkpoint
    model = LitEvidentialUNet.load_from_checkpoint(
        ckpt_path,
        strict=False,
        experiment_tag=EXPERIMENT_TAG
    )
    
    print(f"Successfully loaded evidential model")
    
    # Run testing
    test(model=model, test_loader=test_loader, num_cases=20, output_dir=f"figures/test_{EXPERIMENT_TAG}")

    print("--- Test Finished ---")


if __name__ == "__main__":
    main()
