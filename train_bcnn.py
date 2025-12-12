import os
# Set environment variable to use only GPU 0 (RTX 2080 Ti) before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from Unet import UNetBCNN, bayesian_loss
import main as data_main
from MultiBkgDataset import MultiBkgDataset
from Litmus_test import litmus_test
from data_loader import load_data, FieldsDataset, AugmentedFieldsDataset



class LitBCNNUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,  # 2 for prediction + 2 for log variance
        base_channels: int = 64,
        dropout_rate: float = 0.05,
        lr: float = 1e-3,
        experiment_tag: str = "experiment",
    ):
        super().__init__()
        # Save hyperparameters, ignoring large tensors.
        self.save_hyperparameters(ignore=['bkgs', 'bkg_sct_fields'])
        self.model = UNetBCNN(
            in_channels=in_channels, 
            out_channels=out_channels, 
            base_channels=base_channels,
            dropout_rate=dropout_rate
        )
        self.criterion = bayesian_loss
        #self.criterion = torch.nn.MSELoss()
        self.experiment_tag = experiment_tag

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    
    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Additional metrics for TensorBoard
        if batch_idx % 50 == 0:  # Log every 50 batches to avoid overhead
            with torch.no_grad():
                # Log learning rate
                self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
                
                # Log gradient norms
                total_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.log("grad_norm", total_norm, on_step=True)
                
                # Log model statistics
                mae = torch.mean(torch.abs(y_hat[:,:2,:,:] - y))
                self.log("train_mae", mae, on_step=True)
                
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        
        # Log validation metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Additional validation metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat[:,:2,:,:] - y))
            mse = torch.mean((y_hat[:,:2,:,:] - y) ** 2)
            self.log("val_mae", mae, on_step=False, on_epoch=True)
            self.log("val_mse", mse, on_step=False, on_epoch=True)
            
            # Log per-channel metrics
            for c in range(y.shape[1]):
                channel_mae = torch.mean(torch.abs(y_hat[:, c] - y[:, c]))
                self.log(f"val_mae_channel_{c}", channel_mae, on_step=False, on_epoch=True)
        
        # Log images to TensorBoard every 5 epochs
        if batch_idx == 0 and (self.current_epoch+1) % 5 == 0:
            self._log_images_to_tensorboard(x, y, y_hat)
            #self._log_reconstruction_images(fg_grid, all_bg_grids, fg_sct_field, all_bg_fields)
            
        return val_loss
        
    
    def _log_images_to_tensorboard(self, x, y, y_hat):
        """Log sample images to TensorBoard"""
        n_examples = min(4, x.size(0))
        
        def normalize_for_tensorboard(tensor):
            """Normalize tensor to [0, 1] range for TensorBoard visualization"""
            # Work with the original tensor shape
            original_shape = tensor.shape
            batch_size = original_shape[0]
            
            # Flatten each image in the batch individually
            tensor_flat = tensor.view(batch_size, -1)
            tensor_min = tensor_flat.min(dim=1, keepdim=True)[0]
            tensor_max = tensor_flat.max(dim=1, keepdim=True)[0]
            tensor_range = tensor_max - tensor_min
            
            # Avoid division by zero
            tensor_range = torch.where(tensor_range == 0, torch.ones_like(tensor_range), tensor_range)
            
            # Normalize and reshape back to original shape
            tensor_flat_norm = (tensor_flat - tensor_min) / tensor_range
            return tensor_flat_norm.view(original_shape)
        
        # Create image grids for TensorBoard
        input_images = []
        target_images = []
        pred_images = []
        
        for i in range(n_examples):
            # Real part (channel 0)
            input_images.append(x[i, 0].cpu().unsqueeze(0))  # Add channel dim
            target_images.append(y[i, 0].cpu().unsqueeze(0))
            pred_images.append(y_hat[i, 0].detach().cpu().unsqueeze(0))
        
        # Stack and normalize for TensorBoard
        input_grid = torch.stack(input_images)
        target_grid = torch.stack(target_images)
        pred_grid = torch.stack(pred_images)
        
        # Normalize all grids
        input_grid_norm = normalize_for_tensorboard(input_grid)
        target_grid_norm = normalize_for_tensorboard(target_grid)
        pred_grid_norm = normalize_for_tensorboard(pred_grid)
        
        # Log to TensorBoard
        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_images(
                'validation/input_real', input_grid_norm, self.current_epoch
            )
            self.logger.experiment.add_images(
                'validation/target_real', target_grid_norm, self.current_epoch
            )
            self.logger.experiment.add_images(
                'validation/prediction_real', pred_grid_norm, self.current_epoch
            )
            
            # Also log imaginary parts (channel 1)
            input_imag = torch.stack([x[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
            target_imag = torch.stack([y[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
            pred_imag = torch.stack([y_hat[i, 1].detach().cpu().unsqueeze(0) for i in range(n_examples)])
            
            # Normalize imaginary parts
            input_imag_norm = normalize_for_tensorboard(input_imag)
            target_imag_norm = normalize_for_tensorboard(target_imag)
            pred_imag_norm = normalize_for_tensorboard(pred_imag)
            
            self.logger.experiment.add_images(
                'validation/input_imag', input_imag_norm, self.current_epoch
            )
            self.logger.experiment.add_images(
                'validation/target_imag', target_imag_norm, self.current_epoch
            )
            self.logger.experiment.add_images(
                'validation/prediction_imag', pred_imag_norm, self.current_epoch
            )
        
        # Also save local figures as before
        fig, axes = plt.subplots(n_examples, 3, figsize=(9, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n_examples):
            # Show real part of channel 0 for input, target, prediction
            im0 = axes[i, 0].imshow(x[i, 0].cpu(), cmap="viridis")
            axes[i, 0].set_title("Input (real)")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            im1 = axes[i, 1].imshow(y[i, 0].cpu(), cmap="viridis")
            axes[i, 1].set_title("Target (real)")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[i, 2].imshow(y_hat[i, 0].detach().cpu(), cmap="viridis")
            axes[i, 2].set_title("Pred (real)")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            for j in range(3):
                axes[i, j].axis("off")
        plt.tight_layout()
        fig_dir = f"figures/{self.experiment_tag}/epoch_{self.current_epoch}"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "val_examples.pdf")
        plt.savefig(fig_path)
        plt.close(fig)


    def _log_weight_histograms(self):
        """Log weight histograms and statistics to TensorBoard"""
        if self.logger and hasattr(self.logger, 'experiment'):
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Log weight histograms
                    self.logger.experiment.add_histogram(
                        f'weights/{name}', param.data, self.current_epoch
                    )
                    
                    # Log gradient histograms
                    self.logger.experiment.add_histogram(
                        f'gradients/{name}', param.grad.data, self.current_epoch
                    )
                    
                    # Log weight statistics as scalars
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_max = param.data.max().item()
                    weight_min = param.data.min().item()
                    weight_norm = param.data.norm().item()
                    
                    self.log(f'weight_stats/{name}_mean', weight_mean, on_epoch=True)
                    self.log(f'weight_stats/{name}_std', weight_std, on_epoch=True)
                    self.log(f'weight_stats/{name}_max', weight_max, on_epoch=True)
                    self.log(f'weight_stats/{name}_min', weight_min, on_epoch=True)
                    self.log(f'weight_stats/{name}_norm', weight_norm, on_epoch=True)
                    
                    # Log gradient statistics
                    if param.grad is not None:
                        grad_mean = param.grad.data.mean().item()
                        grad_std = param.grad.data.std().item()
                        grad_norm = param.grad.data.norm().item()
                        
                        self.log(f'grad_stats/{name}_mean', grad_mean, on_epoch=True)
                        self.log(f'grad_stats/{name}_std', grad_std, on_epoch=True)
                        self.log(f'grad_stats/{name}_norm', grad_norm, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def build_loaders(
    file_path: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    test_field_type: str = "cal",
    num_synthetic_test_samples: int = 25,
    augment_noise_std: float = 0.0,
    augment_noise_type: str = 'absolute'
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
        num_synthetic_test_samples: Number of samples to reserve for testing
        augment_noise_std: Standard deviation of noise to add for data augmentation (0 = no augmentation)
        augment_noise_type: Type of noise - 'absolute' or 'relative'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data with train/test split already done
    (synth_fields, cal_e_fields, grids, uncal_spars,
     synth_fields_test, cal_e_fields_test, grids_test, uncal_spars_test) = load_data(
        file_path, seed=seed, num_synthetic_test_samples=num_synthetic_test_samples
    )
    
    # Always use synth fields for training/validation
    train_val_dataset = FieldsDataset(x_np=synth_fields, y_np=grids)
    
    # Select test field data
    if test_field_type == "synth":
        test_fields = synth_fields_test
        test_grids = grids_test
    elif test_field_type == "cal":
        test_fields = cal_e_fields_test
        test_grids = grids_test
    elif test_field_type == "uncal":
        test_fields = uncal_spars_test
        test_grids = grids_test
    else:
        raise ValueError(f"Unknown test_field_type: {test_field_type}. Choose from 'synth', 'cal', or 'uncal'")
    
    # Create test dataset
    test_dataset = FieldsDataset(x_np=test_fields, y_np=test_grids)
    
    # Split train/val dataset
    n_total = len(train_val_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_val_dataset, [n_train, n_val], generator=g)
    
    # Apply data augmentation to training set if noise_std > 0
    if augment_noise_std > 0:
        print(f"Applying data augmentation with {augment_noise_type} noise (std={augment_noise_std})")
        train_ds = AugmentedFieldsDataset(train_ds, noise_std=augment_noise_std, noise_type=augment_noise_type, size_multiplier=25)
    
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
    model: LitBCNNUNet,
    test_loader,
    num_cases: int = 10,
    output_dir: str = "figures/test",
    device: str | None = None,
):
    """Run reconstruction on a handful of test samples using backgrounds stored in the model.

    Args:
        model: Trained LitUNet instance with internal backgrounds (bkgs & bkg_sct_fields buffers).
        test_loader: DataLoader providing experimental (x,y) pairs (x: (B,2,24,24), y: (B,2,100,100)).
        num_cases: Number of individual samples to visualize.
        output_dir: Directory to save figures.
        device: Optional device override.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()
    model.enable_dropout()

    collected = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            B = x.shape[0]
            take = min(B, num_cases - collected)
            if take <= 0:
                break
            x = x.to(device)[:take]
            y = y.to(device)[:take]

            MC_count = (15)
            contrasts_pred_K = torch.zeros((MC_count, take, 4, 100, 100), dtype=torch.float32)
            old_contrasts_pred_K = torch.zeros((MC_count, take, 4, 100, 100), dtype=torch.float32)
            for k in range(MC_count):
                contrasts_pred = model(x)  # (take*nbkgs,2,100,100)
                contrasts_pred = contrasts_pred.view(take, 4, 100, 100)
                contrasts_pred_K[k,...] = contrasts_pred.cpu()

            predictions = contrasts_pred_K[:,:,:2,:,:]
            variances = contrasts_pred_K[:,:,2:,:,:]
            mean_pred = torch.mean(predictions, axis=0)
            std_pred = torch.sqrt(torch.mean(torch.exp(variances), axis=0) + torch.mean((predictions[:,:,:2,:,:] - mean_pred.unsqueeze(0))**2, axis=0))

            mae = torch.mean(torch.abs(mean_pred[:,:2,:,:] - y.cpu())).item()
            mse = torch.mean((mean_pred[:,:2,:,:] - y.cpu()) ** 2).item()
            print(f"Samples {collected}-{collected+take} MAE={mae:.4e} MSE={mse:.4e}")
            
            sample_idx = collected
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
            def show(ax, tensor2d, title, cmap='viridis'):
                im = ax.imshow(tensor2d, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            show(axes[0], contrasts_pred_K[0,0,0,...], 'Real Recon 1')
            show(axes[1], contrasts_pred_K[1,0,0,...], 'Real Recon 2')
            show(axes[2], contrasts_pred_K[2,0,0,...], 'Real Recon 3')
            show(axes[3], contrasts_pred_K[3,0,0,...], 'Real Recon 4')
            show(axes[4], contrasts_pred_K[4,0,0,...], 'Real Recon 5')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"MC_dropout_check_{sample_idx}.pdf"), dpi=150)
            plt.close(fig)

            for i in range(take):
                idx_global = collected + i
                # Updated: 2x4 layout (remove first background visualization)
                fig, axes = plt.subplots(2, 4, figsize=(14, 6))
                def show(ax, tensor2d, title, cmap='viridis'):
                    im = ax.imshow(tensor2d, cmap=cmap)
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Row 0 (Real): GT | Mean | Std | Abs Err
                show(axes[0,0], y[i,0].cpu(), 'GT (Real)')
                show(axes[0,1], mean_pred[i,0], 'Mean Recon (Real)')
                show(axes[0,2], std_pred[i,0], 'Std (Real)', cmap='plasma')
                show(axes[0,3], torch.abs(mean_pred[i,0]-y[i,0].cpu()), 'Abs Err (Real)', cmap='magma')
                # Row 1 (Imag): GT | Mean | Std | Abs Err
                show(axes[1,0], y[i,1].cpu(), 'GT (Imag)')
                show(axes[1,1], mean_pred[i,1], 'Mean Recon (Imag)')
                show(axes[1,2], std_pred[i,1], 'Std (Imag)', cmap='plasma')
                show(axes[1,3], torch.abs(mean_pred[i,1]-y[i,1].cpu()), 'Abs Err (Imag)', cmap='magma')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx_global}.pdf"), dpi=150)
                plt.close(fig)
            collected += take
            if collected >= num_cases:
                break
    print(f"Saved {collected} test reconstruction figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train BCNN UNet for EM field reconstruction")
    parser.add_argument("--data", type=str, default="all_data.mat", help="Path to .mat file")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--reg-coef-1", type=float, default=1, help="Regularization coefficient for reg_loss_1")
    parser.add_argument("--reg-coef-2", type=float, default=0.01, help="Regularization coefficient for reg_loss_2")
    parser.add_argument("--mse-coef", type=float, default=0.0, help="MSE loss coefficient (always included in training)")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run test on a checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint for testing. If None, finds latest.")
    parser.add_argument("--experiment-tag", type=str, default="bcnn_experiment_800", help="Tag for experiment (used in logging)")
    parser.add_argument("--test-field-type", type=str, default="synth", choices=["synth", "cal", "uncal"], 
                        help="Type of field data to use for test set (training always uses synth)")
    parser.add_argument("--augment-noise-std", type=float, default=0.1, 
                        help="Standard deviation of noise for data augmentation (0 = no augmentation)")
    parser.add_argument("--augment-noise-type", type=str, default="relative", choices=["absolute", "relative"],
                        help="Type of noise: 'absolute' (fixed std) or 'relative' (proportional to signal)")

    args = parser.parse_args()
    
    EXPERIMENT_TAG = args.experiment_tag

    train_loader, val_loader, test_loader = build_loaders(
        file_path=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        test_field_type=args.test_field_type,
        augment_noise_std=args.augment_noise_std,
        augment_noise_type=args.augment_noise_type
    )

    #args.test_only = True
    if not args.test_only:
        # --- Training Phase ---
        model = LitBCNNUNet(
            in_channels=2,
            out_channels=4,
            base_channels=args.base_channels,
            lr=args.lr,
            experiment_tag=EXPERIMENT_TAG
        )

        logger = TensorBoardLogger(save_dir="lightning_logs", name=EXPERIMENT_TAG)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(logger.log_dir, 'checkpoints'),
            filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="cuda",
            devices=1,
            log_every_n_steps=50,
            logger=logger,
            fast_dev_run=args.fast_dev_run,
            deterministic="warn",
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
            list_of_files = glob.glob(f'lightning_logs/{EXPERIMENT_TAG}/version_*/checkpoints/*.ckpt')
            if not list_of_files:
                raise FileNotFoundError("No checkpoints found to test.")
            ckpt_path = max(list_of_files, key=os.path.getctime)
        
        print(f"Loading model from checkpoint: {ckpt_path}")

    # --- Testing Phase ---
    print(f"--- Starting Test Phase on {ckpt_path} ---")
    
    # First, load the checkpoint to inspect its contents
    # checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # # Extract background tensors from checkpoint
    # bkgs_from_ckpt = checkpoint['state_dict']['bkgs']
    # bkg_sct_fields_from_ckpt = checkpoint['state_dict']['bkg_sct_fields']
        
    # Load model from checkpoint with the background tensors
    model = LitBCNNUNet.load_from_checkpoint(
        ckpt_path, 
        strict=False,
        experiment_tag=EXPERIMENT_TAG
    )
    
    # print summary of the model
    
    test(model=model, test_loader=test_loader)
    # generate_calibration_curve(model=model, test_loader=test_loader, output_dir="figures/test")
    # litmus_test(model=model, in_range_test_loader=test_loader, out_of_range_test_loader=sparam_test_loader)

    print("--- Test Finished ---")


if __name__ == "__main__":
    main()

