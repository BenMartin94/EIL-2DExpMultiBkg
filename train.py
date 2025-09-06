import os
# Set environment variable to use only GPU 0 (RTX 2080 Ti) before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from Unet import UNet
import main as data_main
from MultiBkgDataset import MultiBkgDataset


class FieldsDataset(Dataset):
    """
    Dataset wrapping channels-last numpy arrays.
    X: (N, 24, 24, 2)
    Y: (N, 100, 100, 2)
    Returns tensors in channels-first: X -> (2,24,24), Y -> (2,100,100)
    """

    def __init__(self, x_np: np.ndarray, y_np: np.ndarray, dtype: torch.dtype = torch.float32):
        assert x_np.ndim == 4 and y_np.ndim == 4, "Expected (N,H,W,C) arrays"
        assert x_np.shape[0] == y_np.shape[0], "Mismatched batch sizes"
        assert x_np.shape[-1] == 2 and y_np.shape[-1] == 2, "Expected 2-channel real/imag in last dim"
        self.x = x_np
        self.y = y_np
        self.dtype = dtype

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        # (H,W,C) -> (C,H,W)
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).to(self.dtype)
        y = torch.from_numpy(np.ascontiguousarray(np.transpose(y, (2, 0, 1)))).to(self.dtype)
        return x, y


class LitUNet(pl.LightningModule):
    def __init__(self, in_channels: int = 2, out_channels: int = 2, base_channels: int = 64, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def reconstruction(self, fg_sct, bg_sct, bg_grid):
        """Reconstruct the full fg_grid from fg_sct-bg_sct and bg_grid
        by adding the background field back into the predicted contrast for each bg

        """
        # fg_sct: (B,2,24,24)
        # bg_sct: (B,nbkgs,2,24,24)
        # bg_grid: (B,nbkgs,2,100,100)
        B, nbkgs, _, H, W = bg_sct.shape
        fg_sct_expanded = fg_sct.unsqueeze(1).expand(-1, nbkgs, -1, -1, -1)  # (B,nbkgs,2,24,24)
        contrast = fg_sct_expanded - bg_sct  # (B,nbkgs,2,24,24)
        contrast_reshaped = contrast.view(B * nbkgs, 2, H, W)
        pred_contrast_grid = self(contrast_reshaped)  # (B*nbkgs,2,100,100)
        pred_contrast_grid = pred_contrast_grid.view(B, nbkgs, 2, 100, 100)
        # Reconstruct fg_grid by adding bg_grid back
        # fg_grid = pred_contrast_grid * bg_grid + bg_grid
        # first convert pred_contrast_grid and bg_grid to complex to handle complex arithmetic correctly
        # Note: For channels-first format (B,nbkgs,2,H,W), real is channel 0, imag is channel 1
        pred_contrast_complex = pred_contrast_grid[:, :, 0] + 1j * pred_contrast_grid[:, :, 1]
        bg_grid_complex = bg_grid[:, :, 0] + 1j * bg_grid[:, :, 1]
        fg_grid_complex = pred_contrast_complex * bg_grid_complex + bg_grid_complex
        # Convert back to channels-first format with torch.stack
        fg_grid = torch.stack([fg_grid_complex.real, fg_grid_complex.imag], dim=2)  # (B,nbkgs,2,100,100)
        fg_grid_mu = fg_grid.mean(dim=1)  # (B,2,100,100)
        fg_grid_deviation = torch.std(fg_grid, dim=1)  # (B,2,100,100)
        return fg_grid, fg_grid_mu, fg_grid_deviation


    def training_step(self, batch, batch_idx: int):
        x, y, fg_grid, bg_grid, fg_sct_field, bg_sct_field = batch
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
                mae = torch.mean(torch.abs(y_hat - y))
                self.log("train_mae", mae, on_step=True)
                
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y, fg_grid, bg_grid, fg_sct_field, bg_sct_field = batch

        # Get all backgrounds from the dataset for proper reconstruction evaluation
        dataset = self.trainer.val_dataloaders.dataset
        if hasattr(dataset, 'dataset'):  # Handle case where it's wrapped in a Subset
            dataset = dataset.dataset
        all_bg_grids, all_bg_fields = dataset.get_backgrounds()
        
        # Move to device
        all_bg_grids = all_bg_grids.to(self.device)
        all_bg_fields = all_bg_fields.to(self.device)
        
        
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        
        # Log validation metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Additional validation metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat - y))
            mse = torch.mean((y_hat - y) ** 2)
            self.log("val_mae", mae, on_step=False, on_epoch=True)
            self.log("val_mse", mse, on_step=False, on_epoch=True)
            
            # Log per-channel metrics
            for c in range(y.shape[1]):
                channel_mae = torch.mean(torch.abs(y_hat[:, c] - y[:, c]))
                self.log(f"val_mae_channel_{c}", channel_mae, on_step=False, on_epoch=True)
        
        # Log images to TensorBoard every 5 epochs
        if batch_idx == 0 and (self.current_epoch + 1) % 5 == 0:
            self._log_images_to_tensorboard(x, y, y_hat)
            self._log_reconstruction_images(fg_grid, all_bg_grids, fg_sct_field, all_bg_fields)
            
        # Log weight histograms every 10 epochs
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            self._log_weight_histograms()
            
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
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="viridis")
            axes[i, 0].set_title("Input (real)")
            axes[i, 1].imshow(y[i, 0].cpu(), cmap="viridis")
            axes[i, 1].set_title("Target (real)")
            axes[i, 2].imshow(y_hat[i, 0].detach().cpu(), cmap="viridis")
            axes[i, 2].set_title("Pred (real)")
            for j in range(3):
                axes[i, j].axis("off")
        plt.tight_layout()
        fig_dir = f"figures/epoch_{self.current_epoch}"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "val_examples.png")
        plt.savefig(fig_path)
        plt.close(fig)

    def _log_reconstruction_images(self, fg_grid, all_bg_grids, fg_sct_field, all_bg_fields):
        """Log reconstruction images using all backgrounds from the dataset
        
        Args:
            fg_grid: Foreground grids for the batch (B, 2, 100, 100)
            all_bg_grids: All background grids from dataset (n_backgrounds, 2, 100, 100)
            fg_sct_field: Foreground scattered fields for the batch (B, 2, 24, 24) 
            all_bg_fields: All background scattered fields from dataset (n_backgrounds, 2, 24, 24)
        """
        with torch.no_grad():
            batch_size = fg_grid.size(0)
            n_backgrounds = all_bg_grids.size(0)
            
            # Expand backgrounds to match batch size
            # all_bg_grids: (n_backgrounds, 2, 100, 100) -> (batch_size, n_backgrounds, 2, 100, 100)
            bg_grid_expanded = all_bg_grids.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            
            # all_bg_fields: (n_backgrounds, 2, 24, 24) -> (batch_size, n_backgrounds, 2, 24, 24)  
            bg_sct_field_expanded = all_bg_fields.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            
            # Perform reconstruction using all available backgrounds
            fg_grid_reconstructed, fg_grid_mean, fg_grid_std = self.reconstruction(
                fg_sct_field, bg_sct_field_expanded, bg_grid_expanded
            )
            
            n_examples = min(4, fg_grid.size(0))
            
            # Log to TensorBoard with proper normalization
            if self.logger and hasattr(self.logger, 'experiment'):
                
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
                
                # Log original foreground grids (real part)
                orig_fg_real = torch.stack([fg_grid[i, 0].cpu().unsqueeze(0) for i in range(n_examples)])
                orig_fg_real_norm = normalize_for_tensorboard(orig_fg_real)
                self.logger.experiment.add_images(
                    'reconstruction/original_fg_real', orig_fg_real_norm, self.current_epoch
                )
                
                # Log mean reconstruction (real part)
                mean_recon_real = torch.stack([fg_grid_mean[i, 0].cpu().unsqueeze(0) for i in range(n_examples)])
                mean_recon_real_norm = normalize_for_tensorboard(mean_recon_real)
                self.logger.experiment.add_images(
                    'reconstruction/mean_reconstruction_real', mean_recon_real_norm, self.current_epoch
                )
                
                # Log standard deviation of reconstructions (real part)
                std_recon_real = torch.stack([fg_grid_std[i, 0].cpu().unsqueeze(0) for i in range(n_examples)])
                std_recon_real_norm = normalize_for_tensorboard(std_recon_real)
                self.logger.experiment.add_images(
                    'reconstruction/std_reconstruction_real', std_recon_real_norm, self.current_epoch
                )
                
                # Log individual reconstructions for first background (real part)
                first_bg_recon_real = torch.stack([fg_grid_reconstructed[i, 0, 0].cpu().unsqueeze(0) for i in range(n_examples)])
                first_bg_recon_real_norm = normalize_for_tensorboard(first_bg_recon_real)
                self.logger.experiment.add_images(
                    'reconstruction/first_bg_reconstruction_real', first_bg_recon_real_norm, self.current_epoch
                )
                
                # Also log imaginary parts
                orig_fg_imag = torch.stack([fg_grid[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
                orig_fg_imag_norm = normalize_for_tensorboard(orig_fg_imag)
                self.logger.experiment.add_images(
                    'reconstruction/original_fg_imag', orig_fg_imag_norm, self.current_epoch
                )
                
                mean_recon_imag = torch.stack([fg_grid_mean[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
                mean_recon_imag_norm = normalize_for_tensorboard(mean_recon_imag)
                self.logger.experiment.add_images(
                    'reconstruction/mean_reconstruction_imag', mean_recon_imag_norm, self.current_epoch
                )
                
                std_recon_imag = torch.stack([fg_grid_std[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
                std_recon_imag_norm = normalize_for_tensorboard(std_recon_imag)
                self.logger.experiment.add_images(
                    'reconstruction/std_reconstruction_imag', std_recon_imag_norm, self.current_epoch
                )
            
            # Create comparison figure showing original vs reconstructed vs mean vs std
            fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4 * n_examples))
            if n_examples == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(n_examples):
                # Original foreground (real part)
                axes[i, 0].imshow(fg_grid[i, 0].cpu(), cmap="viridis")
                axes[i, 0].set_title("Original FG (real)")
                
                # Mean reconstruction (real part)
                axes[i, 1].imshow(fg_grid_mean[i, 0].cpu(), cmap="viridis")
                axes[i, 1].set_title("Mean Reconstruction (real)")
                
                # First background reconstruction (real part)
                axes[i, 2].imshow(fg_grid_reconstructed[i, 0, 0].cpu(), cmap="viridis")
                axes[i, 2].set_title("First BG Recon (real)")
                
                # Standard deviation (real part)
                axes[i, 3].imshow(fg_grid_std[i, 0].cpu(), cmap="plasma")
                axes[i, 3].set_title("Std Dev (real)")
                
                for j in range(4):
                    axes[i, j].axis("off")
                    
            plt.tight_layout()
            fig_dir = f"figures/epoch_{self.current_epoch}"
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, "reconstruction_examples.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            # Log reconstruction metrics
            with torch.no_grad():
                # Compute reconstruction error metrics
                recon_mae = torch.mean(torch.abs(fg_grid_mean - fg_grid))
                recon_mse = torch.mean((fg_grid_mean - fg_grid) ** 2)
                mean_std = torch.mean(fg_grid_std)
                
                self.log("recon_mae", recon_mae, on_step=False, on_epoch=True)
                self.log("recon_mse", recon_mse, on_step=False, on_epoch=True)
                self.log("recon_mean_std", mean_std, on_step=False, on_epoch=True)
                
                # Log per-channel reconstruction metrics
                for c in range(fg_grid.shape[1]):
                    channel_recon_mae = torch.mean(torch.abs(fg_grid_mean[:, c] - fg_grid[:, c]))
                    self.log(f"recon_mae_channel_{c}", channel_recon_mae, on_step=False, on_epoch=True)

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


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    grids, uncal_spars, cal_e_fields, synth_fields = data_main.read_mat_file(file_path)
    # Convert complex to real/imag last dim
    grids = data_main.split_complex_to_real_imag(grids)       # (N,100,100,2)
    synth_fields = data_main.split_complex_to_real_imag(synth_fields)  # (N,24,24,2)
    cal_e_fields = data_main.split_complex_to_real_imag(cal_e_fields)  # (N,24,24,2)

    # Cast to float32 for training
    grids = grids.astype(np.float32, copy=False)
    synth_fields = synth_fields.astype(np.float32, copy=False)
    cal_e_fields = cal_e_fields.astype(np.float32, copy=False)
    return cal_e_fields, grids


def build_loaders(fields: np.ndarray, grids: np.ndarray, batch_size: int, val_split: float, num_workers: int, seed: int):
    dataset = MultiBkgDataset(fields=fields, grids=grids, n_backgrounds=5)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train UNet to map synth fields (24x24x2) -> grids (100x100x2)")
    parser.add_argument("--data", type=str, default="all_data.mat", help="Path to .mat file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    fields, grids = load_data(args.data)
    train_loader, val_loader = build_loaders(fields, grids, args.batch_size, args.val_split, args.num_workers, args.seed)

    model = LitUNet(in_channels=2, out_channels=2, base_channels=args.base_channels, lr=args.lr)

    ckpt_dir = os.path.join("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Set up TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="exp_experiment",
        version=None,  # Auto-increment version
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,  # Use 1 GPU (will be GPU 0 due to CUDA_VISIBLE_DEVICES)
        default_root_dir=ckpt_dir,
        log_every_n_steps=10,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

