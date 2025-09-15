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

from Unet import UNet
import main as data_main
from MultiBkgDataset import MultiBkgDataset
from Litmus_test import litmus_test


class FieldsDataset(Dataset):
    """
    Dataset wrapping channels-last numpy arrays.
    X: (N, 24, 24, 2)
    Y: (N, 100, 100, 2)
    Returns tensors in channels-first: X -> (2,24,24), Y -> (2,100,100)
    """

    def __init__(self, x_np: np.ndarray, y_np: np.ndarray, dtype: torch.dtype = torch.float32):
        assert x_np.ndim == 4 and y_np.ndim == 4, "Expected (N,H,W) arrays"
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
    def __init__(
        self,
        bkgs: torch.Tensor | None = None,
        bkg_sct_fields: torch.Tensor | None = None,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        # Save hyperparameters, ignoring large tensors.
        self.save_hyperparameters(ignore=['bkgs', 'bkg_sct_fields'])
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        self.criterion = torch.nn.MSELoss()

        # Register buffers. They will be populated from the checkpoint if loading.
        self.register_buffer('bkgs', bkgs)
        self.register_buffer('bkg_sct_fields', bkg_sct_fields)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

    
    def get_mask(self, bg, debug=False):
        """Generate a weight mask from a background field.
        Uses Canny edge detection on a safely normalized image. Constant or near-constant
        images produce an all-ones weight mask (no exclusion).
        """
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        bg = np.asarray(bg)
        bg_min = float(np.nanmin(bg))
        bg_max = float(np.nanmax(bg))
        rng = bg_max - bg_min
        if not np.isfinite(rng) or rng < 1e-12:
            # Degenerate / constant image: just use zeros image for edges later
            bg_normed = np.zeros_like(bg, dtype=np.float32)
        else:
            bg_normed = (bg - bg_min) / rng
            # Guard against tiny numerical drift
            bg_normed = np.clip(bg_normed, 0.0, 1.0)

        # Convert to uint8 for OpenCV operations
        bg_uint8 = (bg_normed * 255.0).round().astype(np.uint8)

        # Find edges in background (on constant images this will be all zeros)
        edges = cv2.Canny(bg_uint8, 100, 200)

        # Dilate edges to create mask of regions to exclude
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)

        # weights: 0 where edges (dilated), 1 elsewhere
        weights = np.where(mask == 255, 0, 1).astype(np.float32)

        if debug:
            fig, ax = plt.subplots(1, 4, figsize=(10, 3))
            ax[0].imshow(bg, cmap='viridis'); ax[0].set_title('bg'); ax[0].axis('off')
            ax[1].imshow(bg_normed, cmap='viridis'); ax[1].set_title('normed'); ax[1].axis('off')
            ax[2].imshow(edges, cmap='gray'); ax[2].set_title('edges'); ax[2].axis('off')
            ax[3].imshow(weights, cmap='gray'); ax[3].set_title('weights'); ax[3].axis('off')
            plt.tight_layout(); plt.show()
        return weights

    def get_bkg_masks(self, bg_grids):
        """Get masks for each background grid to identify valid regions.
        Supports inputs of shape:
          (nbkgs, 2, H, W) or (B, nbkgs, 2, H, W)
        Returns a tensor/ndarray of same leading dimensionality with masks in {0,1}.
        """
        import cv2
        import numpy as np
        import torch

        # Accept torch or numpy input
        is_torch = isinstance(bg_grids, torch.Tensor)
        device = bg_grids.device if is_torch else None
        if is_torch:
            grids_np = bg_grids.detach().cpu().numpy()
        else:
            grids_np = bg_grids

        if grids_np.ndim == 4:  # (nbkgs,2,H,W)
            grids_np = grids_np[None, ...]  # add batch dim -> (1,nbkgs,2,H,W)
            added_batch = True
        elif grids_np.ndim == 5:  # (B,nbkgs,2,H,W)
            added_batch = False
        else:
            raise ValueError(f"bg_grids must have 4 or 5 dims, got {grids_np.shape}")

        B, nbkgs, _, H, W = grids_np.shape
        masks_np = np.zeros((B, nbkgs, 2, H, W), dtype=np.float32)

        for b in range(B):
            for k in range(nbkgs):
                bg_real = grids_np[b, k, 0]
                bg_imag = grids_np[b, k, 1]
                masks_np[b, k, 0] = self.get_mask(bg_real)
                masks_np[b, k, 1] = self.get_mask(bg_imag)

        # Remove artificial batch dim if originally absent
        if added_batch:
            masks_np = masks_np[0]  # (nbkgs,2,H,W)

        if is_torch:
            return torch.from_numpy(masks_np).to(device)
        return masks_np


    
    
    def reconstruction(self, fg_sct, bg_sct, bg_grid):
        """Reconstruct the full fg_grid from fg_sct-bg_sct and bg_grid
        by adding the background field back into the predicted contrast for each bg

        """
        # fg_sct: (B,2,24,24)
        # bg_sct: (B,nbkgs,2,24,24)
        # bg_grid: (B,nbkgs,2,100,100)
        B, nbkgs, _, H, W = bg_sct.shape
        # Updated: get_bkg_masks now returns shape (B,nbkgs,2,100,100)
        weights_for_mean = self.get_bkg_masks(bg_grid)  # (B,nbkgs,2,100,100) or (nbkgs,2,100,100)
        if weights_for_mean.dim() == 4:  # no batch dim, expand to match
            weights_for_mean = weights_for_mean.unsqueeze(0).expand(B, -1, -1, -1, -1)

        fg_sct_expanded = fg_sct.unsqueeze(1).expand(-1, nbkgs, -1, -1, -1)  # (B,nbkgs,2,24,24)
        contrast = fg_sct_expanded - bg_sct  # (B,nbkgs,2,24,24)
        contrast_reshaped = contrast.view(B * nbkgs, 2, H, W)
        pred_contrast_grid = self(contrast_reshaped)  # (B*nbkgs,2,100,100)
        pred_contrast_grid = pred_contrast_grid.view(B, nbkgs, 2, 100, 100)

        # Complex reconstruction
        pred_contrast_complex = pred_contrast_grid[:, :, 0] + 1j * pred_contrast_grid[:, :, 1]
        bg_grid_complex = bg_grid[:, :, 0] + 1j * bg_grid[:, :, 1]
        fg_grid_complex = pred_contrast_complex * bg_grid_complex + bg_grid_complex
        fg_grid = torch.stack([fg_grid_complex.real, fg_grid_complex.imag], dim=2)  # (B,nbkgs,2,100,100)

        # Unweighted stats (kept for backward compatibility)
        fg_grid_mu = fg_grid.mean(dim=1)  # (B,2,100,100)
        fg_grid_deviation = torch.std(fg_grid, dim=1)  # (B,2,100,100)

        # Weighted stats (currently unused in return but computed with new batched masks)
        weighted_sum = torch.sum(fg_grid * weights_for_mean, dim=1)  # (B,2,100,100)
        weight_totals = torch.sum(weights_for_mean, dim=1)  # (B,2,100,100)
        mask = weight_totals > 0
        weighted_mean = torch.zeros_like(fg_grid_mu)
        weighted_mean[mask] = weighted_sum[mask] / (weight_totals[mask] + 1e-8)
        weighted_sq_sum = torch.sum((fg_grid - weighted_mean.unsqueeze(1)) ** 2 * weights_for_mean, dim=1)
        weighted_var = weighted_sq_sum / (weight_totals + 1e-8)
        weighted_std = torch.sqrt(weighted_var)
        # (If desired later, can swap fg_grid_mu/fg_grid_deviation with weighted versions.)

        return fg_grid, weighted_mean, weighted_std
    
    def bkgs_mean_error(self, fg_sct, bg_sct, bg_grid):
        """Using the backgrounds, assess the quality of the contrast prediction around the bkgs return mse across all bkgs
        fg_sct: (B,2,24,24)
        bg_sct: (B,nbkgs,2,24,24)
        bg_grid: (B,nbkgs,2,100,100)
        returns mse: (B,nbkgs)
        """
        # start by getting all the bkgs masks well need
        bkg_masks = self.get_bkg_masks(bg_grid)  # (B,nbkgs,2,100,100)
        B, nbkgs, _, H, W = bg_sct.shape
        # generate the contrast predictions for this batch
        fg_sct_expanded = fg_sct.unsqueeze(1).expand(-1, nbkgs, -1, -1, -1)  # (B,nbkgs,2,24,24)
        contrast = fg_sct_expanded - bg_sct  # (B,nbkgs,2,24,24)
        contrast_reshaped = contrast.view(B * nbkgs, 2, H, W)
        pred_epsilon_grid = self.reconstruction(fg_sct, bg_sct, bg_grid)[1]  # (B,2,100,100)
        pred_chi_grid = self(contrast_reshaped)  # (B*nbkgs,2,100,100)
        pred_chi_grid = pred_chi_grid.view(B, nbkgs, 2, 100, 100)

        pred_epsilon_complex = pred_epsilon_grid[:, 0] + 1j * pred_epsilon_grid[:, 1]
        pred_epsilon_complex = pred_epsilon_complex.unsqueeze(1).expand(-1, nbkgs, -1, -1)  # (B,nbkgs,100,100)
        pred_chi_complex = pred_chi_grid[:, :, 0] + 1j * pred_chi_grid[:, :, 1]
        pred_bkg_complex = pred_epsilon_complex/(pred_chi_complex + 1) # (B,nbkgs,100,100)

        # now compute the mse between the predicted bkg and the actual bkg across all bkgs
        pred_bkg_grid = torch.stack([pred_bkg_complex.real, pred_bkg_complex.imag], dim=2)  # (B,nbkgs,2,100,100)
        sq_diff = (pred_bkg_grid - bg_grid) ** 2
        bkg_masks_inversed = 1.0 - bkg_masks  # (B,nbkgs,2,100,100); 1 where edges are
        # Weighted mean over channel + spatial dims (per bkg)
        num = (sq_diff * bkg_masks_inversed).sum(dim=(2, 3, 4))  # (B, nbkgs)
        den = bkg_masks_inversed.sum(dim=(2, 3, 4)).clamp_min(1e-8)  # (B, nbkgs)
        mse = num / den  # (B, nbkgs)
        mse = torch.mean(mse, dim=1)  # (B,) average over bkgs
        return mse

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

        # Use the backgrounds stored in the model for reconstruction evaluation
        all_bg_grids = self.bkgs
        all_bg_fields = self.bkg_sct_fields
        
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
        if batch_idx == 0 and (self.current_epoch+1) % 5 == 0:
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
                
                # Log original foreground grids (imaginary part)
                orig_fg_imag = torch.stack([fg_grid[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
                orig_fg_imag_norm = normalize_for_tensorboard(orig_fg_imag)
                self.logger.experiment.add_images(
                    'reconstruction/original_fg_imag', orig_fg_imag_norm, self.current_epoch
                )
                
                # Log mean reconstruction (imaginary part)
                mean_recon_imag = torch.stack([fg_grid_mean[i, 1].cpu().unsqueeze(0) for i in range(n_examples)])
                mean_recon_imag_norm = normalize_for_tensorboard(mean_recon_imag)
                self.logger.experiment.add_images(
                    'reconstruction/mean_reconstruction_imag', mean_recon_imag_norm, self.current_epoch
                )
                
                # Log standard deviation of reconstructions (imaginary part)
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
                recon_mean_std = torch.mean(fg_grid_std)
                
                self.log("recon_mae", recon_mae, on_step=False, on_epoch=True)
                self.log("recon_mse", recon_mse, on_step=False, on_epoch=True)
                self.log("recon_mean_std", recon_mean_std, on_step=False, on_epoch=True)
                
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
    uncal_spars = data_main.split_complex_to_real_imag(uncal_spars)  # (N,24,24,2)

    # Cast to float32 for training
    grids = grids.astype(np.float32, copy=False)
    synth_fields = synth_fields.astype(np.float32, copy=False)
    cal_e_fields = cal_e_fields.astype(np.float32, copy=False)
    return synth_fields, cal_e_fields, grids, uncal_spars


def build_loaders(file_path:str, batch_size: int, val_split: float, num_workers: int, seed: int):
    synth_fields, cal_e_fields, grids, uncal_spars = load_data(file_path)
    synth_dataset = MultiBkgDataset(fields=synth_fields, grids=grids, n_backgrounds=25)
    exp_dataset = FieldsDataset(x_np=cal_e_fields, y_np=grids)
    sparam_dataset = FieldsDataset(x_np=uncal_spars, y_np=grids)
    n_total = len(synth_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(synth_dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    sparam_test_loader = DataLoader(sparam_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, sparam_test_loader


def test(
    model: LitUNet,
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

    collected = 0
    with torch.no_grad():
        for batch in test_loader:
            # FieldsDataset returns (x,y)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[:2]
            else:
                raise RuntimeError("Expected (x,y) batch from test_loader")
            B = x.shape[0]
            take = min(B, num_cases - collected)
            if take <= 0:
                break
            x = x.to(device)[:take]
            y = y.to(device)[:take]

            nbkgs = model.bkgs.shape[0]
            assert nbkgs > 0, "Model has no backgrounds stored for reconstruction."
            bg_grid = model.bkgs.unsqueeze(0).expand(take, -1, -1, -1, -1)          # (take,nbkgs,2,100,100)
            bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(take, -1, -1, -1, -1)  # (take,nbkgs,2,24,24)

            per_bkg_recons, mean_recon, std_recon = model.reconstruction(x, bg_sct, bg_grid)
            mae = torch.mean(torch.abs(mean_recon - y)).item()
            mse = torch.mean((mean_recon - y) ** 2).item()
            print(f"Samples {collected}-{collected+take} MAE={mae:.4e} MSE={mse:.4e}")

            for i in range(take):
                idx_global = collected + i
                fig, axes = plt.subplots(2, 5, figsize=(18, 6))
                fig.suptitle(f"Test Sample {idx_global}")
                def show(ax, tensor2d, title, cmap='viridis'):
                    im = ax.imshow(tensor2d, cmap=cmap)
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                show(axes[0,0], y[i,0].cpu(), 'GT (Real)')
                show(axes[0,1], mean_recon[i,0].cpu(), 'Mean Recon (Real)')
                show(axes[0,2], bg_grid[i,0,0].cpu(), 'First BG (Real)')
                show(axes[1,0], y[i,1].cpu(), 'GT (Imag)')
                show(axes[1,1], mean_recon[i,1].cpu(), 'Mean Recon (Imag)')
                show(axes[1,2], bg_grid[i,0,1].cpu(), 'First BG (Imag)')
                show(axes[0,3], std_recon[i,0].cpu(), 'Std (Real)', cmap='plasma')
                show(axes[0,4], torch.abs(mean_recon[i,0]-y[i,0]).cpu(), 'Abs Err (Real)', cmap='magma')
                show(axes[1,3], std_recon[i,1].cpu(), 'Std (Imag)', cmap='plasma')
                show(axes[1,4], torch.abs(mean_recon[i,1]-y[i,1]).cpu(), 'Abs Err (Imag)', cmap='magma')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx_global}.png"), dpi=150)
                plt.close(fig)
            collected += take
            if collected >= num_cases:
                break
    print(f"Saved {collected} test reconstruction figures to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train UNet to map synth fields (24x24x2) -> grids (100x100x2)")
    parser.add_argument("--data", type=str, default="all_data.mat", help="Path to .mat file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--debug-recon", action="store_true", help="Run a single debug reconstruction before training")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run test on a checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint for testing. If None, finds latest.")

    EXPERIMENT_TAG = "mbg_train_synth_test_cal_exp_25bkgs"

    args = parser.parse_args()

    train_loader, val_loader, test_loader, sparam_test_loader = build_loaders(
        file_path=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if not args.test_only:
        # --- Training Phase ---
        train_base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
        if not hasattr(train_base_ds, 'get_backgrounds'):
            raise RuntimeError("Training dataset cannot provide backgrounds.")
        
        bg_grids_tensor, bg_fields_tensor = train_base_ds.get_backgrounds()

        model = LitUNet(
            bkgs=bg_grids_tensor,
            bkg_sct_fields=bg_fields_tensor,
            in_channels=2,
            out_channels=2,
            base_channels=args.base_channels,
            lr=args.lr
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
            accelerator="cpu",
            devices=1,
            log_every_n_steps=10,
            logger=logger,
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
            list_of_files = glob.glob(f'lightning_logs/{EXPERIMENT_TAG}/version_*/checkpoints/*.ckpt')
            if not list_of_files:
                raise FileNotFoundError("No checkpoints found to test.")
            ckpt_path = max(list_of_files, key=os.path.getctime)
        
        print(f"Loading model from checkpoint: {ckpt_path}")

    # --- Testing Phase ---
    print(f"--- Starting Test Phase on {ckpt_path} ---")
    
    # First, load the checkpoint to inspect its contents
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Extract background tensors from checkpoint
    bkgs_from_ckpt = checkpoint['state_dict']['bkgs']
    bkg_sct_fields_from_ckpt = checkpoint['state_dict']['bkg_sct_fields']
        
    # Load model from checkpoint with the background tensors
    model = LitUNet.load_from_checkpoint(
        ckpt_path, 
        strict=False,
        bkgs=bkgs_from_ckpt,
        bkg_sct_fields=bkg_sct_fields_from_ckpt
    )
    
    print(f"Successfully loaded model with {model.bkgs.shape[0]} backgrounds")
    
    #test(model=model, test_loader=test_loader)
    litmus_test(model=model, in_range_test_loader=test_loader, out_of_range_test_loader=sparam_test_loader)

    print("--- Test Finished ---")


if __name__ == "__main__":
    main()

