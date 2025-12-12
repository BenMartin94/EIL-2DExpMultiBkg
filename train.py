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

from src.Unet import UNet, bayesian_loss
from src.MultiBkgDataset import LimitedExampleMultiBkgDataset, MultiBkgDataset
from src.data_loader import load_data, FieldsDataset
from src.uncertainty_cal_eval import error_std_correlation


class LitUNet(pl.LightningModule):
    def __init__(
        self,
        bkgs: torch.Tensor | None = None,
        bkg_sct_fields: torch.Tensor | None = None,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        lr: float = 1e-3,
        experiment_tag: str = "experiment",
    ):
        super().__init__()
        # Save hyperparameters, ignoring large tensors.
        self.save_hyperparameters(ignore=['bkgs', 'bkg_sct_fields'])
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        self.criterion = torch.nn.MSELoss()  # Use this for standard UNet with 2 output channels
        self.experiment_tag = experiment_tag

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


    
    
    def reconstruction(self, fg_sct, bg_sct, bg_grid, filter_method='weighted'):
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

        if nbkgs == 1:
            # If only one background, no need to compute weighted stats
            return fg_grid, fg_grid[:, 0], None

        # Unweighted stats (kept for backward compatibility)
        fg_grid_mu = fg_grid.mean(dim=1)  # (B,2,100,100)
        fg_grid_deviation = torch.std(fg_grid, dim=1)  # (B,2,100,100)
        if filter_method == 'none':
            return fg_grid, fg_grid_mu, fg_grid_deviation
        elif filter_method == 'weighted':
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
        else:
            raise ValueError(f"Unknown filter_method: {filter_method}")
    
    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

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
        pred_chi_grid = pred_chi_grid.view(B, nbkgs, 4, 100, 100)

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
                mae = torch.mean(torch.abs(y_hat[:,:2,:,:] - y))
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
            self._log_reconstruction_images(fg_grid, all_bg_grids, fg_sct_field, all_bg_fields)
            
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
                
                # Log standard deviation of reconstructions (real part) - only if available
                if fg_grid_std is not None:
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
                
                # Log standard deviation of reconstructions (imaginary part) - only if available
                if fg_grid_std is not None:
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
                
                # Standard deviation (real part) - only if available
                if fg_grid_std is not None:
                    axes[i, 3].imshow(fg_grid_std[i, 0].cpu(), cmap="plasma")
                    axes[i, 3].set_title("Std Dev (real)")
                else:
                    # If no std available, show a placeholder
                    axes[i, 3].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[i, 3].transAxes)
                    axes[i, 3].set_title("Std Dev (N/A)")
                
                for j in range(4):
                    axes[i, j].axis("off")
                    
            plt.tight_layout()
            fig_dir = f"figures/{self.experiment_tag}/epoch_{self.current_epoch}"
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, "reconstruction_examples.pdf")
            plt.savefig(fig_path)
            plt.close(fig)
            
            # Log reconstruction metrics
            with torch.no_grad():
                # Compute reconstruction error metrics
                recon_mae = torch.mean(torch.abs(fg_grid_mean - fg_grid))
                recon_mse = torch.mean((fg_grid_mean - fg_grid) ** 2)
                
                self.log("recon_mae", recon_mae, on_step=False, on_epoch=True)
                self.log("recon_mse", recon_mse, on_step=False, on_epoch=True)
                
                # Only log std metrics if available (requires multiple backgrounds)
                if fg_grid_std is not None:
                    recon_mean_std = torch.mean(fg_grid_std)
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


def build_loaders(file_path:str, batch_size: int, val_split: float, num_workers: int, seed: int, num_backgrounds: int = 25, steps_per_epoch: int = None, num_synthetic_test_samples: int = 25):
    # Load data with train/test split already done
    (synth_fields, cal_e_fields, grids, uncal_spars,
     synth_fields_test, cal_e_fields_test, grids_test, uncal_spars_test) = load_data(
        file_path, seed=seed, num_synthetic_test_samples=num_synthetic_test_samples
    )

    # cal_e_fields and uncal_spars are only used for testing so there is no need to use their *test* versions, those were only created for clarity and symmetry.
    print(f"length of synth dataset: {len(synth_fields)}")

    synth_dataset = MultiBkgDataset(fields=synth_fields, grids=grids, n_backgrounds=num_backgrounds)
    synth_test_dataset = FieldsDataset(x_np=synth_fields_test, y_np=grids_test)
    exp_dataset = FieldsDataset(x_np=cal_e_fields, y_np=grids)
    sparam_dataset = FieldsDataset(x_np=uncal_spars, y_np=grids)
    n_total = len(synth_dataset)
    
    # Handle small datasets: if dataset is too small for a split, use all for training
    if n_total < 5:
        print(f"Warning: Dataset size ({n_total}) is very small. Using all data for training, validation will use the same data.")
        n_train = n_total
        n_val = n_total
        g = torch.Generator().manual_seed(seed)
        train_ds = synth_dataset
        val_ds = synth_dataset
    else:
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        # Ensure at least 1 sample in training
        if n_train < 1:
            n_train = 1
            n_val = n_total - 1
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(synth_dataset, [n_train, n_val], generator=g)

    # If steps_per_epoch is specified, create a custom sampler that repeats/limits data
    if steps_per_epoch is not None:
        num_samples = steps_per_epoch * batch_size
        # Create a sampler that will provide exactly num_samples samples per epoch
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples, generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size//2, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Deterministic shuffling for test loaders by using explicit generators
    test_gen = torch.Generator().manual_seed(seed + 100)
    sparam_gen = torch.Generator().manual_seed(seed + 200)
    test_loader = DataLoader(
        exp_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=test_gen,
    )
    synth_test_loader = DataLoader(
        synth_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    sparam_test_loader = DataLoader(
        sparam_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=sparam_gen,
    )

    return train_loader, val_loader, synth_test_loader,test_loader, sparam_test_loader


def test(
    model: LitUNet,
    test_loader,
    num_cases: int = 30,
    output_dir: str = "figures/test",
    device: str | None = None,
    percent_noise_level: float = 0.01,
):
    """Run reconstruction on a handful of test samples using backgrounds stored in the model.

    Args:
        model: Trained LitUNet instance with internal backgrounds (bkgs & bkg_sct_fields buffers).
        test_loader: DataLoader providing experimental (x,y) pairs (x: (B,2,24,24), y: (B,2,100,100)).
        num_cases: Number of individual samples to visualize.
        output_dir: Directory to save figures.
        device: Optional device override.
        percent_noise_level: Percentage of signal power to use as noise level when adding noise to inputs. 0.1 for 10%
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    collected = 0
    
    # Lists to collect all predictions for correlation analysis
    all_mean_recons = []
    all_targets = []
    all_std_recons = []
    
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
            
            
            # Compute signal power for each sample in the batch
            signal_power = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True)  # (take, 1, 1, 1)
            
            # Calculate noise standard deviation based on percentage
            noise_std = torch.sqrt(signal_power * percent_noise_level)
            
            # Generate Gaussian noise with the calculated std
            noise = torch.randn_like(x) * noise_std
            
            # Add noise to input
            x = x + noise

            nbkgs = model.bkgs.shape[0]
            assert nbkgs > 0, "Model has no backgrounds stored for reconstruction."
            bg_grid = model.bkgs.unsqueeze(0).expand(take, -1, -1, -1, -1)          # (take,nbkgs,2,100,100)
            bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(take, -1, -1, -1, -1)  # (take,nbkgs,2,24,24)
            per_bkg_recons, mean_recon, std_recon = model.reconstruction(x, bg_sct, bg_grid)

            # Collect for correlation analysis
            all_mean_recons.append(mean_recon.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_std_recons.append(std_recon.cpu().numpy())


            scts = x.unsqueeze(1).expand(-1, nbkgs, -1, -1, -1) - bg_sct  # (take,nbkgs,2,24,24)
            contrasts_pred = model(scts.view(take * nbkgs, 2, 24, 24))  # (take*nbkgs,2,100,100)
            contrasts_pred = contrasts_pred.view(take, nbkgs, 2, 100, 100)
            y_complex = y[:,0] + 1j * y[:,1]
            y_complex = y_complex.unsqueeze(1).expand(-1, nbkgs, -1, -1)  # (take,nbkgs,100,100)
            bg_complex = bg_grid[:,:,0] + 1j * bg_grid[:,:,1]
            contrasts_gt_complex = (y_complex - bg_complex) / bg_complex
            contrasts_gt = torch.stack([contrasts_gt_complex.real, contrasts_gt_complex.imag], dim=2)  # (take,nbkgs,2,100,100)

            mae = torch.mean(torch.abs(mean_recon - y)).item()
            mse = torch.mean((mean_recon - y) ** 2).item()
            print(f"Samples {collected}-{collected+take} MAE={mae:.4e} MSE={mse:.4e}")

            for i in range(take):
                idx_global = collected + i
                
                # Plot 2x2 contrast visualization for each sample
                real_pred = contrasts_pred[i, 0, 0].cpu()  # (100,100)
                real_gt = contrasts_gt[i, 0, 0].cpu()      # (100,100)
                real_err = torch.abs(real_pred - real_gt)
                real_eps_pred = per_bkg_recons[i,0,0].cpu()
                real_eps_gt = y[i,0].cpu()
                real_eps_err = torch.abs(real_eps_pred - real_eps_gt)

                fig_contrast, axes_contrast = plt.subplots(2, 2, figsize=(8, 8))
                def show_contrast(ax, tensor2d, title, cmap='viridis'):
                    im = ax.imshow(tensor2d, cmap=cmap)
                    ax.set_title(title, fontsize=28)
                    ax.axis('off')  # Remove axis ticks and lines
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                show_contrast(axes_contrast[0, 0], real_pred, 'Pred Contrast')
                show_contrast(axes_contrast[0, 1], real_eps_pred, 'Pred Target')
                show_contrast(axes_contrast[1, 0], real_eps_gt, 'GT Target')
                show_contrast(axes_contrast[1, 1], real_eps_err, 'Abs Error')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx_global}_contrasts_real.pdf"), dpi=150)
                plt.close(fig_contrast)
                
                # Updated: 2x4 layout (remove first background visualization)
                fig, axes = plt.subplots(2, 4, figsize=(14, 6))
                def show(ax, tensor2d, title, cmap='viridis'):
                    im = ax.imshow(tensor2d, cmap=cmap)
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Row 0 (Real): GT | Mean | Std | Abs Err
                show(axes[0,0], y[i,0].cpu(), 'GT (Real)')
                show(axes[0,1], mean_recon[i,0].cpu(), 'Mean Recon (Real)')
                show(axes[0,2], std_recon[i,0].cpu(), 'Std (Real)', cmap='plasma')
                show(axes[0,3], torch.abs(mean_recon[i,0]-y[i,0]).cpu(), 'Abs Err (Real)', cmap='magma')
                # Row 1 (Imag): GT | Mean | Std | Abs Err
                show(axes[1,0], y[i,1].cpu(), 'GT (Imag)')
                show(axes[1,1], mean_recon[i,1].cpu(), 'Mean Recon (Imag)')
                show(axes[1,2], std_recon[i,1].cpu(), 'Std (Imag)', cmap='plasma')
                show(axes[1,3], torch.abs(mean_recon[i,1]-y[i,1]).cpu(), 'Abs Err (Imag)', cmap='magma')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx_global}.pdf"), dpi=150)
                plt.close(fig)
            collected += take
            if collected >= num_cases:
                break
    
    # Compute correlation coefficients
    all_mean_recons = np.concatenate(all_mean_recons, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_std_recons = np.concatenate(all_std_recons, axis=0)
    
    print("\n" + "="*60)
    print("UNCERTAINTY CORRELATION ANALYSIS")
    print("="*60)
    
    
    # Compute correlation between prediction error and uncertainty
    correlation, p_value = error_std_correlation(all_mean_recons, all_std_recons, all_targets)
    print(f"Uncertainty-Error Correlation: {correlation:.4f} (p={p_value:.4e})")
    
    print("="*60)
    print(f"Interpretation: Values close to +1 indicate good uncertainty estimates.")
    print(f"High correlation means high uncertainty corresponds to high error.")
    print("="*60 + "\n")
    
    print(f"Saved {collected} test reconstruction figures to {output_dir}")
    print(f"Saved {collected} test reconstruction figures to {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Train UNet to map synth fields (24x24x2) -> grids (100x100x2)")
    parser.add_argument("--data", type=str, default="all_data.mat", help="Path to .mat file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--debug-recon", action="store_true", help="Run a single debug reconstruction before training")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run test on a checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint for testing. If None, finds latest.")
    parser.add_argument("--experiment-tag", type=str, default="mbg_train_synth_test_cal_exp_25bkgs", help="Tag for experiment (used in logging)")
    parser.add_argument("--num-backgrounds", type=int, default=25, help="Number of backgrounds to use from training dataset")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Number of training steps per epoch. If None, uses full dataset. Data will be reused if this exceeds dataset size.")

    args = parser.parse_args()
    
    EXPERIMENT_TAG = args.experiment_tag
    print(f"Steps per epoch: {args.steps_per_epoch}")

    train_loader, val_loader, synth_test_loader, test_loader, sparam_test_loader = build_loaders(
        file_path=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        num_backgrounds=args.num_backgrounds,
        steps_per_epoch=args.steps_per_epoch,
    )

    #args.test_only = True
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
        bkg_sct_fields=bkg_sct_fields_from_ckpt,
        experiment_tag=EXPERIMENT_TAG
    )
    
    # print summary of the model
    

    print(f"Successfully loaded model with {model.bkgs.shape[0]} backgrounds")
    
    test(model=model, test_loader=synth_test_loader)
    # generate_calibration_curve(model=model, test_loader=test_loader, output_dir="figures/test")
    # litmus_test(model=model, in_range_test_loader=test_loader, out_of_range_test_loader=sparam_test_loader)

    print("--- Test Finished ---")


if __name__ == "__main__":
    main()

