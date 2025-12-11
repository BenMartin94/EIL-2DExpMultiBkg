import os
import argparse
import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import prepare_data_3d as prep

# --- New imports for Lightning training and model ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from MultiBkgDataset_3d import Fields3DMultiBkgDataset

from Models_3d import TransformerUNet
from rendering import render_volume
import matplotlib.pyplot as plt


# --- Minimal LightningModule for 3D training ---
class Lit3D(pl.LightningModule):
    def __init__(
        self,
        num_receivers: int,
        num_sources: int,
        num_freqs: int,
        image_dim: int,
        lr: float = 1e-3,
        num_transformer_layers: int = 1,
        num_heads: int = 16,
        dropout_rate: float = 0.0,
        bkgs: torch.Tensor | None = None,            # (n_bgs,1,D,H,W)
        bkg_sct_fields: torch.Tensor | None = None,  # (n_bgs,2F,R,S)
        mbg_dataset = None,                          # MultiBkgDataset for sanity checks
        train_loader = None,                         # Train data loader for sanity checks
        val_loader = None,                           # Val data loader for sanity checks
        test_loader = None,                          # Test data loader for evaluation
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['bkgs', 'bkg_sct_fields', 'mbg_dataset', 'train_loader', 'val_loader', 'test_loader'])
        self.model = TransformerUNet(
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            num_receivers=num_receivers,
            num_sources=num_sources,
            num_freqs=num_freqs,
            image_dim=image_dim,
        )
        self.criterion = torch.nn.MSELoss()
        # store backgrounds as buffers
        self.register_buffer('bkgs', bkgs)
        self.register_buffer('bkg_sct_fields', bkg_sct_fields)
        self.mbg_dataset = mbg_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    # Validation plotting strategy: plot only on first val batch per epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, S, R) = (B, 2, 72, 72) where 2=[real, imag], S=sources, R=receivers
        # TransformerUNet expects (B, C, H*W) where C=2, H*W=S*R=5184
        B, C, S, R = x.shape
        x_flat = x.view(B, C, S * R)  # (B, 2, 5184)
        return self.model(x_flat)  # (B, D, H, W) with D=H=W=image_dim

    def training_step(self, batch, batch_idx: int):
        x, y, fg_vol, bg_vol, fg_sct, bg_sct = batch
        y_hat = self(x).unsqueeze(1)  # (B,1,D,H,W)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Plot once per epoch on the first batch - matching test plot format
        if batch_idx == 0:
            import matplotlib.pyplot as plt
            fig_dir = f"figures/3d/epoch_{self.current_epoch}"
            os.makedirs(fig_dir, exist_ok=True)
            
            # Plot first example in batch
            i = 0
            pred_contrast = y_hat[i, 0].detach().cpu().numpy()  # (D, H, W)
            bg = bg_vol[i, 0].detach().cpu().numpy()  # (D, H, W)
            true_contrast = y[i, 0].detach().cpu().numpy()  # (D, H, W)
            
            D, H, W = pred_contrast.shape
            dz = D // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            vmin = min(pred_contrast[:, :, dz].min(), bg[:, :, dz].min(), true_contrast[:, :, dz].min())
            vmax = max(pred_contrast[:, :, dz].max(), bg[:, :, dz].max(), true_contrast[:, :, dz].max())
            
            im0 = axes[0].imshow(bg[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title('Background Volume Slice', fontsize=14)
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(pred_contrast[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title('Predicted Contrast Slice', fontsize=14)
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(true_contrast[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[2].set_title('True Contrast Slice', fontsize=14)
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            fig_path = os.path.join(fig_dir, "train_contrast_example.pdf")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return loss

    # Plotting controlled by batch_idx == 0; no counters needed

    def validation_step(self, batch, batch_idx: int):
        x, y, fg_vol, bg_vol, fg_sct, bg_sct = batch
        y_hat = self(x).unsqueeze(1)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat - y))
            mse = torch.mean((y_hat - y) ** 2)
            self.log("val_mae", mae, on_step=False, on_epoch=True)
            self.log("val_mse", mse, on_step=False, on_epoch=True)

            # Reconstruct using all stored backgrounds and plot diagnostics, limited per epoch
            
            import matplotlib.pyplot as plt
            B = x.size(0)
            if (self.bkgs is not None and self.bkg_sct_fields is not None and self.bkgs.ndim == 5 and batch_idx == 0):
                nb = self.bkgs.shape[0]
                # Expand backgrounds to (B, nb, ...)
                bg_vol_all = self.bkgs.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
                bg_sct_all = self.bkg_sct_fields.unsqueeze(0).expand(B, -1, -1, -1, -1)
                _, mean_recon, std_recon = self.reconstruction(fg_sct, bg_sct_all, bg_vol_all)

                # Plot a single example with all three orthogonal slices
                i = 0
                gt = fg_vol[i, 0].detach().cpu().numpy()
                pred = mean_recon[i, 0].detach().cpu().numpy()
                sd = std_recon[i, 0].detach().cpu().numpy()
                absdiff = np.abs(pred - gt)

                # Center slice indices (assume cubic)
                D = pred.shape[0]
                H = pred.shape[1]
                W = pred.shape[2]
                dz, hy, wx = D // 2, H // 2, W // 2

                # Build 3x4 grid: rows = Axial/Coronal/Sagittal, cols = GT, Pred mean, Std dev, Abs diff
                # Font size configuration
                TITLE_FONTSIZE = 14  # Change this to adjust column titles
                YLABEL_FONTSIZE = 14  # Change this to adjust slice labels
                
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))

                slice_data = [
                    ("Axial (Z)",    gt[dz],         pred[dz],         sd[dz],         absdiff[dz]),
                    ("Coronal (Y)",  gt[:, hy, :],    pred[:, hy, :],   sd[:, hy, :],    absdiff[:, hy, :]),
                    ("Sagittal (X)", gt[:, :, wx],    pred[:, :, wx],   sd[:, :, wx],    absdiff[:, :, wx]),
                ]
                
                # Column titles (only on top row)
                column_titles = ["Ground Truth", "Prediction", "Standard Deviation", "|Ground Truth - Prediction|"]

                for r, (slice_label, gt2d, pr2d, sd2d, ad2d) in enumerate(slice_data):
                    # Shared color limits for GT and prediction per slice
                    vmin = min(gt2d.min(), pr2d.min())
                    vmax = max(gt2d.max(), pr2d.max())

                    im0 = axes[r, 0].imshow(gt2d, cmap='viridis', vmin=vmin, vmax=vmax)
                    if r == 0:  # Only add title to top row
                        axes[r, 0].set_title(column_titles[0], fontsize=TITLE_FONTSIZE)
                    axes[r, 0].set_ylabel(slice_label, fontsize=YLABEL_FONTSIZE, rotation=90, labelpad=10)
                    axes[r, 0].set_xticks([])
                    axes[r, 0].set_yticks([])
                    fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

                    im1 = axes[r, 1].imshow(pr2d, cmap='viridis', vmin=vmin, vmax=vmax)
                    if r == 0:  # Only add title to top row
                        axes[r, 1].set_title(column_titles[1], fontsize=TITLE_FONTSIZE)
                    axes[r, 1].axis('off')
                    fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

                    im2 = axes[r, 2].imshow(sd2d, cmap='plasma')
                    if r == 0:  # Only add title to top row
                        axes[r, 2].set_title(column_titles[2], fontsize=TITLE_FONTSIZE)
                    axes[r, 2].axis('off')
                    fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

                    im3 = axes[r, 3].imshow(ad2d, cmap='magma')
                    if r == 0:  # Only add title to top row
                        axes[r, 3].set_title(column_titles[3], fontsize=TITLE_FONTSIZE)
                    axes[r, 3].axis('off')
                    fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)

                plt.tight_layout()
                fig_dir = f"figures/3d/epoch_{self.current_epoch}"
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, f"val_step_{batch_idx}_slices.pdf")
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)
                
                # Evaluate on test set once per epoch (at batch_idx == 0)
                if self.test_loader is not None:
                    self._evaluate_test_set(fig_dir)
            
        return val_loss

    def _evaluate_test_set(self, fig_dir: str):
        """Evaluate on test set and create comparison plot"""
        test_losses = []
        first_batch_data = None
        
        for batch_idx, (x_test, y_test) in enumerate(self.test_loader):
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            B = x_test.shape[0]
            
            # Expand backgrounds for all B samples
            bg_sct = self.bkg_sct_fields.unsqueeze(0).expand(B, -1, -1, -1, -1)
            bg_vol = self.bkgs.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
            
            # Perform full reconstruction using all backgrounds
            per_bkg_recon, mean_recon, std_recon = self.reconstruction(x_test, bg_sct, bg_vol)

            # example contrast
            sct1 = x_test[0:1] - self.bkg_sct_fields[0:1]
            contrast1 = (y_test[0:1] - self.bkgs[0:1]) / self.bkgs[0:1]

            # dataset_field = self.mbg_dataset[0][0].to(self.device)
            # dataset_contrast = self.mbg_dataset[0][1].to(self.device)

            # assert torch.allclose(sct1, dataset_field, atol=1e-6), "Sanity check failed: scattered field does not match dataset"
            # assert torch.allclose(contrast1, dataset_contrast, atol=1e-6), "Sanity check failed: contrast volume does not match dataset"

            pred_contrast1 = self(sct1)
            pred_perm = pred_contrast1 * self.bkgs[0:1] + self.bkgs[0:1]

            # assert torch.allclose(pred_perm, per_bkg_recon[0:1, 0], atol=1e-6), "Sanity check failed: per-background reconstruction does not match model output"
            
            # Save first batch for plotting
            if batch_idx == 0:
                first_batch_data = (per_bkg_recon, mean_recon, y_test)
                # plot the pred contrast example along with bkg, and true contrast
                import matplotlib.pyplot as plt
                pc1 = pred_contrast1[0, :, :, :].detach().cpu().numpy()
                bc1 = self.bkgs[0, 0, :, :, :].detach().cpu().numpy()
                tc1 = contrast1[0, 0, :, :, :].detach().cpu().numpy()
                D, H, W = pc1.shape
                dz = D // 2
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                vmin = min(pc1[:, :, dz].min(), bc1[:, :, dz].min(), tc1[:, :, dz].min())
                vmax = max(pc1[:, :, dz].max(), bc1[:, :, dz].max(), tc1[:, :, dz].max())
                im0 = axes[0].imshow(bc1[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0].set_title('Background Volume Slice', fontsize=14)
                axes[0].axis('off')
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                im1 = axes[1].imshow(pc1[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title('Predicted Contrast Slice', fontsize=14)
                axes[1].axis('off')
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                im2 = axes[2].imshow(tc1[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title('True Contrast Slice', fontsize=14)
                axes[2].axis('off')
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig_path = os.path.join(fig_dir, "test_contrast_example.pdf")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            
            # Compute loss
            test_loss = torch.nn.functional.mse_loss(pred_perm, y_test[0:1])
            
            test_losses.append(test_loss.item())
        
        # Log test metrics
        avg_test_loss = np.mean(test_losses)
        self.log("test_loss", avg_test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Create comparison plot
        if first_batch_data is not None:
            per_bkg_recon, mean_recon, y_test = first_batch_data
            
            first_per_bkg = per_bkg_recon[0, 0, 0].detach().cpu().numpy()
            mean_pred = mean_recon[0, 0].detach().cpu().numpy()
            gt = y_test[0, 0].detach().cpu().numpy()
            
            D, H, W = gt.shape
            dz = D // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            vmin = min(first_per_bkg[:, :, dz].min(), mean_pred[:, :, dz].min(), gt[:, :, dz].min())
            vmax = max(first_per_bkg[:, :, dz].max(), mean_pred[:, :, dz].max(), gt[:, :, dz].max())
            
            im0 = axes[0].imshow(first_per_bkg[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title('First Per-Background Recon', fontsize=14)
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(mean_pred[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title('Mean Reconstruction', fontsize=14)
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(gt[:, :, dz], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[2].set_title('Ground Truth', fontsize=14)
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            fig_path = os.path.join(fig_dir, "test_comparison.pdf")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def reconstruction(self, fg_sct: torch.Tensor, bg_sct: torch.Tensor, bg_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct foreground volume using all backgrounds:
          fg_sct: (B, 2, S, R)
          bg_sct: (B, nbgs, 2, S, R)
          bg_vol: (B, nbgs, 1, D, H, W)
        Returns: (per_bkg, mean, std) with shapes
          per_bkg: (B, nbgs, 1, D, H, W)
          mean/std: (B, 1, D, H, W)
        """
        B, nbgs, _, S, R = bg_sct.shape
        C = fg_sct.shape[1]  # C = 2
        D = bg_vol.shape[-3]

        fg_exp = fg_sct.unsqueeze(1).expand(-1, nbgs, -1, -1, -1)  # (B, nbgs, 2, S, R)
        contrast_in = fg_exp - bg_sct  # (B, nbgs, 2, S, R)
        contrast_in = contrast_in.reshape(B * nbgs, C, S, R)  # (B*nbgs, 2, S, R)
        pred_contrast = self(contrast_in).unsqueeze(1)  # (B*nbgs,1,D,H,W)
        pred_contrast = pred_contrast.view(B, nbgs, 1, D, D, D)  # cubic

        fg_vol_pred = pred_contrast * bg_vol + bg_vol  # (B,nbgs,1,D,H,W)
        mean = fg_vol_pred.mean(dim=1)
        std = torch.std(fg_vol_pred, dim=1)
        return fg_vol_pred, mean, std

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_3d_loaders(
    fields_file: str,
    targets_file: str,
    targetless_fields_file: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    n_backgrounds: int,
    num_test: int = 10,
    max_samples: int = None,
):

    # Get training and test arrays in one call
    train_pkg, test_pkg = prep.process_multifreq_data(
        fields_file=fields_file,
        targets_file=targets_file,
        targetless_fields_file=targetless_fields_file,
        num_test=num_test,
        max_samples=max_samples,
    )

    # MultiBkg pairing dataset for training
    ds_full = Fields3DMultiBkgDataset(
        fields_np=train_pkg.fields,
        targets_np=train_pkg.targets,
        n_backgrounds=n_backgrounds,
        dtype=torch.float32,
        seed=seed,
    )

    
    from torch.utils.data import TensorDataset
    test_fields_tensor = torch.tensor(test_pkg.fields, dtype=torch.float32)  # (N, 2, S, R)
    test_targets_tensor = torch.tensor(test_pkg.targets, dtype=torch.float32).unsqueeze(1)  # (N, 1, D, H, W)
    test_ds = TensorDataset(test_fields_tensor, test_targets_tensor)

    # Train/Val split on the paired dataset
    n_total = len(ds_full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Summary
    x0, y0, *_ = ds_full[0]
    print("3D MultiBkg Data summary:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Input x shape: {tuple(x0.shape)}  # (2, S, R) where 2=[real,imag]")
    print(f"  Target y shape: {tuple(y0.shape)}  # (1, D, H, W)")
    return train_loader, val_loader, test_loader


def save_slices(target_vol: np.ndarray, pred_vol: np.ndarray, std_vol: np.ndarray, output_path: str):
    """
    Saves a 3x4 grid showing 3 slices along the last axis (W/X).
    Rows: 3 different slice positions along the last axis
    Columns: GT, Pred, Std dev, Abs diff
    """
    import matplotlib.pyplot as plt

    # Ensure volumes are 3D
    gt = np.squeeze(target_vol)
    pred = np.squeeze(pred_vol)
    sd = np.squeeze(std_vol)
    absdiff = np.abs(gt - pred)

    # Get dimensions
    d, h, w = gt.shape
    
    # Select 3 slice indices along the last axis (W): 1/4, 1/2, 3/4
    slice_indices = [w // 4, w // 2, 3 * w // 4]

    # Font size configuration
    TITLE_FONTSIZE = 20  # Change this to adjust column titles
    YLABEL_FONTSIZE = 20  # Change this to adjust slice labels
    SUPTITLE_FONTSIZE = 28  # Change this to adjust the main figure title
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='w')
    fig.suptitle('3D Reconstruction - Slices', fontsize=SUPTITLE_FONTSIZE)
    
    # Column titles (only on top row)
    column_titles = ["Ground Truth", "Prediction", "Standard Deviation", "|Ground Truth - Prediction|"]

    # Plot the data
    for i, w_idx in enumerate(slice_indices):
        # Extract slices along the last axis: [:, :, w_idx]
        gt_slice = gt[:, :, w_idx]
        pred_slice = pred[:, :, w_idx]
        sd_slice = sd[:, :, w_idx]
        absdiff_slice = absdiff[:, :, w_idx]
        
        # Determine shared color range for target and prediction
        vmin = min(gt_slice.min(), pred_slice.min())
        vmax = max(gt_slice.max(), pred_slice.max())

        im0 = axes[i, 0].imshow(gt_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        if i == 0:  # Only add title to top row
            axes[i, 0].set_title(column_titles[0], fontsize=TITLE_FONTSIZE)
        axes[i, 0].set_ylabel(f"Slice {w_idx}/{w}", fontsize=YLABEL_FONTSIZE, rotation=90, labelpad=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(pred_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        if i == 0:  # Only add title to top row
            axes[i, 1].set_title(column_titles[1], fontsize=TITLE_FONTSIZE)
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(sd_slice, cmap='plasma')
        if i == 0:  # Only add title to top row
            axes[i, 2].set_title(column_titles[2], fontsize=TITLE_FONTSIZE)
        axes[i, 2].axis('off')
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        im3 = axes[i, 3].imshow(absdiff_slice, cmap='magma')
        if i == 0:  # Only add title to top row
            axes[i, 3].set_title(column_titles[3], fontsize=TITLE_FONTSIZE)
        axes[i, 3].axis('off')
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n--- Slice plot saved to: {output_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Train minimal 3D TransformerUNet with Lightning")
    parser.add_argument("--fields-file", type=str, default="./3d_dataset/field_data_with2000Target_72Rx.mat", help="Path to fields .h5 file")
    parser.add_argument("--targets-file", type=str, default="./3d_dataset/targets_data_with2000Target.mat", help="Path to targets .h5 file")
    parser.add_argument("--targetless-fields-file", type=str, default="./3d_dataset/targetless_fielddata_72Rx.mat", help="Path to targetless fields .h5 file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-transformer-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--n-backgrounds", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=2000, help="Limit total dataset size before train/test split (useful for testing)")
    parser.add_argument("--test-only", action="store_true", help="Skip training and load the latest/best checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint. If None, finds latest.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from. Restores model, epoch, step, LR schedulers, etc.")

    args = parser.parse_args()

    # Keep tag aligned with logger name to find checkpoints
    EXPERIMENT_TAG = "mbg_3d_multibkg_10bkgs"

    train_loader, val_loader, test_loader = build_3d_loaders(
        fields_file=args.fields_file,
        targets_file=args.targets_file,
        targetless_fields_file=args.targetless_fields_file,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        n_backgrounds=args.n_backgrounds,
        max_samples=args.max_samples,
    )

    # Infer dimensions and backgrounds from the underlying dataset
    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
    num_receivers = base_ds.R
    num_sources = base_ds.S
    num_freqs = base_ds.C // 2  # C = 2*num_freqs
    image_dim = base_ds.D  # assuming cubic targets (D=H=W)
    bg_vols, bg_fields = base_ds.get_backgrounds()  # (n_bgs,1,D,H,W), (n_bgs,2F,R,S)

    def find_latest_ckpt(experiment_tag: str) -> str:
        pattern = f'lightning_logs/{experiment_tag}/version_*/checkpoints/*.ckpt'
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No checkpoints found under {pattern}")
        return max(files, key=os.path.getctime)

    logger = TensorBoardLogger(save_dir="lightning_logs", name=EXPERIMENT_TAG)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
    )

    ckpt_path = None
    if not args.test_only:
        # Train a fresh model
        model = Lit3D(
            num_receivers=num_receivers,
            num_sources=num_sources,
            num_freqs=num_freqs,
            image_dim=image_dim,
            lr=args.lr,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            dropout_rate=0.0,
            bkgs=bg_vols,
            bkg_sct_fields=bg_fields,
            mbg_dataset=base_ds,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            log_every_n_steps=10,
            logger=logger,
            fast_dev_run=args.fast_dev_run,
            deterministic=True,
            callbacks=[checkpoint_callback],
        )

        print("--- Starting 3D Training (MultiBkg) ---")
        if args.resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, train_loader, val_loader)
        print("--- 3D Training Finished ---")

        ckpt_path = checkpoint_callback.best_model_path
        if ckpt_path:
            print(f"Best checkpoint: {ckpt_path}")
            # Load best weights for evaluation/plotting
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            bkgs_from_ckpt = checkpoint['state_dict'].get('bkgs', None)
            bkg_fields_from_ckpt = checkpoint['state_dict'].get('bkg_sct_fields', None)
            try:
                model = Lit3D.load_from_checkpoint(
                    ckpt_path,
                    strict=False,
                    bkgs=bkgs_from_ckpt,
                    bkg_sct_fields=bkg_fields_from_ckpt,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
            except Exception:
                # Fall back to using the in-memory model if load fails
                print("[WARN] Failed to reload best checkpoint; using in-memory model.")
        else:
            print("[WARN] No best checkpoint path recorded.")
    else:
        # Test-only: load from provided path or find latest
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            print("No checkpoint path provided, finding the latest...")
            ckpt_path = find_latest_ckpt(EXPERIMENT_TAG)
        print(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        bkgs_from_ckpt = checkpoint['state_dict']['bkgs']
        bkg_fields_from_ckpt = checkpoint['state_dict']['bkg_sct_fields']
        model = Lit3D.load_from_checkpoint(
            ckpt_path,
            strict=False,
            bkgs=bkgs_from_ckpt,
            bkg_sct_fields=bkg_fields_from_ckpt,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

    # --- Evaluate on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    print(f"Model background shapes:")
    print(f"  bkg_sct_fields: {model.bkg_sct_fields.shape}")
    print(f"  bkgs: {model.bkgs.shape}")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_losses = []
    test_maes = []
    
    print("\n--- Generating slice plot for visualization ---")
    try:
        # Get a sample from the test loader
        test_batch = next(iter(test_loader))
        x_test_vis, y_test_vis = test_batch
        x_test_vis = x_test_vis.to(device)
        B_vis = x_test_vis.shape[0]

        # Run prediction with full reconstruction using backgrounds
        with torch.no_grad():
            bg_sct_vis = model.bkg_sct_fields.unsqueeze(0).expand(B_vis, -1, -1, -1, -1).to(device)
            bg_vol_vis = model.bkgs.unsqueeze(0).expand(B_vis, -1, -1, -1, -1, -1).to(device)
            _, mean_recon_vis, std_recon_vis = model.reconstruction(x_test_vis, bg_sct_vis, bg_vol_vis)
            
            y_pred = mean_recon_vis[0, 0].detach().cpu().numpy()
            y_std = std_recon_vis[0, 0].detach().cpu().numpy()
        
        y_true = y_test_vis[0, 0].detach().cpu().numpy()

        # Define output directory and save plot
        test_out_dir = os.path.join("figures", "test_set")
        os.makedirs(test_out_dir, exist_ok=True)
        test_output_path = os.path.join(test_out_dir, "test_sample_slices.pdf")

        save_slices(y_true, y_pred, y_std, test_output_path)
        print(f"Test slice plot saved to: {test_output_path}")
        
        # --- Render 3D test volumes using vedo ---
        print("\n--- Rendering 3D test volumes ---")
        try:
            # Render ground truth
            render_volume(
                volume=y_true,
                output_path=os.path.join(test_out_dir, "test_gt_render.png"),
                focal_point=(np.array(y_true.shape) / 2.0),
                image_size=(800, 800),
                colormap="turbo",
                alpha=[0, 0.1, 0.3, 0.6, 1.0],
                show_axes=True,
                background="white",
                vmin=1.0,
                vmax=3
            )
            
            # Render prediction
            render_volume(
                volume=y_pred,
                output_path=os.path.join(test_out_dir, "test_pred_render.png"),
                focal_point=(np.array(y_pred.shape) / 2.0),
                image_size=(800, 800),
                colormap="turbo",
                alpha=[0, 0.1, 0.3, 0.6, 1.0],
                show_axes=True,
                background="white",
                vmin=1.0,
                vmax=3
            )
            
            # Render uncertainty (std)
            render_volume(
                volume=y_std,
                output_path=os.path.join(test_out_dir, "test_std_render.png"),
                focal_point=(np.array(y_std.shape) / 2.0),
                image_size=(800, 800),
                colormap="plasma",
                alpha=[0, 0.2, 0.4, 0.7, 1.0],
                show_axes=True,
                background="white",
                vmin=0.2
            )
            
            # Render absolute difference
            y_diff = np.abs(y_true - y_pred)
            render_volume(
                volume=y_diff,  
                output_path=os.path.join(test_out_dir, "test_diff_render.png"),
                focal_point=(np.array(y_diff.shape) / 2.0),
                image_size=(800, 800),
                colormap="magma",
                alpha=[0, 0.2, 0.4, 0.7, 1.0],
                show_axes=True,
                background="white",
            )
            print("Test volume rendering completed successfully!")
        except Exception as e:
            print(f"[WARN] Volume rendering failed: {e}")

    except StopIteration:
        print("\n[ERROR] Plotting failed: The test dataloader is empty.")
    except Exception as e:
        print(f"\n[ERROR] Plotting failed. An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
