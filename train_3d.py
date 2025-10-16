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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['bkgs', 'bkg_sct_fields'])
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
    # Validation plotting strategy: plot only on first val batch per epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2F, R, S) = (B, 2, 72, 72)
        # TransformerUNet expects (B, C, H*W) where C=2*num_freqs, H*W=R*S
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)  # (B, 2, 5184)
        return self.model(x_flat)  # (B, D, H, W) with D=H=W=image_dim

    def training_step(self, batch, batch_idx: int):
        x, y, fg_vol, bg_vol, fg_sct, bg_sct = batch
        y_hat = self(x).unsqueeze(1)  # (B,1,D,H,W)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))

                slice_data = [
                    ("Axial (Z)",    gt[dz],         pred[dz],         sd[dz],         absdiff[dz]),
                    ("Coronal (Y)",  gt[:, hy, :],    pred[:, hy, :],   sd[:, hy, :],    absdiff[:, hy, :]),
                    ("Sagittal (X)", gt[:, :, wx],    pred[:, :, wx],   sd[:, :, wx],    absdiff[:, :, wx]),
                ]

                for r, (title, gt2d, pr2d, sd2d, ad2d) in enumerate(slice_data):
                    # Shared color limits for GT and prediction per slice
                    vmin = min(gt2d.min(), pr2d.min())
                    vmax = max(gt2d.max(), pr2d.max())

                    im0 = axes[r, 0].imshow(gt2d, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[r, 0].set_title(f"{title} - GT")
                    axes[r, 0].axis('off')
                    fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

                    im1 = axes[r, 1].imshow(pr2d, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[r, 1].set_title(f"{title} - Pred mean")
                    axes[r, 1].axis('off')
                    fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

                    im2 = axes[r, 2].imshow(sd2d, cmap='plasma')
                    axes[r, 2].set_title(f"{title} - Std dev")
                    axes[r, 2].axis('off')
                    fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

                    im3 = axes[r, 3].imshow(ad2d, cmap='magma')
                    axes[r, 3].set_title(f"{title} - Abs diff")
                    axes[r, 3].axis('off')
                    fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)

                plt.tight_layout()
                fig_dir = f"figures/epoch_{self.current_epoch}"
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, f"val_step_{batch_idx}_slices.pdf")
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)
                
                # --- Render 3D volumes using vedo ---
                try:
                    # Render ground truth
                    render_volume(
                        volume=gt,
                        output_path=os.path.join(fig_dir, f"val_step_{batch_idx}_gt_render.png"),
                        camera_position=(10, 10, 100),
                        focal_point=(D//2, H//2, W//2),
                        image_size=(800, 800),
                        colormap="turbo",
                        alpha=[0, 0.1, 0.3, 0.6, 1.0],
                        show_axes=True,
                        background="white",
                        zoom=1.2
                    )
                    
                    # Render prediction
                    render_volume(
                        volume=pred,
                        output_path=os.path.join(fig_dir, f"val_step_{batch_idx}_pred_render.png"),
                        camera_position=(10, 10, 100),
                        focal_point=(D//2, H//2, W//2),
                        image_size=(800, 800),
                        colormap="turbo",
                        alpha=[0, 0.1, 0.3, 0.6, 1.0],
                        threshold=1.1,
                        show_axes=True,
                        background="white",
                        zoom=1.2
                    )
                    
                    # Render uncertainty (std)
                    render_volume(
                        volume=sd,
                        output_path=os.path.join(fig_dir, f"val_step_{batch_idx}_std_render.png"),
                        camera_position=(10, 10, 100),
                        focal_point=(D//2, H//2, W//2),
                        image_size=(800, 800),
                        colormap="plasma",
                        alpha=[0, 0.2, 0.4, 0.7, 1.0],
                        show_axes=True,
                        background="white",
                        zoom=1.2
                    )
                    
                    # Render absolute difference
                    render_volume(
                        volume=absdiff,
                        output_path=os.path.join(fig_dir, f"val_step_{batch_idx}_diff_render.png"),
                        camera_position=(10, 10, 100),
                        focal_point=(D//2, H//2, W//2),
                        image_size=(800, 800),
                        colormap="magma",
                        alpha=[0, 0.2, 0.4, 0.7, 1.0],
                        show_axes=True,
                        background="white",
                        zoom=1.2
                    )
                except Exception as e:
                    print(f"[WARN] Volume rendering failed: {e}")
                    # No counters; plotted only for batch_idx==0
            
        return val_loss

    def reconstruction(self, fg_sct: torch.Tensor, bg_sct: torch.Tensor, bg_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct foreground volume using all backgrounds:
          fg_sct: (B,2F,R,S)
          bg_sct: (B,nbgs,2F,R,S)
          bg_vol: (B,nbgs,1,D,H,W)
        Returns: (per_bkg, mean, std) with shapes
          per_bkg: (B,nbgs,1,D,H,W)
          mean/std: (B,1,D,H,W)
        """
        B, nbgs, _, R, S = bg_sct.shape
        C = fg_sct.shape[1]
        D = bg_vol.shape[-3]

        fg_exp = fg_sct.unsqueeze(1).expand(-1, nbgs, -1, -1, -1)
        contrast_in = fg_exp - bg_sct  # (B,nbgs,2F,R,S)
        contrast_in = contrast_in.reshape(B * nbgs, C, R, S)
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
):

    # Get training arrays (trainset=True normalizes with train stats and saves mean/std)
    train_pkg = prep.process_multifreq_data(
        fields_file=fields_file,
        targets_file=targets_file,
        targetless_fields_file=targetless_fields_file,
        num_test=0,
        trainset=True,
    )

    # MultiBkg pairing dataset
    ds_full = Fields3DMultiBkgDataset(
        fields_np=train_pkg.fields,
        targets_np=train_pkg.targets,
        n_backgrounds=n_backgrounds,
        dtype=torch.float32,
        seed=seed,
    )

    # Train/Val split on the paired dataset
    n_total = len(ds_full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Summary
    x0, y0, *_ = ds_full[0]
    print("3D MultiBkg Data summary:")
    print(f"  Input x shape: {tuple(x0.shape)}  # (2F, R, S)")
    print(f"  Target y shape: {tuple(y0.shape)}  # (1, D, H, W)")
    return train_loader, val_loader


def save_orthogonal_slices(target_vol: np.ndarray, pred_vol: np.ndarray, std_vol: np.ndarray, output_path: str):
    """
    Saves a 3x4 grid of orthogonal slices (axial, coronal, sagittal) for
    target, prediction, absolute difference, and reconstruction std dev.
    """
    import matplotlib.pyplot as plt

    # Ensure volumes are 3D
    target_vol = np.squeeze(target_vol)
    pred_vol = np.squeeze(pred_vol)
    std_vol = np.squeeze(std_vol)
    diff_vol = np.abs(target_vol - pred_vol)

    # Get center slice indices
    d, h, w = target_vol.shape
    d_slice, h_slice, w_slice = d // 2, h // 2, w // 2

    # Create a figure
    fig, axs = plt.subplots(3, 4, figsize=(16, 12), facecolor='w')
    fig.suptitle('Orthogonal Slices of 3D Validation Sample', fontsize=15)

    # Define what to plot
    slice_data = {
        "Axial (Z)": (target_vol[d_slice, :, :], pred_vol[d_slice, :, :], diff_vol[d_slice, :, :], std_vol[d_slice, :, :]),
        "Coronal (Y)": (target_vol[:, h_slice, :], pred_vol[:, h_slice, :], diff_vol[:, h_slice, :], std_vol[:, h_slice, :]),
        "Sagittal (X)": (target_vol[:, :, w_slice], pred_vol[:, :, w_slice], diff_vol[:, :, w_slice], std_vol[:, :, w_slice]),
    }

    # Plot the data
    for i, (title, (t_slice, p_slice, d_slice, s_slice)) in enumerate(slice_data.items()):
        # Determine shared color range for target and prediction
        vmin = min(t_slice.min(), p_slice.min())
        vmax = max(t_slice.max(), p_slice.max())

        axs[i, 0].imshow(t_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"{title} - Target")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(p_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"{title} - Prediction")
        axs[i, 1].axis('off')

        im = axs[i, 2].imshow(d_slice, cmap='magma')
        axs[i, 2].set_title(f"{title} - Abs Difference")
        axs[i, 2].axis('off')
        fig.colorbar(im, ax=axs[i, 2], fraction=0.046, pad=0.04)

        im_s = axs[i, 3].imshow(s_slice, cmap='plasma')
        axs[i, 3].set_title(f"{title} - Std Dev")
        axs[i, 3].axis('off')
        fig.colorbar(im_s, ax=axs[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n--- Orthogonal slice plot saved to: {output_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Train minimal 3D TransformerUNet with Lightning")
    parser.add_argument("--fields-file", type=str, default="./3d_dataset/field_data_with2000Target_72Rx.mat", help="Path to fields .h5 file")
    parser.add_argument("--targets-file", type=str, default="./3d_dataset/targets_data_with2000Target.mat", help="Path to targets .h5 file")
    parser.add_argument("--targetless-fields-file", type=str, default="./3d_dataset/targetless_fielddata_72Rx.mat", help="Path to targetless fields .h5 file")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-transformer-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--n-backgrounds", type=int, default=25)
    parser.add_argument("--test-only", action="store_true", help="Skip training and load the latest/best checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint. If None, finds latest.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from. Restores model, epoch, step, LR schedulers, etc.")

    args = parser.parse_args()

    # Keep tag aligned with logger name to find checkpoints
    EXPERIMENT_TAG = "mbg_3d_multibkg"

    train_loader, val_loader = build_3d_loaders(
        fields_file=args.fields_file,
        targets_file=args.targets_file,
        targetless_fields_file=args.targetless_fields_file,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        n_backgrounds=args.n_backgrounds,
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
        )

    # --- Save orthogonal slices of one validation example to a PNG file ---
    print("\n--- Generating slice plot for visualization ---")
    try:
        # Get a sample from the validation loader
        val_batch = next(iter(val_loader))
        xb, yb, fg_vol_b, bg_vol_b, fg_sct_b, bg_sct_b = val_batch

        # Ensure model is on the correct device and in eval mode
        model.eval()
        device = model.device if hasattr(model, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.to(device)

        # Run prediction
        with torch.no_grad():
            xb = xb.to(device)
            yp = model(xb)  # (B, D, H, W)
            _ = yp[0].detach().cpu().numpy()  # contrast prediction (unused here)
            # reconstruct foreground volume using stored backgrounds
            # Build tensors for single example reconstruction using all backgrounds
            fg_sct = fg_sct_b[0:1].to(device)  # (1,2F,R,S)
            bg_sct = model.bkg_sct_fields.unsqueeze(0).to(device)  # (1,nb,2F,R,S)
            bg_vol = model.bkgs.unsqueeze(0).to(device)  # (1,nb,1,D,H,W)
            _, mean_recon, std_recon = model.reconstruction(fg_sct, bg_sct, bg_vol)
            y_pred = mean_recon[0, 0].detach().cpu().numpy()
            y_std = std_recon[0, 0].detach().cpu().numpy()
            y_true = fg_vol_b[0, 0].detach().cpu().numpy()

        # Define output directory and save plot
        out_dir = os.path.join("figures", "remote_view")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "validation_slices.pdf")

        save_orthogonal_slices(y_true, y_pred, y_std, output_path)
        
        # --- Render 3D volumes using vedo ---
        print("\n--- Rendering 3D volumes ---")
        try:
            # Render ground truth
            render_volume(
                volume=y_true,
                output_path=os.path.join(out_dir, "validation_gt_render.png"),
                camera_position=(10, 10, 100),
                focal_point=(np.array(y_true.shape) / 2.0),
                image_size=(800, 800),
                colormap="turbo",
                alpha=[0, 0.1, 0.3, 0.6, 1.0],
                show_axes=True,
                background="white",
                zoom=1.2
            )
            
            # Render prediction
            render_volume(
                volume=y_pred,
                output_path=os.path.join(out_dir, "validation_pred_render.png"),
                camera_position=(10, 10, 100),
                focal_point=(np.array(y_pred.shape) / 2.0),
                image_size=(800, 800),
                colormap="turbo",
                alpha=[0, 0.1, 0.3, 0.6, 1.0],
                show_axes=True,
                background="white",
                zoom=1.2
            )
            
            # Render uncertainty (std)
            render_volume(
                volume=y_std,
                output_path=os.path.join(out_dir, "validation_std_render.png"),
                camera_position=(10, 10, 100),
                focal_point=(np.array(y_std.shape) / 2.0),
                image_size=(800, 800),
                colormap="plasma",
                alpha=[0, 0.2, 0.4, 0.7, 1.0],
                show_axes=True,
                background="white",
                zoom=1.2
            )
            
            # Render absolute difference
            y_diff = np.abs(y_true - y_pred)
            render_volume(
                volume=y_diff,
                output_path=os.path.join(out_dir, "validation_diff_render.png"),
                camera_position=(10, 10, 100),
                focal_point=(np.array(y_diff.shape) / 2.0),
                image_size=(800, 800),
                colormap="magma",
                alpha=[0, 0.2, 0.4, 0.7, 1.0],
                show_axes=True,
                background="white",
                zoom=1.2
            )
            print("Volume rendering completed successfully!")
        except Exception as e:
            print(f"[WARN] Volume rendering failed: {e}")

    except StopIteration:
        print("\n[ERROR] Plotting failed: The validation dataloader is empty.")
    except Exception as e:
        print(f"\n[ERROR] Plotting failed. An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
