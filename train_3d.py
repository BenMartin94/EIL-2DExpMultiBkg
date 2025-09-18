import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import prepare_data_3d as prep

# --- New imports for Lightning training and model ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import importlib.util

from Models_3d import TransformerUNet


class Fields3DDataset(Dataset):
    """
    PyTorch wrapper around 3D inversion data produced by prepare_data.process_multifreq_data.

    Inputs (fields): numpy array shaped (N, num_sources, num_receivers, 2*num_freqs)
      - We return as channels-first 2D tensors suitable for 2D processing of the measurement grid:
        x -> (2*num_freqs, num_receivers, num_sources)

    Targets (volumes): numpy array shaped (N, D, H, W)
      - We return as channels-first 3D tensors:
        y -> (1, D, H, W)
    """

    def __init__(self, fields_np: np.ndarray, targets_np: np.ndarray, dtype: torch.dtype = torch.float32):
        assert fields_np.ndim == 4, f"fields must be (N, S, R, 2F), got {fields_np.shape}"
        assert targets_np.ndim == 4, f"targets must be (N, D, H, W), got {targets_np.shape}"
        assert fields_np.shape[0] == targets_np.shape[0], "Mismatched batch sizes"
        self.fields = fields_np.astype(np.float32, copy=False)
        self.targets = targets_np.astype(np.float32, copy=False)
        self.dtype = dtype

        self.N, self.S, self.R, self.C = self.fields.shape  # C = 2*num_freqs
        self.D, self.H, self.W = self.targets.shape[1:]

    def __len__(self) -> int:
        return self.fields.shape[0]

    def __getitem__(self, idx: int):
        x = self.fields[idx]  # (S, R, 2F)
        y = self.targets[idx]  # (D, H, W)

        # fields channels-last (S, R, 2F) -> channels-first (2F, R, S)
        x_cf = np.ascontiguousarray(np.transpose(x, (2, 1, 0)))
        x_t = torch.from_numpy(x_cf).to(self.dtype)

        # add channel dim for target -> (1, D, H, W)
        y_t = torch.from_numpy(y[None, ...]).to(self.dtype)

        return x_t, y_t


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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerUNet(
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            num_receivers=num_receivers,
            num_sources=num_sources,
            num_freqs=num_freqs,
            image_dim=image_dim,
        )
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2F, R, S) -> (B, 2F, R*S)
        B, C, R, S = x.shape
        x_seq = x.view(B, C, R * S)
        return self.model(x_seq)  # (B, D, H, W) with D=H=W=image_dim

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x: (B,2F,R,S), y: (B,1,D,H,W)
        y_hat = self(x).unsqueeze(1)  # -> (B,1,D,H,W) to match target
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x).unsqueeze(1)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

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
):

    # Get training arrays (trainset=True normalizes with train stats and saves mean/std)
    train_pkg = prep.process_multifreq_data(
        fields_file=fields_file,
        targets_file=targets_file,
        targetless_fields_file=targetless_fields_file,
        num_test=0,
        trainset=True,
    )

    # Wrap as PyTorch datasets
    train_ds_full = Fields3DDataset(train_pkg.fields, train_pkg.targets)

    # Train/Val split
    n_total = len(train_ds_full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=g)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Print a quick summary
    print("3D Data summary:")
    print(f"  Fields shape (train[0] x): {train_ds_full[0][0].shape}  # (2F, R, S)")
    print(f"  Target shape (train[0] y): {train_ds_full[0][1].shape}  # (1, D, H, W)")
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train minimal 3D TransformerUNet with Lightning")
    parser.add_argument("--fields-file", type=str, default="/home/ben/School/EIL/2DMultiBkgExperimental/3d_dataset/field_data_with2000Target.mat", help="Path to fields .h5 file")
    parser.add_argument("--targets-file", type=str, default="/home/ben/School/EIL/2DMultiBkgExperimental/3d_dataset/targets_data_with2000Target.mat", help="Path to targets .h5 file")
    parser.add_argument("--targetless-fields-file", type=str, default="/home/ben/School/EIL/2DMultiBkgExperimental/3d_dataset/background_field_data.mat", help="Path to targetless fields .h5 file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-transformer-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    # Optionally control visible GPUs (match your 2D script style)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    train_loader, val_loader = build_3d_loaders(
        fields_file=args.fields_file,
        targets_file=args.targets_file,
        targetless_fields_file=args.targetless_fields_file,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Infer dimensions from the underlying dataset
    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
    num_receivers = base_ds.R
    num_sources = base_ds.S
    num_freqs = base_ds.C // 2  # C = 2*num_freqs
    image_dim = base_ds.D  # assuming cubic targets (D=H=W)

    model = Lit3D(
        num_receivers=num_receivers,
        num_sources=num_sources,
        num_freqs=num_freqs,
        image_dim=image_dim,
        lr=args.lr,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="mbg_3d_minimal")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
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

    print("--- Starting 3D Training ---")
    trainer.fit(model, train_loader, val_loader)
    print("--- 3D Training Finished ---")


if __name__ == "__main__":
    main()
