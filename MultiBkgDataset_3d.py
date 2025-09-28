import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import random


class MultiBkgDataset3D(Dataset):
	"""
	Multi-background dataset for 3D complex-valued volumes.

	Mirrors the 2D MultiBkgDataset API, but expects arrays with shape (N, D, H, W, 2)
	where the last dimension stores [real, imag].

	It reserves n background pairs (grid, field) and forms contrast pairs by
	combining every foreground with every background:

	  - Input (x): foreground_scattered_field - background_scattered_field  -> (2, D, H, W)
	  - Target (y): (foreground_grid - background_grid) / background_grid   -> (2, D, H, W)

	All tensors are returned channels-first.
	"""

	def __init__(
		self,
		grids: np.ndarray,
		fields: np.ndarray,
		n_backgrounds: int,
		dtype: torch.dtype = torch.float32,
		seed: Optional[int] = None,
	):
		"""
		Args:
			grids: Array of shape (N, D, H, W, 2) containing complex grid data (real/imag)
			fields: Array of shape (N, D, H, W, 2) containing complex field data (real/imag)
			n_backgrounds: Number of pairs to hold out as backgrounds
			dtype: PyTorch dtype for output tensors
			seed: Optional seed for reproducible background selection
		"""
		assert grids.ndim == 5 and fields.ndim == 5, "Expected (N,D,H,W,2) arrays"
		assert grids.shape[0] == fields.shape[0], "Mismatched batch sizes"
		assert grids.shape[-1] == 2 and fields.shape[-1] == 2, "Last dim must be 2 (real, imag)"
		assert n_backgrounds > 0, "Must have at least 1 background"
		assert n_backgrounds < grids.shape[0], "Number of backgrounds must be less than total samples"

		self.dtype = dtype

		if seed is not None:
			np.random.seed(seed)
			random.seed(seed)

		total = grids.shape[0]
		background_indices = np.random.choice(total, n_backgrounds, replace=False)
		foreground_indices = np.setdiff1d(np.arange(total), background_indices)

		self.background_grids = grids[background_indices]
		self.background_fields = fields[background_indices]

		self.foreground_grids = grids[foreground_indices]
		self.foreground_fields = fields[foreground_indices]

		self.n_backgrounds = n_backgrounds
		self.n_foregrounds = len(foreground_indices)

		print("MultiBkgDataset3D initialized:")
		print(f"  - {self.n_backgrounds} background pairs reserved")
		print(f"  - {self.n_foregrounds} foreground pairs available")
		print(f"  - Total dataset size: {len(self)} (each foreground paired with each background)")

	def __len__(self) -> int:
		return self.n_foregrounds * self.n_backgrounds

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Returns a tuple:
		  (input_tensor, target_tensor, fg_grid, bg_grid, fg_field, bg_field)

		Shapes:
		  - input_tensor: (2, D, H, W)
		  - target_tensor: (2, D, H, W)
		  - fg_grid/bg_grid/fg_field/bg_field: (2, D, H, W)
		"""
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")

		fg_idx = idx // self.n_backgrounds
		bg_idx = idx % self.n_backgrounds

		fg_grid = self.foreground_grids[fg_idx]      # (D,H,W,2)
		fg_field = self.foreground_fields[fg_idx]    # (D,H,W,2)
		bg_grid = self.background_grids[bg_idx]      # (D,H,W,2)
		bg_field = self.background_fields[bg_idx]    # (D,H,W,2)

		# Complex arithmetic on volume grid
		fg_complex = fg_grid[..., 0] + 1j * fg_grid[..., 1]
		bg_complex = bg_grid[..., 0] + 1j * bg_grid[..., 1]

		contrast_complex = (fg_complex - bg_complex) / bg_complex
		contrast = np.stack([contrast_complex.real, contrast_complex.imag], axis=-1)  # (D,H,W,2)

		sct_field = fg_field - bg_field  # (D,H,W,2)

		# Convert to channels-first tensors: (2,D,H,W)
		def to_cf(x: np.ndarray) -> torch.Tensor:
			return torch.from_numpy(np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2)))).to(self.dtype)

		input_tensor = to_cf(sct_field)
		target_tensor = to_cf(contrast)

		fg_grid_t = to_cf(fg_grid)
		bg_grid_t = to_cf(bg_grid)
		fg_field_t = to_cf(fg_field)
		bg_field_t = to_cf(bg_field)

		return input_tensor, target_tensor, fg_grid_t, bg_grid_t, fg_field_t, bg_field_t

	def get_background_info(self) -> dict:
		return {
			"n_backgrounds": self.n_backgrounds,
			"background_grid_shape": self.background_grids.shape,
			"background_field_shape": self.background_fields.shape,
		}

	def get_original_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return the original (foreground-only) field and grid as tensors:
		  (field_tensor, grid_tensor) with shape (2, D, H, W)
		"""
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")

		fg_idx = idx // self.n_backgrounds
		grid = self.foreground_grids[fg_idx]
		field = self.foreground_fields[fg_idx]

		grid_t = torch.from_numpy(np.ascontiguousarray(np.transpose(grid, (3, 0, 1, 2)))).to(self.dtype)
		field_t = torch.from_numpy(np.ascontiguousarray(np.transpose(field, (3, 0, 1, 2)))).to(self.dtype)
		return field_t, grid_t

	def get_background_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return the background (field, grid) tensors for the dataset index.
		Each is shaped (2, D, H, W)
		"""
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")

		bg_idx = idx % self.n_backgrounds
		bg_grid = self.background_grids[bg_idx]
		bg_field = self.background_fields[bg_idx]

		bg_grid_t = torch.from_numpy(np.ascontiguousarray(np.transpose(bg_grid, (3, 0, 1, 2)))).to(self.dtype)
		bg_field_t = torch.from_numpy(np.ascontiguousarray(np.transpose(bg_field, (3, 0, 1, 2)))).to(self.dtype)
		return bg_field_t, bg_grid_t

	def get_indices_info(self, idx: int) -> dict:
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")
		fg_idx = idx // self.n_backgrounds
		bg_idx = idx % self.n_backgrounds
		return {
			"dataset_idx": idx,
			"fg_idx": fg_idx,
			"bg_idx": bg_idx,
			"total_foregrounds": self.n_foregrounds,
			"total_backgrounds": self.n_backgrounds,
		}

	def get_backgrounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return all reserved backgrounds as tensors with shape:
		  - bg_grids: (n_backgrounds, 2, D, H, W)
		  - bg_fields: (n_backgrounds, 2, D, H, W)
		"""
		bg_grids_t = torch.from_numpy(
			np.ascontiguousarray(np.transpose(self.background_grids, (0, 4, 1, 2, 3)))
		).to(self.dtype)
		bg_fields_t = torch.from_numpy(
			np.ascontiguousarray(np.transpose(self.background_fields, (0, 4, 1, 2, 3)))
		).to(self.dtype)
		return bg_grids_t, bg_fields_t


# Backwards-friendly alias
MultiBkgDataset_3d = MultiBkgDataset3D

class Fields3DMultiBkgDataset(Dataset):
	"""
	3D multi-background dataset tailored for the measurement-to-volume pipeline.

	- fields_np: (N, S, R, 2F) channels-last measurement data
	- targets_np: (N, D, H, W) real-valued volumes

	Returns per pair (foreground x background):
	  x: (2F, R, S) = fg_sct - bg_sct
	  y: (1, D, H, W) = (fg_vol - bg_vol)/(bg_vol+eps)
	  extras: (fg_vol, bg_vol, fg_sct_cf, bg_sct_cf)
	"""

	def __init__(self, fields_np: np.ndarray, targets_np: np.ndarray, n_backgrounds: int, dtype: torch.dtype = torch.float32, seed: int | None = None):
		assert fields_np.ndim == 4, f"fields must be (N, S, R, 2F), got {fields_np.shape}"
		assert targets_np.ndim == 4, f"targets must be (N, D, H, W), got {targets_np.shape}"
		assert fields_np.shape[0] == targets_np.shape[0], "Mismatched batch sizes"
		assert 0 < n_backgrounds < fields_np.shape[0], "Invalid n_backgrounds"

		self.dtype = dtype
		self.fields = fields_np.astype(np.float32, copy=False)
		self.targets = targets_np.astype(np.float32, copy=False)

		if seed is not None:
			np.random.seed(seed)

		N = self.fields.shape[0]
		bkg_idx = np.random.choice(N, n_backgrounds, replace=False)
		fg_idx = np.setdiff1d(np.arange(N), bkg_idx)

		self.background_fields = self.fields[bkg_idx]
		self.background_vols = self.targets[bkg_idx]
		self.foreground_fields = self.fields[fg_idx]
		self.foreground_vols = self.targets[fg_idx]

		self.n_backgrounds = n_backgrounds
		self.n_foregrounds = len(fg_idx)

		# Shapes
		self.S = self.fields.shape[1]
		self.R = self.fields.shape[2]
		self.C = self.fields.shape[3]
		self.D, self.H, self.W = self.targets.shape[1:]

		print("Fields3DMultiBkgDataset initialized:")
		print(f"  - {self.n_backgrounds} background pairs reserved")
		print(f"  - {self.n_foregrounds} foreground pairs available")
		print(f"  - Total dataset size: {len(self)} (each foreground paired with each background)")

	def __len__(self) -> int:
		return self.n_foregrounds * self.n_backgrounds

	def __getitem__(self, idx: int):
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")
		fg_i = idx // self.n_backgrounds
		bg_i = idx % self.n_backgrounds

		fg_sct = self.foreground_fields[fg_i]  # (S,R,2F)
		bg_sct = self.background_fields[bg_i]  # (S,R,2F)
		fg_vol = self.foreground_vols[fg_i]    # (D,H,W)
		bg_vol = self.background_vols[bg_i]    # (D,H,W)

		x = fg_sct - bg_sct
		x_cf = np.ascontiguousarray(np.transpose(x, (2, 1, 0)))  # (2F,R,S)
		x_t = torch.from_numpy(x_cf).to(self.dtype)

		eps = 1e-8
		y = (fg_vol - bg_vol) / (bg_vol + eps)
		y_t = torch.from_numpy(y[None, ...]).to(self.dtype)  # (1,D,H,W)

		fg_vol_t = torch.from_numpy(fg_vol[None, ...]).to(self.dtype)
		bg_vol_t = torch.from_numpy(bg_vol[None, ...]).to(self.dtype)
		fg_sct_t = torch.from_numpy(x_cf + np.ascontiguousarray(np.transpose(bg_sct, (2, 1, 0)))).to(self.dtype)
		bg_sct_t = torch.from_numpy(np.ascontiguousarray(np.transpose(bg_sct, (2, 1, 0)))).to(self.dtype)

		return x_t, y_t, fg_vol_t, bg_vol_t, fg_sct_t, bg_sct_t

	def get_backgrounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
		bg_vols = torch.from_numpy(self.background_vols[:, None, ...]).to(self.dtype)  # (n_bgs,1,D,H,W)
		bg_sct_cf = np.ascontiguousarray(np.transpose(self.background_fields, (0, 3, 2, 1)))  # (n_bgs,2F,R,S)
		bg_fields = torch.from_numpy(bg_sct_cf).to(self.dtype)
		return bg_vols, bg_fields

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	print("Simple 3D test to verify contrast math and background consistency...")

	# Synthesize small 3D complex volumes
	N, D, H, W = 12, 24, 24, 24
	rng = np.random.default_rng(123)

	def make_complex_volumes(n, d, h, w):
		real = rng.normal(size=(n, d, h, w)).astype(np.float32)
		imag = rng.normal(size=(n, d, h, w)).astype(np.float32)
		return np.stack([real, imag], axis=-1)  # (N,D,H,W,2)

	grids = make_complex_volumes(N, D, H, W)
	fields = make_complex_volumes(N, D, H, W)

	n_bgs = 3
	ds = MultiBkgDataset3D(grids=grids, fields=fields, n_backgrounds=n_bgs, seed=42)

	# Train/val split
	from torch.utils.data import random_split
	total_len = len(ds)
	train_len = int(0.8 * total_len)
	val_len = total_len - train_len
	train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

	print(f"Dataset size: {len(ds)}, Train: {len(train_ds)}, Val: {len(val_ds)}")

	# Pick a random train sample for math verification
	train_idx = min(10, len(train_ds) - 1)
	orig_idx = train_ds.indices[train_idx]

	# Fetch tensors
	sct_field, contrast, fg_grid, bg_grid, fg_field, bg_field = ds[orig_idx]
	orig_field, orig_grid = ds.get_original_pair(orig_idx)
	bg_field_chk, bg_grid_chk = ds.get_background_pair(orig_idx)

	# Sanity shape checks
	assert sct_field.shape == (2, D, H, W)
	assert contrast.shape == (2, D, H, W)
	assert torch.allclose(bg_grid, bg_grid_chk)
	assert torch.allclose(bg_field, bg_field_chk)

	# Verify math: original_grid == bg_grid * (contrast + 1)
	contrast_complex = contrast[0].numpy() + 1j * contrast[1].numpy()    # (D,H,W)
	bg_complex = bg_grid[0].numpy() + 1j * bg_grid[1].numpy()
	orig_complex = orig_grid[0].numpy() + 1j * orig_grid[1].numpy()
	recon_complex = bg_complex * (contrast_complex + 1)

	err = np.abs(recon_complex - orig_complex)
	mean_err = err.mean()
	max_err = err.max()
	print(f"Reconstruction error (mean): {mean_err:.8f}")
	print(f"Reconstruction error (max): {max_err:.8f}")
	print(f"Math verification: {'✅ PASSED' if mean_err < 1e-5 else '❌ FAILED'}")

	# Background consistency tests
	print(f"\n{'='*50}")
	print("TESTING BACKGROUND CONSISTENCY ACROSS TRAIN/VAL SUBSETS")
	print(f"{'='*50}")

	bg_grids_all, bg_fields_all = ds.get_backgrounds()  # (n_bgs,2,D,H,W)

	def check_subset(name, subset):
		ok = True
		for i in range(min(5, len(subset))):
			oidx = subset.indices[i]
			bg_field_i, bg_grid_i = ds.get_background_pair(oidx)
			bg_idx = oidx % ds.n_backgrounds
			exp_bg_grid = bg_grids_all[bg_idx]
			exp_bg_field = bg_fields_all[bg_idx]
			grid_match = torch.allclose(bg_grid_i, exp_bg_grid, atol=1e-8)
			field_match = torch.allclose(bg_field_i, exp_bg_field, atol=1e-8)
			if grid_match and field_match:
				print(f"  ✅ {name} sample {i} (orig_idx {oidx}, bg_idx {bg_idx}): backgrounds match")
			else:
				ok = False
				print(f"  ❌ {name} sample {i} (orig_idx {oidx}, bg_idx {bg_idx}): grid_match={grid_match}, field_match={field_match}")
		return ok

	train_ok = check_subset("Train", train_ds)
	val_ok = check_subset("Val", val_ds)

	# Cross-check: same bg_idx across subsets are identical
	train_bg_idxs = set(oidx % ds.n_backgrounds for oidx in train_ds.indices)
	val_bg_idxs = set(oidx % ds.n_backgrounds for oidx in val_ds.indices)
	common = list(train_bg_idxs.intersection(val_bg_idxs))
	cross_ok = True
	if common:
		test_bg = common[0]
		# Find any sample in each subset using this bg
		t_oidx = next(oidx for oidx in train_ds.indices if (oidx % ds.n_backgrounds) == test_bg)
		v_oidx = next(oidx for oidx in val_ds.indices if (oidx % ds.n_backgrounds) == test_bg)
		t_field, t_grid = ds.get_background_pair(t_oidx)
		v_field, v_grid = ds.get_background_pair(v_oidx)
		grid_match = torch.allclose(t_grid, v_grid, atol=1e-8)
		field_match = torch.allclose(t_field, v_field, atol=1e-8)
		if grid_match and field_match:
			print(f"\n  ✅ Background {test_bg} identical between train (orig_idx {t_oidx}) and val (orig_idx {v_oidx})")
		else:
			cross_ok = False
			print(f"\n  ❌ Background {test_bg} differs between train and val!")
	else:
		print("\n  ⚠️  No common background indices between train and val sets")

	overall = train_ok and val_ok and cross_ok and (mean_err < 1e-5)
	if overall:
		print(f"\n✅ BACKGROUND CONSISTENCY TEST PASSED!")
	else:
		print(f"\n❌ BACKGROUND CONSISTENCY TEST FAILED!")

	# Optional: visualize center slices for quick inspection
	try:
		z = D // 2
		fig, axes = plt.subplots(2, 4, figsize=(14, 7))
		fig.suptitle("3D Contrast Math Verification (center slice)")

		im0 = axes[0, 0].imshow(orig_grid[0, z], cmap='viridis'); axes[0, 0].set_title('Original Grid (Real)'); plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
		im1 = axes[1, 0].imshow(orig_grid[1, z], cmap='viridis'); axes[1, 0].set_title('Original Grid (Imag)'); plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

		im2 = axes[0, 1].imshow(bg_grid[0, z], cmap='viridis'); axes[0, 1].set_title('Background Grid (Real)'); plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
		im3 = axes[1, 1].imshow(bg_grid[1, z], cmap='viridis'); axes[1, 1].set_title('Background Grid (Imag)'); plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)

		im4 = axes[0, 2].imshow(contrast[0, z], cmap='viridis'); axes[0, 2].set_title('Contrast (Real)'); plt.colorbar(im4, ax=axes[0, 2], shrink=0.8)
		im5 = axes[1, 2].imshow(contrast[1, z], cmap='viridis'); axes[1, 2].set_title('Contrast (Imag)'); plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)

		im6 = axes[0, 3].imshow(recon_complex.real[z], cmap='viridis'); axes[0, 3].set_title('Reconstructed (Real)'); plt.colorbar(im6, ax=axes[0, 3], shrink=0.8)
		im7 = axes[1, 3].imshow(recon_complex.imag[z], cmap='viridis'); axes[1, 3].set_title('Reconstructed (Imag)'); plt.colorbar(im7, ax=axes[1, 3], shrink=0.8)

		for ax in axes.ravel():
			ax.axis('off')
		plt.tight_layout()
		out_dir = 'figures/litmus_tests'
		import os
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, 'mbg3d_verification.png')
		plt.savefig(out_path, dpi=120)
		plt.close()
		print(f"\n--- Verification plot saved to: {out_path} ---")
	except Exception as e:
		print(f"\n[Plot skipped] {e}")

	print(f"\n{'='*50}")
	print("✅ Simple 3D test completed!")
	print(f"{'='*50}")

