import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import random


def split_complex_to_real_imag(arr):
    """
    Converts an n-dimensional complex array to an (n+1)-dimensional array
    with the last dimension of size 2: [real, imag].
    """
    if not np.iscomplexobj(arr):
        raise ValueError("Input array must be of complex type.")
    return np.stack((arr.real, arr.imag), axis=-1)


def read_mat_file(file_path):
    """Reads a .mat file and returns the contained matrix."""
    mat_data = sio.loadmat(file_path)
    num_samples = mat_data['data'].shape[1]
    grid = mat_data['data'][0, 0][0]
    uncal_s_par = mat_data['data'][0, 0][1]
    cal_e_field = mat_data['data'][0, 0][2]
    synth_field = mat_data['data'][0, 0][3]
    grids = np.zeros((num_samples, *grid.shape), dtype=np.complex128)
    uncal_s_pars = np.zeros((num_samples, *uncal_s_par.shape), dtype=np.complex128)
    cal_e_fields = np.zeros((num_samples, *cal_e_field.shape), dtype=np.complex128)
    synth_fields = np.zeros((num_samples, *synth_field.shape), dtype=np.complex128)
    for i in range(num_samples):
        grids[i] = mat_data['data'][0, i][0]
        uncal_s_pars[i] = mat_data['data'][0, i][1]
        cal_e_fields[i] = mat_data['data'][0, i][2]
        synth_fields[i] = mat_data['data'][0, i][3]

    return grids, uncal_s_pars, cal_e_fields, synth_fields


class MultiBkgDataset(Dataset):
    """
    Multi-background dataset that reserves n pairs of grids and fields as backgrounds,
    then creates contrast pairs by augmenting the remaining data with these backgrounds.
    
    The dataset creates contrast pairs by combining foreground data with background data:
    - Input: foreground_grid + background_grid
    - Target: foreground_field + background_field
    
    This allows the model to learn to separate foreground from background contributions.
    """
    
    def __init__(
        self, 
        grids: np.ndarray, 
        fields: np.ndarray, 
        n_backgrounds: int,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None
    ):
        """
        Initialize the MultiBkgDataset.
        
        Args:
            grids: Array of shape (N, H, W, 2) containing grid data (real/imag)
            fields: Array of shape (N, H, W, 2) containing field data (real/imag)
            n_backgrounds: Number of pairs to reserve as backgrounds
            contrast_factor: Factor to control the strength of background addition
            dtype: PyTorch dtype for output tensors
            seed: Random seed for reproducible background selection
        """
        assert grids.ndim == 4 and fields.ndim == 4, "Expected (N,H,W,C) arrays"
        assert grids.shape[0] == fields.shape[0], "Mismatched batch sizes"
        assert grids.shape[-1] == 2 and fields.shape[-1] == 2, "Expected 2-channel real/imag in last dim"
        assert n_backgrounds > 0, "Must have at least 1 background"
        assert n_backgrounds < grids.shape[0], "Number of backgrounds must be less than total samples"
        
        self.dtype = dtype
        
        # Set random seed for reproducible background selection
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Randomly select background indices
        total_samples = grids.shape[0]
        background_indices = np.random.choice(total_samples, n_backgrounds, replace=False)
        foreground_indices = np.setdiff1d(np.arange(total_samples), background_indices)
        
        # Store backgrounds
        self.background_grids = grids[background_indices]
        self.background_fields = fields[background_indices]
        
        # Store foregrounds (the main data we'll augment)
        self.foreground_grids = grids[foreground_indices]
        self.foreground_fields = fields[foreground_indices]
        
        self.n_backgrounds = n_backgrounds
        self.n_foregrounds = len(foreground_indices)
        
        print(f"MultiBkgDataset initialized:")
        print(f"  - {self.n_backgrounds} background pairs reserved")
        print(f"  - {self.n_foregrounds} foreground pairs available")
        print(f"  - Total dataset size: {len(self)} (each foreground paired with each background)")
    
    def __len__(self) -> int:
        """Each foreground is paired with each background."""
        return self.n_foregrounds * self.n_backgrounds
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a contrast pair by combining a foreground with a background.
        
        Returns:
            input_tensor: Combined grid (foreground + background)
            target_tensor: Combined field (foreground + background)
        """
        # Calculate which foreground and background to use
        fg_idx = idx // self.n_backgrounds
        bg_idx = idx % self.n_backgrounds
        
        # Get foreground and background data
        fg_grid = self.foreground_grids[fg_idx]  # Shape: (H, W, 2)
        fg_sct_field = self.foreground_fields[fg_idx]  # Shape: (H, W, 2)
        bg_grid = self.background_grids[bg_idx]  # Shape: (H, W, 2)
        bg_sct_field = self.background_fields[bg_idx]  # Shape: (H, W, 2)
        
        # Create contrast pairs using complex arithmetic
        # Convert to complex domain for proper division
        fg_complex = fg_grid[..., 0] + 1j * fg_grid[..., 1]  # real + j*imag
        bg_complex = bg_grid[..., 0] + 1j * bg_grid[..., 1]  # real + j*imag
        
        # Compute contrast in complex domain
        contrast_complex = (fg_complex - bg_complex) / bg_complex
        
        # Convert back to real/imaginary representation
        contrast = np.stack([contrast_complex.real, contrast_complex.imag], axis=-1)
        
        sct_field = fg_sct_field - bg_sct_field
        
        # Convert to PyTorch tensors and change from (H,W,C) to (C,H,W)
        input_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(sct_field, (2, 0, 1)))
        ).to(self.dtype)
        
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(contrast, (2, 0, 1)))
        ).to(self.dtype)

        # Convert extra returns to tensors and transpose them as well
        fg_grid_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(fg_grid, (2, 0, 1)))
        ).to(self.dtype)
        
        bg_grid_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(bg_grid, (2, 0, 1)))
        ).to(self.dtype)
        
        fg_sct_field_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(fg_sct_field, (2, 0, 1)))
        ).to(self.dtype)
        
        bg_sct_field_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(bg_sct_field, (2, 0, 1)))
        ).to(self.dtype)
        
        return input_tensor, target_tensor, fg_grid_tensor, bg_grid_tensor, fg_sct_field_tensor, bg_sct_field_tensor
    
    def get_background_info(self) -> dict:
        """Return information about the reserved backgrounds."""
        return {
            'n_backgrounds': self.n_backgrounds,
            'background_grid_shape': self.background_grids.shape,
            'background_field_shape': self.background_fields.shape,
        }
    
    def get_original_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an original foreground pair without background augmentation for a given dataset index.
        
        Args:
            idx: Regular dataset index (same as used in __getitem__)
            
        Returns:
            field_tensor: Original field data (input, foreground only, no background)
            grid_tensor: Original grid data (target, foreground only, no background)
        """
        if idx >= len(self):
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self)})")
        
        # Calculate which foreground this dataset index corresponds to
        fg_idx = idx // self.n_backgrounds
        
        # Get the original foreground data (no background augmentation)
        grid = self.foreground_grids[fg_idx]
        field = self.foreground_fields[fg_idx]
        
        # Convert to PyTorch tensors and change from (H,W,C) to (C,H,W)
        grid_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(grid, (2, 0, 1)))
        ).to(self.dtype)
        
        field_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(field, (2, 0, 1)))
        ).to(self.dtype)
        
        return field_tensor, grid_tensor
    
    def get_background_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the background pair corresponding to a dataset index.
        
        Args:
            idx: Regular dataset index (same as used in __getitem__)
            
        Returns:
            bg_field_tensor: Background field data
            bg_grid_tensor: Background grid data
        """
        if idx >= len(self):
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self)})")
        
        # Calculate which background this dataset index corresponds to
        bg_idx = idx % self.n_backgrounds
        
        # Get the background data
        bg_grid = self.background_grids[bg_idx]
        bg_field = self.background_fields[bg_idx]
        
        # Convert to PyTorch tensors and change from (H,W,C) to (C,H,W)
        bg_grid_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(bg_grid, (2, 0, 1)))
        ).to(self.dtype)
        
        bg_field_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(bg_field, (2, 0, 1)))
        ).to(self.dtype)
        
        return bg_field_tensor, bg_grid_tensor
    
    def get_indices_info(self, idx: int) -> dict:
        """
        Get information about which foreground and background indices correspond to a dataset index.
        
        Args:
            idx: Regular dataset index (same as used in __getitem__)
            
        Returns:
            dict with 'fg_idx', 'bg_idx', and other useful info
        """
        if idx >= len(self):
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self)})")
        
        fg_idx = idx // self.n_backgrounds
        bg_idx = idx % self.n_backgrounds
        
        return {
            'dataset_idx': idx,
            'fg_idx': fg_idx,
            'bg_idx': bg_idx,
            'total_foregrounds': self.n_foregrounds,
            'total_backgrounds': self.n_backgrounds
        }
    
    def get_backgrounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all reserved background grids and fields.
        
        Returns:
            bg_grids_tensor: Tensor of shape (n_backgrounds, 2, H, W)
            bg_fields_tensor: Tensor of shape (n_backgrounds, 2, H, W)
        """
        bg_grids_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(self.background_grids, (0, 3, 1, 2)))
        ).to(self.dtype)
        
        bg_fields_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(self.background_fields, (0, 3, 1, 2)))
        ).to(self.dtype)
        
        return bg_grids_tensor, bg_fields_tensor


class LimitedExampleMultiBkgDataset(MultiBkgDataset):
    """
    A subclass of MultiBkgDataset that limits the number of foreground training examples
    while still using all the backgrounds.
    
    For example, with n_examples=1 and n_backgrounds=25, this will:
    - Select 25 backgrounds
    - Select only 1 foreground example
    - Create 25 contrast maps by pairing that 1 foreground with each of the 25 backgrounds
    
    This is useful for testing model performance with limited training data.
    """
    
    def __init__(
        self, 
        grids: np.ndarray, 
        fields: np.ndarray, 
        n_backgrounds: int,
        n_examples: int,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None
    ):
        """
        Initialize the LimitedExampleMultiBkgDataset.
        
        Args:
            grids: Array of shape (N, H, W, 2) containing grid data (real/imag)
            fields: Array of shape (N, H, W, 2) containing field data (real/imag)
            n_backgrounds: Number of pairs to reserve as backgrounds
            n_examples: Number of foreground examples to use (will be paired with all backgrounds)
            dtype: PyTorch dtype for output tensors
            seed: Random seed for reproducible background and example selection
        """
        assert grids.ndim == 4 and fields.ndim == 4, "Expected (N,H,W,C) arrays"
        assert grids.shape[0] == fields.shape[0], "Mismatched batch sizes"
        assert grids.shape[-1] == 2 and fields.shape[-1] == 2, "Expected 2-channel real/imag in last dim"
        assert n_backgrounds > 0, "Must have at least 1 background"
        assert n_examples > 0, "Must have at least 1 example"
        assert n_backgrounds + n_examples <= grids.shape[0], "n_backgrounds + n_examples must be <= total samples"
        
        self.dtype = dtype
        
        # Set random seed for reproducible selection
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        total_samples = grids.shape[0]
        
        # First, randomly select background indices
        background_indices = np.random.choice(total_samples, n_backgrounds, replace=False)
        
        # Then, select foreground indices from the remaining samples
        remaining_indices = np.setdiff1d(np.arange(total_samples), background_indices)
        foreground_indices = np.random.choice(remaining_indices, n_examples, replace=False)
        
        # Store backgrounds
        self.background_grids = grids[background_indices]
        self.background_fields = fields[background_indices]
        
        # Store limited foregrounds
        self.foreground_grids = grids[foreground_indices]
        self.foreground_fields = fields[foreground_indices]
        
        self.n_backgrounds = n_backgrounds
        self.n_foregrounds = n_examples  # This is now limited to n_examples
        
        print(f"LimitedExampleMultiBkgDataset initialized:")
        print(f"  - {self.n_backgrounds} background pairs reserved")
        print(f"  - {self.n_foregrounds} foreground examples (LIMITED)")
        print(f"  - Total dataset size: {len(self)} (each of {n_examples} foreground(s) paired with each of {n_backgrounds} background(s))")
        print(f"  - Dataset will contain {n_examples} × {n_backgrounds} = {len(self)} samples")


def test_multibkg_dataset():
    """Test the standard MultiBkgDataset with contrast math verification."""
    import torch
    from torch.utils.data import random_split
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "="*70)
    print("TESTING MultiBkgDataset")
    print("="*70)
    print("\nSimple test to verify contrast math...")
    
    # Load data
    grids, _, cal_e_fields, _ = read_mat_file('all_data.mat')
    grids = split_complex_to_real_imag(grids)
    cal_e_fields = split_complex_to_real_imag(cal_e_fields)
    
    # Create dataset
    dataset = MultiBkgDataset(grids=grids, fields=cal_e_fields, n_backgrounds=5, seed=42)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Dataset size: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Test on a random train sample
    train_idx = 100  # arbitrary index
    original_idx = train_dataset.indices[train_idx]
    
    # Get data from dataset
    field, contrast, _, _, _, _ = dataset[original_idx]
    original_field, original_grid = dataset.get_original_pair(original_idx)
    bg_field, bg_grid = dataset.get_background_pair(original_idx)
    
    print(f"\nTesting train sample {train_idx} (original idx {original_idx})")
    
    # Verify math: contrast = (original - background) / background
    # So: original = contrast * background + background = background * (contrast + 1)
    
    # Convert to complex for math
    contrast_complex = contrast[0] + 1j * contrast[1]  # (H, W)
    bg_complex = bg_grid[0] + 1j * bg_grid[1]  # (H, W) 
    original_complex = original_grid[0] + 1j * original_grid[1]  # (H, W)
    
    # Reconstruct original from contrast and background
    reconstructed_complex = bg_complex * (contrast_complex + 1)
    
    # Check if reconstruction matches original
    error = np.abs(reconstructed_complex - original_complex).mean()
    max_error = np.abs(reconstructed_complex - original_complex).max()
    
    print(f"Reconstruction error (mean): {error:.8f}")
    print(f"Reconstruction error (max): {max_error:.8f}")
    print(f"Math verification: {'✅ PASSED' if error < 1e-6 else '❌ FAILED'}")
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Contrast Math Verification - Train Sample {train_idx}', fontsize=14)
    
    # Original grid (real and imaginary)
    im1 = axes[0, 0].imshow(original_grid[0].cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Original Grid (Real)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[1, 0].imshow(original_grid[1].cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Original Grid (Imag)')
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    
    # Background grid (real and imaginary)
    im3 = axes[0, 1].imshow(bg_grid[0].cpu().numpy(), cmap='viridis')
    axes[0, 1].set_title('Background Grid (Real)')
    plt.colorbar(im3, ax=axes[0, 1], shrink=0.8)
    
    im4 = axes[1, 1].imshow(bg_grid[1].cpu().numpy(), cmap='viridis')
    axes[1, 1].set_title('Background Grid (Imag)')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    # Contrast (real and imaginary)
    im5 = axes[0, 2].imshow(contrast[0].cpu().numpy(), cmap='viridis')
    axes[0, 2].set_title('Contrast (Real)')
    plt.colorbar(im5, ax=axes[0, 2], shrink=0.8)
    
    im6 = axes[1, 2].imshow(contrast[1].cpu().numpy(), cmap='viridis')
    axes[1, 2].set_title('Contrast (Imag)')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    # Reconstructed grid (real and imaginary)
    im7 = axes[0, 3].imshow(reconstructed_complex.real, cmap='viridis')
    axes[0, 3].set_title('Reconstructed (Real)')
    plt.colorbar(im7, ax=axes[0, 3], shrink=0.8)
    
    im8 = axes[1, 3].imshow(reconstructed_complex.imag, cmap='viridis')
    axes[1, 3].set_title('Reconstructed (Imag)')
    plt.colorbar(im8, ax=axes[1, 3], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Also test on a val sample
    val_idx = 50  # arbitrary index
    original_idx = val_dataset.indices[val_idx]
    
    field, contrast, _, _, _, _ = dataset[original_idx]
    original_field, original_grid = dataset.get_original_pair(original_idx)
    bg_field, bg_grid = dataset.get_background_pair(original_idx)
    
    print(f"\nTesting val sample {val_idx} (original idx {original_idx})")
    
    # Same math verification
    contrast_complex = contrast[0] + 1j * contrast[1]
    bg_complex = bg_grid[0] + 1j * bg_grid[1]
    original_complex = original_grid[0] + 1j * original_grid[1]
    reconstructed_complex = bg_complex * (contrast_complex + 1)
    
    error = np.abs(reconstructed_complex - original_complex).mean()
    max_error = np.abs(reconstructed_complex - original_complex).max()
    
    print(f"Reconstruction error (mean): {error:.8f}")
    print(f"Reconstruction error (max): {max_error:.8f}")
    print(f"Math verification: {'✅ PASSED' if error < 1e-6 else '❌ FAILED'}")
    
    # Test that backgrounds are identical across train/val subsets
    print(f"\n{'='*50}")
    print("TESTING BACKGROUND CONSISTENCY ACROSS TRAIN/VAL SUBSETS")
    print(f"{'='*50}")
    
    # Get backgrounds from original dataset
    original_bg_grids, original_bg_fields = dataset.get_backgrounds()
    
    # Test a few samples from train subset
    print("\nTesting background consistency for train samples...")
    train_bg_consistent = True
    for i in range(min(5, len(train_dataset))):
        original_idx = train_dataset.indices[i]
        bg_field, bg_grid = dataset.get_background_pair(original_idx)
        
        # Determine which background index this corresponds to
        bg_idx = original_idx % dataset.n_backgrounds
        
        # Compare with original backgrounds
        expected_bg_grid = original_bg_grids[bg_idx]
        expected_bg_field = original_bg_fields[bg_idx]
        
        grid_match = torch.allclose(bg_grid, expected_bg_grid, atol=1e-8)
        field_match = torch.allclose(bg_field, expected_bg_field, atol=1e-8)
        
        if not (grid_match and field_match):
            train_bg_consistent = False
            print(f"  ❌ Train sample {i} (orig_idx {original_idx}, bg_idx {bg_idx}): grid_match={grid_match}, field_match={field_match}")
        else:
            print(f"  ✅ Train sample {i} (orig_idx {original_idx}, bg_idx {bg_idx}): backgrounds match")
    
    # Test a few samples from val subset  
    print("\nTesting background consistency for val samples...")
    val_bg_consistent = True
    for i in range(min(5, len(val_dataset))):
        original_idx = val_dataset.indices[i]
        bg_field, bg_grid = dataset.get_background_pair(original_idx)
        
        # Determine which background index this corresponds to
        bg_idx = original_idx % dataset.n_backgrounds
        
        # Compare with original backgrounds
        expected_bg_grid = original_bg_grids[bg_idx]
        expected_bg_field = original_bg_fields[bg_idx]
        
        grid_match = torch.allclose(bg_grid, expected_bg_grid, atol=1e-8)
        field_match = torch.allclose(bg_field, expected_bg_field, atol=1e-8)
        
        if not (grid_match and field_match):
            val_bg_consistent = False
            print(f"  ❌ Val sample {i} (orig_idx {original_idx}, bg_idx {bg_idx}): grid_match={grid_match}, field_match={field_match}")
        else:
            print(f"  ✅ Val sample {i} (orig_idx {original_idx}, bg_idx {bg_idx}): backgrounds match")
    
    # Cross-check: verify train and val see same backgrounds for same bg_idx
    print("\nCross-checking: train vs val backgrounds for same bg_idx...")
    cross_check_passed = True
    
    # Find a background index that appears in both train and val
    train_bg_indices = set(train_dataset.indices[i] % dataset.n_backgrounds for i in range(len(train_dataset)))
    val_bg_indices = set(val_dataset.indices[i] % dataset.n_backgrounds for i in range(len(val_dataset)))
    common_bg_indices = train_bg_indices.intersection(val_bg_indices)
    
    if len(common_bg_indices) > 0:
        test_bg_idx = list(common_bg_indices)[0]
        
        # Find samples from train and val that use this background
        train_sample_with_bg = None
        val_sample_with_bg = None
        
        for i in range(len(train_dataset)):
            if train_dataset.indices[i] % dataset.n_backgrounds == test_bg_idx:
                train_sample_with_bg = train_dataset.indices[i]
                break
                
        for i in range(len(val_dataset)):
            if val_dataset.indices[i] % dataset.n_backgrounds == test_bg_idx:
                val_sample_with_bg = val_dataset.indices[i]
                break
        
        if train_sample_with_bg is not None and val_sample_with_bg is not None:
            train_bg_field, train_bg_grid = dataset.get_background_pair(train_sample_with_bg)
            val_bg_field, val_bg_grid = dataset.get_background_pair(val_sample_with_bg)
            
            grid_match = torch.allclose(train_bg_grid, val_bg_grid, atol=1e-8)
            field_match = torch.allclose(train_bg_field, val_bg_field, atol=1e-8)
            
            if grid_match and field_match:
                print(f"  ✅ Background {test_bg_idx} identical between train (orig_idx {train_sample_with_bg}) and val (orig_idx {val_sample_with_bg})")
            else:
                cross_check_passed = False
                print(f"  ❌ Background {test_bg_idx} differs between train and val!")
        else:
            print(f"  ⚠️  Could not find samples in both train and val for bg_idx {test_bg_idx}")
    else:
        print(f"  ⚠️  No common background indices between train and val sets")
    
    # Summary
    overall_bg_consistent = train_bg_consistent and val_bg_consistent and cross_check_passed
    
    if overall_bg_consistent:
        print(f"\n✅ BACKGROUND CONSISTENCY TEST PASSED!")
        print("  - Train subset backgrounds match original dataset")
        print("  - Val subset backgrounds match original dataset") 
        print("  - Same background indices return identical data across subsets")
    else:
        print(f"\n❌ BACKGROUND CONSISTENCY TEST FAILED!")
        if not train_bg_consistent:
            print("  - Train subset background mismatch detected")
        if not val_bg_consistent:
            print("  - Val subset background mismatch detected")
        if not cross_check_passed:
            print("  - Cross-subset background inconsistency detected")
    
    print(f"\n{'='*50}")
    print("✅ MultiBkgDataset test completed!")
    print(f"{'='*50}\n")


def test_limited_example_dataset():
    """Test the LimitedExampleMultiBkgDataset with few examples and many backgrounds."""
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "="*70)
    print("TESTING LimitedExampleMultiBkgDataset")
    print("="*70)
    
    # Load data
    grids, _, cal_e_fields, _ = read_mat_file('all_data.mat')
    grids = split_complex_to_real_imag(grids)
    cal_e_fields = split_complex_to_real_imag(cal_e_fields)
    
    # Create dataset with limited examples
    n_examples = 2
    n_backgrounds = 10
    dataset = LimitedExampleMultiBkgDataset(
        grids=grids, 
        fields=cal_e_fields, 
        n_backgrounds=n_backgrounds,
        n_examples=n_examples,
        seed=42
    )
    
    print(f"\nDataset created with {n_examples} examples and {n_backgrounds} backgrounds")
    print(f"Total dataset size: {len(dataset)} (should be {n_examples * n_backgrounds})")
    assert len(dataset) == n_examples * n_backgrounds, "Dataset size mismatch!"
    
    # Verify that we only have n_examples unique foregrounds
    print(f"\nVerifying that only {n_examples} unique foregrounds are used...")
    unique_fg_indices = set()
    for idx in range(len(dataset)):
        info = dataset.get_indices_info(idx)
        unique_fg_indices.add(info['fg_idx'])
    
    print(f"  Found {len(unique_fg_indices)} unique foreground indices: {sorted(unique_fg_indices)}")
    assert len(unique_fg_indices) == n_examples, f"Expected {n_examples} unique foregrounds, got {len(unique_fg_indices)}"
    print(f"  ✅ Correct number of unique foregrounds!")
    
    # Verify that all backgrounds are used
    print(f"\nVerifying that all {n_backgrounds} backgrounds are used...")
    unique_bg_indices = set()
    for idx in range(len(dataset)):
        info = dataset.get_indices_info(idx)
        unique_bg_indices.add(info['bg_idx'])
    
    print(f"  Found {len(unique_bg_indices)} unique background indices: {sorted(unique_bg_indices)}")
    assert len(unique_bg_indices) == n_backgrounds, f"Expected {n_backgrounds} unique backgrounds, got {len(unique_bg_indices)}"
    print(f"  ✅ All backgrounds are used!")
    
    # Verify contrast math for a few samples
    print(f"\nVerifying contrast math for sample indices...")
    test_indices = [0, 5, len(dataset) - 1]  # First, middle, last
    
    for idx in test_indices:
        field, contrast, _, _, _, _ = dataset[idx]
        original_field, original_grid = dataset.get_original_pair(idx)
        bg_field, bg_grid = dataset.get_background_pair(idx)
        
        # Convert to complex for math
        contrast_complex = contrast[0] + 1j * contrast[1]
        bg_complex = bg_grid[0] + 1j * bg_grid[1]
        original_complex = original_grid[0] + 1j * original_grid[1]
        
        # Reconstruct original from contrast and background
        reconstructed_complex = bg_complex * (contrast_complex + 1)
        
        # Check if reconstruction matches original
        error = np.abs(reconstructed_complex - original_complex).mean()
        max_error = np.abs(reconstructed_complex - original_complex).max()
        
        info = dataset.get_indices_info(idx)
        status = '✅' if error < 1e-6 else '❌'
        print(f"  {status} Sample {idx} (fg={info['fg_idx']}, bg={info['bg_idx']}): mean_error={error:.8f}, max_error={max_error:.8f}")
        assert error < 1e-6, f"Reconstruction error too large for sample {idx}!"
    
    print(f"\n✅ All contrast math verifications passed!")
    
    # Test that each foreground is paired with all backgrounds
    print(f"\nVerifying that each foreground is paired with all backgrounds...")
    for fg_idx in range(n_examples):
        # Find all dataset indices that use this foreground
        indices_with_this_fg = [i for i in range(len(dataset)) if dataset.get_indices_info(i)['fg_idx'] == fg_idx]
        
        # Get the background indices for these samples
        bg_indices_for_this_fg = [dataset.get_indices_info(i)['bg_idx'] for i in indices_with_this_fg]
        
        print(f"  Foreground {fg_idx} is paired with backgrounds: {sorted(bg_indices_for_this_fg)}")
        assert len(bg_indices_for_this_fg) == n_backgrounds, f"Foreground {fg_idx} not paired with all backgrounds!"
        assert len(set(bg_indices_for_this_fg)) == n_backgrounds, f"Foreground {fg_idx} has duplicate background pairings!"
        assert set(bg_indices_for_this_fg) == set(range(n_backgrounds)), f"Foreground {fg_idx} missing some backgrounds!"
    
    print(f"  ✅ Each foreground is correctly paired with all backgrounds!")
    
    print(f"\n{'='*50}")
    print("✅ LimitedExampleMultiBkgDataset test completed!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Run tests
    test_multibkg_dataset()
    test_limited_example_dataset()