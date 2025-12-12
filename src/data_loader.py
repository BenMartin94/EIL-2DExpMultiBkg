"""
Shared data loading utilities for EM field reconstruction.
Used by both standard training and evidential training scripts.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
import scipy.io as sio


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


class AugmentedFieldsDataset(Dataset):
    """
    Dataset wrapper that adds random Gaussian noise to input fields during training.
    
    This is useful for data augmentation to make models more robust to noisy inputs.
    
    Args:
        base_dataset: Underlying dataset (e.g., FieldsDataset or random_split subset)
        noise_std: Standard deviation of Gaussian noise to add (relative to signal)
        noise_type: Type of noise - 'absolute' or 'relative' (default: 'absolute')
                    - 'absolute': adds noise with fixed std
                    - 'relative': adds noise proportional to signal magnitude
    
    Example:
        >>> base_ds = FieldsDataset(x_np, y_np)
        >>> augmented_ds = AugmentedFieldsDataset(base_ds, noise_std=0.01)
        >>> train_loader = DataLoader(augmented_ds, batch_size=32, shuffle=True)
    """
    def __init__(self, base_dataset, noise_std: float = 0.0, noise_type: str = 'absolute', size_multiplier=1):
        self.base_dataset = base_dataset
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.size_multiplier = size_multiplier

    def __len__(self) -> int:
        return len(self.base_dataset) * self.size_multiplier

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx % len(self.base_dataset)]
        
        # Add noise to input if noise_std > 0
        if self.noise_std > 0:
            if self.noise_type == 'relative':
                # Noise proportional to signal magnitude
                signal_magnitude = torch.sqrt(x[0]**2 + x[1]**2).mean()
                noise = torch.randn_like(x) * self.noise_std * signal_magnitude
            else:  # absolute
                # Fixed noise level
                noise = torch.randn_like(x) * self.noise_std
            
            x = x + noise
        
        return x, y


def load_data(file_path: str, seed: int, num_synthetic_test_samples: int = 25) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load data from .mat file and convert to real/imaginary representation.
    
    Args:
        file_path: Path to the .mat file containing the data
        seed: Random seed for shuffling
        num_synthetic_test_samples: Number of samples to reserve for testing
        
    Returns:
        synth_fields: Synthesized scattered fields for training (N-num_test, 24, 24, 2)
        cal_e_fields: Calibrated E-fields for training (N-num_test, 24, 24, 2)
        grids: Epsilon grids for training (N-num_test, 100, 100, 2)
        uncal_spars: Uncalibrated S-parameters for training (N-num_test, 24, 24, 2)
        synth_fields_test: Synthesized scattered fields for testing (num_test, 24, 24, 2)
        cal_e_fields_test: Calibrated E-fields for testing (num_test, 24, 24, 2)
        grids_test: Epsilon grids for testing (num_test, 100, 100, 2)
        uncal_spars_test: Uncalibrated S-parameters for testing (num_test, 24, 24, 2)
    """
    np.random.seed(seed)
    grids, uncal_spars, cal_e_fields, synth_fields = read_mat_file(file_path)
    
    # Convert complex to real/imag last dim
    grids = split_complex_to_real_imag(grids)       # (N,100,100,2)
    synth_fields = split_complex_to_real_imag(synth_fields)  # (N,24,24,2)
    cal_e_fields = split_complex_to_real_imag(cal_e_fields)  # (N,24,24,2)
    uncal_spars = split_complex_to_real_imag(uncal_spars)  # (N,24,24,2)

    # Cast to float32 for training
    grids = grids.astype(np.float32, copy=False)
    synth_fields = synth_fields.astype(np.float32, copy=False)
    cal_e_fields = cal_e_fields.astype(np.float32, copy=False)
    uncal_spars = uncal_spars.astype(np.float32, copy=False)

    # Shuffle the data
    indices = np.arange(len(synth_fields))
    np.random.shuffle(indices)
    synth_fields = synth_fields[indices]
    cal_e_fields = cal_e_fields[indices]
    grids = grids[indices]
    uncal_spars = uncal_spars[indices]

    # Split into train and test sets
    synth_fields_test = synth_fields[:num_synthetic_test_samples]
    cal_e_fields_test = cal_e_fields[:num_synthetic_test_samples]
    grids_test = grids[:num_synthetic_test_samples]
    uncal_spars_test = uncal_spars[:num_synthetic_test_samples]
    
    synth_fields = synth_fields[num_synthetic_test_samples:]
    cal_e_fields = cal_e_fields[num_synthetic_test_samples:]
    grids = grids[num_synthetic_test_samples:]
    uncal_spars = uncal_spars[num_synthetic_test_samples:]

    return (synth_fields, cal_e_fields, grids, uncal_spars,
            synth_fields_test, cal_e_fields_test, grids_test, uncal_spars_test)





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
