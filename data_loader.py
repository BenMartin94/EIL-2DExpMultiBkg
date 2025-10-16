"""
Shared data loading utilities for EM field reconstruction.
Used by both standard training and evidential training scripts.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

import main as data_main


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


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from .mat file and convert to real/imaginary representation.
    
    Args:
        file_path: Path to the .mat file containing the data
        
    Returns:
        synth_fields: Synthesized scattered fields (N, 24, 24, 2)
        cal_e_fields: Calibrated E-fields (N, 24, 24, 2)
        grids: Epsilon grids (N, 100, 100, 2)
        uncal_spars: Uncalibrated S-parameters (N, 24, 24, 2)
    """
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
    uncal_spars = uncal_spars.astype(np.float32, copy=False)
    
    return synth_fields, cal_e_fields, grids, uncal_spars
