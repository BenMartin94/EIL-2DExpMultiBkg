import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

def litmus_test(
    model,
    in_range_test_loader,
    out_of_range_test_loader,
    output_dir: str = "figures/litmus_tests",
    device: str | None = None,
):
    """
    For each test example, run the anomaly_test function which returns an average error around the background
    The theory here is that when a target is in range of the training data, its performance around the bkg region is great.
    When a target is out of range of the training data, its performance around the bkg region is poor.
    We can use that to determine if a target is in or out of range.

    Args:
        model: The trained model to be tested.
        in_range_test_loader: DataLoader for in-range test data.
        out_of_range_test_loader: DataLoader for out-of-range test data.
        output_dir: Directory to save the output figures.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        A histogram plot saved to the output directory. Two histograms are plotted: one for in-range test data and one for out-of-range test data.
    """

    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running litmus test on device: {device}")

    model.to(device)
    model.eval()
    in_range_errors = []
    out_of_range_errors = []

    batch_size = in_range_test_loader.batch_size
    assert batch_size == out_of_range_test_loader.batch_size, "Batch sizes of the two loaders must be the same."

    nbkgs = model.bkgs.shape[0]
    assert nbkgs > 0, "Model has no backgrounds stored for reconstruction."


    with torch.no_grad():
        for data in tqdm(in_range_test_loader, desc="In-range", unit="batch", total=len(in_range_test_loader)):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            this_batch_size = inputs.shape[0]
            bg_grid = model.bkgs.unsqueeze(0).expand(this_batch_size, -1, -1, -1, -1)
            bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(this_batch_size, -1, -1, -1, -1)
            errors = model.bkgs_mean_error(inputs, bg_sct, bg_grid)
            # anomaly_test may return shape [B, nbkgs]; flatten so we track all values
            in_range_errors.append(errors.flatten())

        for data in tqdm(out_of_range_test_loader, desc="Out-of-range", unit="batch", total=len(out_of_range_test_loader)):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            this_batch_size = inputs.shape[0]
            bg_grid = model.bkgs.unsqueeze(0).expand(this_batch_size, -1, -1, -1, -1)
            bg_sct = model.bkg_sct_fields.unsqueeze(0).expand(this_batch_size, -1, -1, -1, -1)
            errors = model.bkgs_mean_error(inputs, bg_sct, bg_grid)
            # anomaly_test may return shape [B, nbkgs]; flatten so we track all values
            out_of_range_errors.append(errors.flatten())

    in_range_errors = torch.cat(in_range_errors).cpu().numpy()
    out_of_range_errors = torch.cat(out_of_range_errors).cpu().numpy()
    print(f"In-range errors: mean={np.mean(in_range_errors):.4f}, std={np.std(in_range_errors):.4f}")
    print(f"Out-of-range errors: mean={np.mean(out_of_range_errors):.4f}, std={np.std(out_of_range_errors):.4f}")

    # Robust plotting: limit displayed range to 1st–99th percentiles to avoid outliers dominating the bins
    combined = np.concatenate([in_range_errors, out_of_range_errors])
    p_low, p_high = np.percentile(combined, [1, 99])
    # Guard against degenerate ranges
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        p_low = float(np.min(combined))
        p_high = float(np.max(combined))
        if p_high <= p_low:
            p_high = p_low + 1e-6

    n_clip_in = int(((in_range_errors < p_low) | (in_range_errors > p_high)).sum())
    n_clip_out = int(((out_of_range_errors < p_low) | (out_of_range_errors > p_high)).sum())

    plt.figure(figsize=(10, 6))
    plt.hist(
        in_range_errors,
        bins='fd',  # Freedman–Diaconis is robust to outliers
        alpha=0.5,
        label=f'In-Range (clipped {n_clip_in})',
        color='blue',
        density=True,
        range=(p_low, p_high),
    )
    plt.hist(
        out_of_range_errors,
        bins='fd',
        alpha=0.5,
        label=f'Out-of-Range (clipped {n_clip_out})',
        color='red',
        density=True,
        range=(p_low, p_high),
    )
    plt.xlabel('Average Error Around Background')
    plt.ylabel('Density (1–99% range)')
    plt.title(f'Litmus Test: In-Range vs Out-of-Range Errors (showing 1–99% range)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'litmus_test_histogram.png'))
    plt.close()
    print(f"Litmus test histogram saved to {os.path.join(output_dir, 'litmus_test_histogram.png')} (clipped range: [{p_low:.3g}, {p_high:.3g}])")