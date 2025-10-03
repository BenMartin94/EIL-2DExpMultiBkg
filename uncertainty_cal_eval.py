import numpy as np
from scipy.stats import norm

def calibration_curve(mean, std, targets, pis=None):
    """
    Compute calibration curve for regression uncertainty.

    Args:
        mean (np.ndarray): predicted means, shape [N] or [B, ...] (flattenable).
        std (np.ndarray): predicted std devs (same shape as mean).
        targets (np.ndarray): ground-truth values (same shape).
        pis (list or np.ndarray): expected confidence levels in [0,1].
                                  If None, defaults to 20 points between 0.05 and 0.95.

    Returns:
        expected_conf (np.ndarray): pis
        observed_conf (np.ndarray): empirical coverage for each pi
    """
    # Flatten everything
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    targets = targets.reshape(-1)

    if pis is None:
        pis = np.linspace(0.01, 0.99, 30)

    observed_conf = []
    for p in pis:
        q = mean + std * norm.ppf(p)
        # proportion of targets <= q
        obs = np.mean(targets <= q)
        observed_conf.append(obs)

    return np.array(pis), np.array(observed_conf)
