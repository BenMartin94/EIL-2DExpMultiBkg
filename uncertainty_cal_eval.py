import numpy as np
from scipy.stats import norm, pearsonr

def calibration_curve(mean, std, targets, pis=None):
    """
    Compute calibration curve for regression uncertainty using CONFIDENCE INTERVALS.
    
    This measures: "Does my X% confidence interval actually contain the true value X% of the time?"

    Args:
        mean (np.ndarray): predicted means, shape [N] or [B, ...] (flattenable).
        std (np.ndarray): predicted std devs (same shape as mean).
        targets (np.ndarray): ground-truth values (same shape).
        pis (list or np.ndarray): confidence levels in [0,1] (e.g., 0.90 for 90% CI).
                                  If None, defaults to 30 points between 0.01 and 0.99.

    Returns:
        expected_conf (np.ndarray): pis (expected confidence levels)
        observed_conf (np.ndarray): empirical coverage for each confidence level
    """
    # Flatten everything
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    targets = targets.reshape(-1)

    if pis is None:
        pis = np.linspace(0.01, 0.99, 30)

    observed_conf = []
    for p in pis:
        # For a p-level confidence interval centered around mean:
        # CI = [mean - z*std, mean + z*std]
        # where z is chosen such that a Gaussian has p probability mass in [-z, z]
        z = norm.ppf((1 + p) / 2)  # e.g., for p=0.95, z ≈ 1.96
        
        lower = mean - z * std
        upper = mean + z * std
        
        # Check what fraction of targets fall within their CI
        in_interval = (targets >= lower) & (targets <= upper)
        obs = np.mean(in_interval)
        observed_conf.append(obs)

    return np.array(pis), np.array(observed_conf)


def expected_calibration_error(mean, std, targets, pis=None):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the average deviation between expected and observed 
    confidence levels. Lower is better (perfect = 0).
    
    Args:
        mean, std, targets: same as calibration_curve
        pis: confidence levels to evaluate
    
    Returns:
        ece (float): Expected Calibration Error
    """
    expected_conf, observed_conf = calibration_curve(mean, std, targets, pis)
    ece = np.mean(np.abs(expected_conf - observed_conf))
    return ece

def wei_ece(mean, std, targets):
    dirac_delta = 0.05
    pj = norm.cdf(mean + dirac_delta, loc=mean, scale=std) - norm.cdf(mean - dirac_delta, loc=mean, scale=std)
    pj_flat = pj.reshape(-1)
    L = 20
    delta_p = 1 / L
    creds = []
    Sl_list = []
    for l in range(L):
        bin_lower = l * delta_p
        bin_upper = (l + 1) * delta_p
        in_bin = (pj_flat > bin_lower) & (pj_flat <= bin_upper)
        Sl_delta = np.where(in_bin)[0]
        Sl_list.append(Sl_delta)
        # now average of pjs in Sl_delta
        if len(Sl_delta) > 0:
            cred = np.mean(pj_flat[Sl_delta])
        else:
            cred = 0
        creds.append(cred)

        # now find accuracy in Sl_delta
    accs = []
    for Sl_delta in Sl_list:
        if len(Sl_delta) > 0:
            correct = 0
            for idx in Sl_delta:
                # check if target is within mean ± dirac_delta
                if (targets.reshape(-1)[idx] >= (mean.reshape(-1)[idx] - dirac_delta)) and (targets.reshape(-1)[idx] <= (mean.reshape(-1)[idx] + dirac_delta)):
                    correct += 1
            acc = correct / len(Sl_delta)
        else:
            acc = 0
        accs.append(acc)
    M = len(mean.reshape(-1))
    ece = 0
    for i in range(L - 1):
        ece += (len(Sl_list[i]) / M) * np.abs(creds[i] - accs[i])
    return ece


def sharpness(std):
    """
    Compute sharpness: average width of prediction intervals.
    
    Lower sharpness = tighter/more informative predictions.
    Should be balanced with good calibration.
    
    Args:
        std (np.ndarray): predicted standard deviations
    
    Returns:
        sharp (float): mean standard deviation
    """
    return np.mean(std.reshape(-1))


def confidence_interval_coverage(mean, std, targets, confidence_level=0.95):
    """
    Check if a SPECIFIC confidence interval has correct coverage.
    
    This is essentially one point on the calibration curve.
    For a 95% confidence interval, ~95% of true values should fall within it.
    
    Example interpretation:
    - coverage = 0.95 → GOOD! Your 95% CI is well-calibrated
    - coverage = 0.60 → BAD! You're overconfident (intervals too narrow)
    - coverage = 0.99 → BAD! You're underconfident (intervals too wide)
    
    Args:
        mean (np.ndarray): predicted means
        std (np.ndarray): predicted standard deviations
        targets (np.ndarray): ground truth values
        confidence_level (float): e.g., 0.95 for 95% CI
    
    Returns:
        coverage (float): fraction of samples where target is in CI
        lower_bound (np.ndarray): lower CI bounds for each sample
        upper_bound (np.ndarray): upper CI bounds for each sample
    """
    # Flatten
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    targets = targets.reshape(-1)
    
    # For Gaussian: p-level CI is mean ± z * std
    z = norm.ppf((1 + confidence_level) / 2)
    
    lower = mean - z * std
    upper = mean + z * std
    
    # Check if targets fall within bounds
    in_interval = (targets >= lower) & (targets <= upper)
    coverage = np.mean(in_interval)
    
    return coverage, lower, upper


def error_std_correlation(mean, std, targets):
    """
    Compute Pearson correlation coefficient between prediction error and standard deviation.
    
    Measures if predicted uncertainty (std) is correlated with actual prediction error.
    Higher correlation means uncertainty estimates are more informative about errors.
    
    Args:
        mean (np.ndarray): predicted means, shape [N] or [B, ...] (flattenable).
        std (np.ndarray): predicted standard deviations (same shape as mean).
        targets (np.ndarray): ground truth values (same shape).
    
    Returns:
        correlation (float): Pearson correlation coefficient between -1 and 1.
                            Values close to 1 indicate high uncertainty corresponds to high error.
        p_value (float): Two-tailed p-value for testing non-correlation.
    """
    # Flatten all arrays
    mean_flat = mean.reshape(-1)
    std_flat = std.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Compute absolute prediction errors
    errors = np.abs(targets_flat - mean_flat)
    
    # Compute Pearson correlation between errors and uncertainties
    correlation, p_value = pearsonr(errors, std_flat)
    
    return correlation, p_value




def test_wei_ece():
    np.random.seed(0)
    n_samples = 1000
    for std_mag in np.linspace(0.01, 0.5, 10):
        targets = np.random.randn(n_samples)
        mean = targets + np.random.randn(n_samples)*0.2
        std = np.ones(n_samples) * std_mag

        ece = wei_ece(mean, std, targets)
        print(f"Std mag: {std_mag:.2f}, WEI-ECE: {ece:.4f}")

def test_correlation_coeff():
    np.random.seed(0)
    n_samples = 1000
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    # make the first half 1
    x[:n_samples//4] = 1
    y[:n_samples//4] = 1
    corr, p_val = error_std_correlation(np.zeros_like(x), x, y)
    print(f"Error-Std Correlation: {corr:.4f}, p-value: {p_val:.4e}")


# ============================================================================
# EXAMPLE: Visualizing the difference
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_wei_ece()
    test_correlation_coeff()
    # np.random.seed(42)
    # n_samples = 1000
    #
    # # True targets
    # targets = np.random.randn(n_samples)
    #
    # # Scenario 1: Well-calibrated
    # mean_good = targets + np.random.randn(n_samples) * 0.3
    # std_good = np.ones(n_samples) * 0.3
    #
    # # Scenario 2: Overconfident (std too small)
    # mean_over = targets + np.random.randn(n_samples) * 0.5
    # std_over = np.ones(n_samples) * 0.1  # Claims high confidence but large errors!
    #
    # # Scenario 3: Underconfident (std too large)
    # mean_under = targets + np.random.randn(n_samples) * 0.1
    # std_under = np.ones(n_samples) * 1.0  # Claims low confidence but small errors!
    #
    # # Compute calibration curves
    # pis_good, obs_good = calibration_curve(mean_good, std_good, targets)
    # pis_over, obs_over = calibration_curve(mean_over, std_over, targets)
    # pis_under, obs_under = calibration_curve(mean_under, std_under, targets)
    #
    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(pis_good, obs_good, 'o-', label='Well-calibrated', linewidth=2, markersize=4)
    # plt.plot(pis_over, obs_over, 's-', label='Overconfident (below diagonal)', linewidth=2, markersize=4)
    # plt.plot(pis_under, obs_under, '^-', label='Underconfident (above diagonal)', linewidth=2, markersize=4)
    # plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    # plt.xlabel('Expected Confidence Level', fontsize=13)
    # plt.ylabel('Observed Coverage', fontsize=13)
    # plt.title('Calibration Curve: Confidence Interval Coverage', fontsize=14)
    # plt.legend(fontsize=11)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('calibration_corrected.png', dpi=150)
    # plt.show()
    #
    # # Compute metrics
    # print("\n" + "="*70)
    # print("CALIBRATION METRICS")
    # print("="*70)
    #
    # ece_good = expected_calibration_error(mean_good, std_good, targets)
    # ece_over = expected_calibration_error(mean_over, std_over, targets)
    # ece_under = expected_calibration_error(mean_under, std_under, targets)
    #
    # print(f"\nExpected Calibration Error (ECE):")
    # print(f"  Well-calibrated:  {ece_good:.4f}")
    # print(f"  Overconfident:    {ece_over:.4f}  ← High ECE = bad!")
    # print(f"  Underconfident:   {ece_under:.4f}  ← High ECE = bad!")
    #
    # # Check specific CI coverage
    # cov_good, _, _ = confidence_interval_coverage(mean_good, std_good, targets, 0.95)
    # cov_over, _, _ = confidence_interval_coverage(mean_over, std_over, targets, 0.95)
    # cov_under, _, _ = confidence_interval_coverage(mean_under, std_under, targets, 0.95)
    #
    # print(f"\n95% Confidence Interval Coverage (should be ~0.95):")
    # print(f"  Well-calibrated:  {cov_good:.3f}  ← Close to 0.95 ✓")
    # print(f"  Overconfident:    {cov_over:.3f}  ← Too low! Intervals miss true values")
    # print(f"  Underconfident:   {cov_under:.3f}  ← Too high! Intervals too wide")
    # print("="*70 + "\n")
    #
