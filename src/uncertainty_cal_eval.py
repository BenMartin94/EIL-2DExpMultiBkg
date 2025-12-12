import numpy as np
from scipy.stats import norm, pearsonr, t as student_t


def evidential_to_student_t(gamma, v, alpha, beta):
    """
    Convert evidential (NIG) parameters to Student-t distribution parameters.
    
    The predictive distribution from a Normal Inverse-Gamma (NIG) prior is a Student-t.
    
    Args:
        gamma (np.ndarray): predicted mean
        v (np.ndarray): evidence parameter (pseudo-observations)
        alpha (np.ndarray): shape parameter
        beta (np.ndarray): scale parameter
    
    Returns:
        mean (np.ndarray): mean of Student-t (same as gamma)
        scale (np.ndarray): scale parameter for Student-t
        nu (np.ndarray): degrees of freedom for Student-t
    """
    mean = gamma
    nu = 2 * alpha
    scale = np.sqrt(beta * (1 + v) / (v * alpha))
    return mean, scale, nu

def wei_ece(mean, std, targets, distribution='gaussian', nu=None):
    """
    Compute Weighted Expected Calibration Error (WEI-ECE).
    
    Args:
        mean (np.ndarray): predicted means
        std (np.ndarray): predicted standard deviations (or scale parameter for Student-t)
        targets (np.ndarray): ground truth values
        distribution (str): 'gaussian' or 'student_t' to specify the predictive distribution
        nu (np.ndarray, optional): degrees of freedom for Student-t distribution (required if distribution='student_t')
    
    Returns:
        ece (float): Weighted Expected Calibration Error
        creds (np.ndarray): credibility values (expected confidence per bin)
        accs (np.ndarray): accuracy values (observed accuracy per bin)
    """
    dirac_delta = 0.05

    mean_shape = mean.shape
    std_shape = std.shape
    targets_shape = targets.shape

    if len(mean_shape) != len(targets_shape):
        # treat first dimension as K rather than batch
        pj = np.zeros(mean_shape[1:])
        K = mean_shape[0]
        for k in range(K):
            if distribution == 'student_t':
                if nu is None:
                    raise ValueError("nu (degrees of freedom) must be provided for Student-t distribution")
                # For Student-t: use loc and scale arguments
                nu_k = nu[k] if len(nu.shape) > len(mean_shape) - 1 else nu
                pj += student_t.cdf(mean[k] + dirac_delta, df=nu_k, loc=mean[k], scale=std[k]) - \
                      student_t.cdf(mean[k] - dirac_delta, df=nu_k, loc=mean[k], scale=std[k])
            else:  # gaussian
                pj += norm.cdf(mean[k] + dirac_delta, loc=mean[k], scale=std[k]) - \
                      norm.cdf(mean[k] - dirac_delta, loc=mean[k], scale=std[k])
        pj /= K  # Average over K MC samples
    else:
        if distribution == 'student_t':
            if nu is None:
                raise ValueError("nu (degrees of freedom) must be provided for Student-t distribution")
            # For Student-t: use loc and scale arguments
            pj = student_t.cdf(mean + dirac_delta, df=nu, loc=mean, scale=std) - \
                 student_t.cdf(mean - dirac_delta, df=nu, loc=mean, scale=std)
        else:  # gaussian
            pj = norm.cdf(mean + dirac_delta, loc=mean, scale=std) - \
                 norm.cdf(mean - dirac_delta, loc=mean, scale=std)

        
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
    M = len(pj_flat)
    ece = 0
    # print("="*20)
    # print("Length of bins:", [len(Sl) for Sl in Sl_list])
    # print("="*20)

    for i in range(L):
        ece += (len(Sl_list[i]) / M) * np.abs(creds[i] - accs[i])
    
    return ece, np.array(creds), np.array(accs)


def confidence_interval_coverage(mean, std, targets, confidence_level=0.95, distribution='gaussian', nu=None):
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
        std (np.ndarray): predicted standard deviations or scale parameter
        targets (np.ndarray): ground truth values
        confidence_level (float): e.g., 0.95 for 95% CI
        distribution (str): 'gaussian' or 'student_t' to specify the predictive distribution
        nu (np.ndarray, optional): degrees of freedom for Student-t (required if distribution='student_t')
    
    Returns:
        coverage (float): fraction of samples where target is in CI
        lower_bound (np.ndarray): lower CI bounds for each sample
        upper_bound (np.ndarray): upper CI bounds for each sample
    """
    # Flatten
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    targets = targets.reshape(-1)
    
    if distribution == 'student_t':
        if nu is None:
            raise ValueError("nu (degrees of freedom) must be provided for Student-t distribution")
        nu = nu.reshape(-1)
        # For Student-t: p-level CI is mean ± t * std (use standardized quantile)
        z = student_t.ppf((1 + confidence_level) / 2, df=nu, loc=0, scale=1)
    else:  # gaussian
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

        ece, creds, accs = wei_ece(mean, std, targets)
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

