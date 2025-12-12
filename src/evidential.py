import torch
import torch.nn.functional as F

from src.Unet import UNet


def evidential_NLL(eps_gt, gamma, v, alpha, beta):
    """
    Compute the Negative Log-Likelihood (NLL) loss for evidential regression.
    
    Based on the NIG (Normal Inverse-Gamma) distribution.
    This is the key loss that trains the uncertainty estimates.

    Parameters:
    eps_gt : torch.Tensor
        The true target values.
    gamma : torch.Tensor
        The predicted mean values.
    v : torch.Tensor
        The predicted variance scaling factor (pseudo-observations).
    alpha : torch.Tensor
        The predicted shape parameter of the Inverse-Gamma distribution.
    beta : torch.Tensor
        The predicted scale parameter of the Inverse-Gamma distribution.

    Returns:
    torch.Tensor
        The computed NLL loss.
    """
    # Compute the NLL for Normal Inverse-Gamma distribution
    # This measures how well the uncertainty estimates match the data
    omega = 2 * beta * (1 + v)
    
    nll = 0.5 * torch.log(torch.pi / v) \
        - alpha * torch.log(omega) \
        + (alpha + 0.5) * torch.log(v * (eps_gt - gamma) ** 2 + omega) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    
    return nll.mean()

def reg_loss_1(eps_gt, gamma, v, alpha):
    """
    Compute the regularization loss for evidential regression.
    
    This penalizes the network for having high evidence (low uncertainty)
    when the prediction is wrong. This is CRITICAL to prevent the blob issue.

    Parameters:
    eps_gt : torch.Tensor
        The true target values.
    gamma : torch.Tensor
        The predicted mean values.
    v : torch.Tensor
        The predicted variance scaling factor (evidence).
    alpha : torch.Tensor
        The predicted shape parameter.

    Returns:
    torch.Tensor
        The computed regularization loss.
    """
    # When prediction is wrong, penalize high evidence
    # This forces the network to output high uncertainty when uncertain
    error = torch.abs(eps_gt - gamma)
    reg = error * (2 * v + alpha)
    return reg.mean()


def reg_loss_2(eps_gt, gamma, v, alpha, beta):
    """
    Compute an alternative regularization loss for evidential regression.
    
    This is another form that helps prevent the network from outputting
    unreasonably small variances.

    Parameters:
    eps_gt : torch.Tensor
        The true target values.
    gamma : torch.Tensor
        The predicted mean values.
    v : torch.Tensor
        The predicted variance scaling factor.
    alpha : torch.Tensor
        The predicted shape parameter.
    beta : torch.Tensor
        The predicted scale parameter.

    Returns:
    torch.Tensor
        The computed alternative regularization loss.
    """
    error = (eps_gt - gamma) ** 2
    # Prevent division by very small beta
    reg = error * (2 * alpha + v) / (beta + 1e-10)
    return reg.mean()

class EvidentialUnet(UNet):
    def __init__(self, in_channels, out_channels, base_channels, bilinear=True):
        # Output 4 parameters per channel: gamma, v, alpha, beta
        super(EvidentialUnet, self).__init__(in_channels, out_channels * 4, base_channels, bilinear)
        self.out_channels = out_channels

    def forward(self, x):
        x = super().forward(x)
        # Split the output into gamma, v, alpha, beta for each output channel
        # x shape: (B, out_channels*4, H, W)
        
        # Predicted mean (no activation - can be any real value)
        gamma = x[:, 0:self.out_channels, :, :]
        
        # Evidence parameter (v): Number of "pseudo-observations"
        # Higher v = more confident, lower v = less confident
        # Use softplus to ensure positive, but DON'T clamp too aggressively
        v = F.softplus(x[:, self.out_channels:2*self.out_channels, :, :])
        # Only prevent it from being too small (which would cause numerical issues)
        v = torch.clamp(v, min=0.01)  # Remove upper clamp to allow high confidence when appropriate
        
        # Shape parameter (alpha): Must be > 1 for mean to exist, > 2 for variance
        # But don't add too much! Start smaller to allow network to learn
        alpha = F.softplus(x[:, 2*self.out_channels:3*self.out_channels, :, :]) + 1.0
        # Clamp to prevent numerical issues but allow learning
        alpha = torch.clamp(alpha, min=1.01, max=10.0)
        
        # Scale parameter (beta): Related to the variance
        # Should be positive but not clamped too tightly
        beta = F.softplus(x[:, 3*self.out_channels:4*self.out_channels, :, :])
        # Only prevent explosion, but be less restrictive
        beta = torch.clamp(beta, min=0.01, max=10.0)
        
        return gamma, v, alpha, beta
