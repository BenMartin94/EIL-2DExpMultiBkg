# PyTorch U-Net that maps (N, 2, 24, 24) -> (N, 2, 100, 100)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def bayesian_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Torch implementation of the bayesian_loss function
    """

    y_true = y_true.float()
    mu = y_pred[:, :2, :, :]    # Image part of the prediction
    log_var = y_pred[:, 2:, :, :]  # Uncertainty part of the prediction

    loss = 0.5 * torch.exp(-log_var) * (y_true - mu)**2 + 0.5 * log_var
    return torch.mean(loss)
    #return torch.mean((y_true - y_img) ** 2)


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2, preserving HxW (padding=1). Original version without dropout."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DoubleConvBCNN(nn.Module):
    """(conv => BN => ReLU) * 2 with Dropout for BCNN, preserving HxW (padding=1)."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.05):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DownBCNN(nn.Module):
    """Downscaling with maxpool then double conv with dropout for BCNN."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.05):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBCNN(in_channels, out_channels, dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv.

    Parameters:
      - x1_ch: channels coming from the decoder (deeper layer)
      - x2_ch: channels from the encoder skip connection
      - out_ch: number of output channels after this Up block
    """

    def __init__(self, x1_ch: int, x2_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # Reduce decoder feature maps to out_ch so that cat([skip, up]) has x2_ch + out_ch channels
            self.reduce = nn.Conv2d(x1_ch, out_ch, kernel_size=1)
        else:
            # Transposed conv to both upsample and set channels to out_ch
            self.up = nn.ConvTranspose2d(x1_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        # After concat, channels = out_ch (from up path) + x2_ch (skip)
        self.conv = DoubleConv(out_ch + x2_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        # Pad/crop to match spatial dims of skip connection if needed
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBCNN(nn.Module):
    """Upscaling then double conv with dropout for BCNN.

    Parameters:
      - x1_ch: channels coming from the decoder (deeper layer)
      - x2_ch: channels from the encoder skip connection
      - out_ch: number of output channels after this Up block
    """

    def __init__(self, x1_ch: int, x2_ch: int, out_ch: int, dropout_rate: float = 0.05, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # Reduce decoder feature maps to out_ch so that cat([skip, up]) has x2_ch + out_ch channels
            self.reduce = nn.Conv2d(x1_ch, out_ch, kernel_size=1)
        else:
            # Transposed conv to both upsample and set channels to out_ch
            self.up = nn.ConvTranspose2d(x1_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        # After concat, channels = out_ch (from up path) + x2_ch (skip)
        self.conv = DoubleConvBCNN(out_ch + x2_ch, out_ch, dropout_rate)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        # Pad/crop to match spatial dims of skip connection if needed
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpNoSkip(nn.Module):
    """
    Upscale by 2x without a skip connection, then apply a DoubleConv.
    Keeps feature width at out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        #     self.reduce = nn.Identity()

        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        return self.conv(x)


class UpNoSkipBCNN(nn.Module):
    """
    Upscale by 2x without a skip connection, then apply a DoubleConv with dropout for BCNN.
    Keeps feature width at out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.05, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConvBCNN(out_channels, out_channels, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        return self.conv(x)



class UNet(nn.Module):
    """
    U-Net that maps 2-channel complex field (real, imag) from input size to a target size.
    It reconstructs back to the input resolution, then performs extra up-convolutions
    to exceed the target resolution, and finally downsamples to the exact output size.

    Defaults assume input of (N, 2, 24, 24) and output of (N, 2, 100, 100).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        bilinear: bool = True,
        output_size: Tuple[int, int] = (100, 100),
        input_size: Tuple[int, int] = (24, 24),
    ):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size  # decoder base size before overshoot

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)  # 24 -> 12 -> 6 -> 3 (for default input)

        # Decoder back to input_size
        self.up1 = Up(c4, c3, c3, bilinear)  # 3 -> 6
        self.up2 = Up(c3, c2, c2, bilinear)  # 6 -> 12
        self.up3 = Up(c2, c1, c1, bilinear)  # 12 -> 24

        # Compute extra ups to exceed target size
        cur_h, cur_w = self.input_size
        tgt_h, tgt_w = self.output_size
        extra = 0
        while (cur_h * 2) <= tgt_h and (cur_w * 2) <= tgt_w:
            cur_h *= 2
            cur_w *= 2
            extra += 1

        # Extra upsampling blocks without skips (keep feature width at c1)
        self.extra_ups = nn.ModuleList([UpNoSkip(c1, c1, bilinear) for _ in range(extra)])

        # Final projection to output channels (applied after overshoot)
        self.outc = OutConv(c1, out_channels)

        # Cache whether we will end up shrinking at the end (for interpolation mode choice)
        self._will_shrink = (cur_h > tgt_h) and (cur_w > tgt_w)

    def forward(self, x: torch.Tensor, *, channels_last: bool = False) -> torch.Tensor:
        if channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder back to input_size
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Overshoot beyond target
        for up in self.extra_ups:
            x = up(x)

        # Project to output channels
        x = self.outc(x)


        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        return x


class UNetBCNN(nn.Module):
    """
    U-Net with Dropout layers for Bayesian CNN uncertainty estimation.
    Maps 2-channel complex field (real, imag) from input size to a target size.
    Outputs 4 channels: 2 for predictions (real, imag) + 2 for log variance (real, imag).
    
    This version includes dropout layers that can be enabled during test time
    for Monte Carlo dropout sampling to estimate epistemic uncertainty.

    Defaults assume input of (N, 2, 24, 24) and output of (N, 4, 100, 100).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,  # 2 for prediction + 2 for log variance
        base_channels: int = 64,
        dropout_rate: float = 0.05,
        bilinear: bool = True,
        output_size: Tuple[int, int] = (100, 100),
        input_size: Tuple[int, int] = (24, 24),
    ):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        # Encoder with dropout
        self.inc = DoubleConvBCNN(in_channels, c1, dropout_rate)
        self.down1 = DownBCNN(c1, c2, dropout_rate)
        self.down2 = DownBCNN(c2, c3, dropout_rate)
        self.down3 = DownBCNN(c3, c4, dropout_rate)

        # Decoder with dropout
        self.up1 = UpBCNN(c4, c3, c3, dropout_rate, bilinear)
        self.up2 = UpBCNN(c3, c2, c2, dropout_rate, bilinear)
        self.up3 = UpBCNN(c2, c1, c1, dropout_rate, bilinear)

        # Compute extra ups to exceed target size
        cur_h, cur_w = self.input_size
        tgt_h, tgt_w = self.output_size
        extra = 0
        while (cur_h * 2) <= tgt_h and (cur_w * 2) <= tgt_w:
            cur_h *= 2
            cur_w *= 2
            extra += 1

        # Extra upsampling blocks without skips (keep feature width at c1)
        self.extra_ups = nn.ModuleList([UpNoSkipBCNN(c1, c1, dropout_rate, bilinear) for _ in range(extra)])

        # Final projection to output channels (4 channels: prediction + log variance)
        self.outc = OutConv(c1, out_channels)

        self._will_shrink = (cur_h > tgt_h) and (cur_w > tgt_w)

    def forward(self, x: torch.Tensor, *, channels_last: bool = False) -> torch.Tensor:
        if channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Extra upsampling
        for up in self.extra_ups:
            x = up(x)

        # Project to output channels (4: prediction + log variance)
        x = self.outc(x)

        # Resize to target output size
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        return x


def channels_last_to_first(x: torch.Tensor) -> torch.Tensor:
    """Helper: (N, H, W, C) -> (N, C, H, W)."""
    if x.dim() != 4:
        raise ValueError("Expected 4D tensor (N, H, W, C)")
    return x.permute(0, 3, 1, 2).contiguous()


if __name__ == "__main__":
    # Minimal shape check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test original UNet
    print("Testing UNet (original)...")
    model = UNet().to(device)
    print(model)

    # channels-first input
    x = torch.randn(4, 2, 24, 24, device=device)
    y = model(x)
    assert y.shape == (4, 2, 100, 100), f"Unexpected shape: {y.shape}"

    # channels-last input
    x_cl = torch.randn(4, 24, 24, 2, device=device)
    y2 = model(x_cl, channels_last=True)
    assert y2.shape == (4, 2, 100, 100), f"Unexpected shape: {y2.shape}"

    # different base width
    model_wide = UNet(base_channels=32).to(device)
    y3 = model_wide(x)
    assert y3.shape == (4, 2, 100, 100), f"Unexpected shape: {y3.shape}"

    print("✓ UNet shape test passed.")
    
    # Test BCNN version
    print("\nTesting UNetBCNN (with dropout)...")
    model_bcnn = UNetBCNN().to(device)
    y_bcnn = model_bcnn(x)
    assert y_bcnn.shape == (4, 4, 100, 100), f"Unexpected shape: {y_bcnn.shape}"
    
    # Test with dropout enabled (MC dropout mode)
    model_bcnn.train()
    y_bcnn_mc1 = model_bcnn(x)
    y_bcnn_mc2 = model_bcnn(x)
    # Results should differ slightly due to dropout
    assert not torch.allclose(y_bcnn_mc1, y_bcnn_mc2), "MC dropout should produce different results"
    
    print("✓ UNetBCNN shape test passed.")
    print("\nAll tests passed!")
