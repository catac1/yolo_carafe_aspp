# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom segmentation modules for YOLO26s-Seg experiments including CARAFE and ASPP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["CARAFE"]


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE) upsampling module.

    CARAFE predicts content-aware reassembly kernels for feature upsampling via a lightweight
    kernel generating network, then reassembles features in a local region with the predicted kernels.

    Reference:
        Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures", ICCV 2019.
        https://arxiv.org/abs/1905.02188

    Attributes:
        scale (int): Upsampling ratio (default: 2).
        kernel_size (int): Reassembly kernel size (default: 5).
        encoder_kernel (int): Kernel size for kernel generator convolution (default: 3).
        c_mid (int): Channels after channel compression.
        compress (Conv): 1x1 convolution for channel compression.
        kernel_predictor (nn.Conv2d): Convolution predicting reassembly kernels.
    """

    def __init__(
        self,
        c1: int,
        scale: int = 2,
        kernel_size: int = 5,
        encoder_kernel: int = 3,
        compressed_channels: int = 64,
    ):
        """Initializes the CARAFE module.

        Args:
            c1 (int): Number of input channels.
            scale (int): Upsampling scale factor. Must be positive integer (default: 2).
            kernel_size (int): Size of the reassembly kernel (default: 5).
            encoder_kernel (int): Kernel size for the kernel generating network (default: 3).
            compressed_channels (int): Intermediate channels for the kernel generator (default: 64).
        """
        super().__init__()
        assert scale >= 1, f"Scale must be >= 1, got {scale}"
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        assert encoder_kernel % 2 == 1, f"encoder_kernel must be odd, got {encoder_kernel}"

        self.c1 = c1
        self.scale = scale
        self.kernel_size = kernel_size
        self.encoder_kernel = encoder_kernel
        self.c_mid = min(c1, compressed_channels)

        # 1. Channel compressor
        self.compress = Conv(c1, self.c_mid, k=1)

        # 2. Content-aware kernel generator
        # Output channels = (scale * scale) * (kernel_size * kernel_size)
        out_channels = (scale**2) * (kernel_size**2)
        self.kernel_predictor = nn.Conv2d(
            self.c_mid,
            out_channels,
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2,
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights for the kernel prediction layer."""
        nn.init.normal_(self.kernel_predictor.weight, std=0.001)
        if self.kernel_predictor.bias is not None:
            nn.init.zeros_(self.kernel_predictor.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CARAFE upsampling.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Reassembled upsampled feature map of shape (B, C, H * scale, W * scale).
        """
        b, c, h, w = x.shape
        s = self.scale
        k = self.kernel_size
        pad = k // 2

        # Step 1: Predict reassembly kernels from compressed features
        # compressed: (B, c_mid, H, W)
        compressed = self.compress(x)
        # raw_kernel: (B, s * s * k * k, H, W)
        raw_kernel = self.kernel_predictor(compressed)
        # Reshape to (B, s, s, k * k, H, W) to normalize per subpixel
        kernel = raw_kernel.view(b, s, s, k * k, h, w)
        # Softmax normalization over the reassembly kernel spatial footprint (k * k)
        kernel = F.softmax(kernel, dim=3)  # (B, s, s, k * k, H, W)

        # Step 2: Unfold input feature patches
        # x_unfold: (B, C * k * k, H * W)
        x_unfold = F.unfold(x, kernel_size=k, padding=pad)
        # Reshape to (B, C, 1, 1, k * k, H, W) for broadcasting
        x_unfold = x_unfold.view(b, c, 1, 1, k * k, h, w)

        # Step 3: Content-aware feature reassembly
        # kernel: (B, 1, s, s, k * k, H, W)
        kernel = kernel.unsqueeze(1)
        # Reassembled subpixels: (B, C, s, s, H, W)
        out = (x_unfold * kernel).sum(dim=4)

        # Step 4: Spatial rearrangement to (B, C, H * s, W * s)
        # Permute dimensions: (B, C, H, s, W, s)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        # View as (B, C, H * s, W * s)
        return out.view(b, c, h * s, w * s)
