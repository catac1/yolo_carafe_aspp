# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom segmentation modules for YOLO26s-Seg experiments including CARAFE and ASPP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment26

__all__ = ["CARAFE", "ASPP", "DeepLabV3PlusProto", "DeepLabV3PlusSegment26"]


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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module with DeepLabV3+ architecture.

    Features 5 parallel branches:
    1. 1x1 standard convolution
    2. 3x3 dilated convolution with dilation rate r1
    3. 3x3 dilated convolution with dilation rate r2
    4. 3x3 dilated convolution with dilation rate r3
    5. Global average pooling branch with 1x1 conv and bilinear interpolation
    Followed by concatenation of all branches and a 1x1 projection back to the target channel dimension.

    Reference:
        Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018.
        https://arxiv.org/abs/1802.02611
    """

    def __init__(
        self,
        c1: int,
        c2: int = None,
        rates: tuple = (3, 6, 9),
        c_mid: int = None,
    ):
        """Initializes the ASPP module.

        Args:
            c1 (int): Input channel dimension.
            c2 (int, optional): Output channel dimension. Defaults to c1 (channel preservation).
            rates (tuple): Dilation rates for the three parallel atrous convolutions. Default (3, 6, 9).
            c_mid (int, optional): Internal intermediate channel dimension per branch. Defaults to max(c1 // 4, 64).
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2 if c2 is not None else c1
        self.c_mid = c_mid if c_mid is not None else max(self.c2 // 4, 64)
        if isinstance(rates, list):
            rates = tuple(rates)
        self.rates = rates

        # Branch 1: 1x1 Conv
        self.branch1 = Conv(c1, self.c_mid, k=1)

        # Branch 2, 3, 4: 3x3 Dilated Convolutions
        self.branch2 = Conv(c1, self.c_mid, k=3, p=rates[0], d=rates[0])
        self.branch3 = Conv(c1, self.c_mid, k=3, p=rates[1], d=rates[1])
        self.branch4 = Conv(c1, self.c_mid, k=3, p=rates[2], d=rates[2])

        # Branch 5: Global Image Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.branch5 = Conv(c1, self.c_mid, k=1)

        # Output projection
        total_mid_channels = self.c_mid * 5
        self.project = Conv(total_mid_channels, self.c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP parallel branches and projection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Context-enhanced output tensor of shape (B, C2, H, W).
        """
        _, _, h, w = x.shape

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x_pool = self.pool(x)
        if self.training and x.shape[0] == 1:
            # Handle batch size 1 in training mode so BatchNorm has >1 value per channel
            x_pool = F.interpolate(x_pool, size=(h, w), mode="nearest")
            b5 = self.branch5(x_pool)
        else:
            b5 = self.branch5(x_pool)
            b5 = F.interpolate(b5, size=(h, w), mode="bilinear", align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.project(out)


class DeepLabV3PlusProto(nn.Module):
    """DeepLabV3+-style low-level decoder and mask prototype generator.

    Combines:
    1. Low-level feature from P2 (stride 4) projected via a 1x1 conv to low channels (e.g. 48).
    2. High-level feature from P3 (stride 8) upsampled to stride 4 using CARAFE.
    3. Channel concatenation and two 3x3 refinement convolutions.
    4. Mask prototype generation (nm=32 prototypes).
    5. Auxiliary semantic segmentation branch for YOLO26 training loss.
    """

    def __init__(
        self,
        c_p2: int,
        c_high: int,
        c_mid: int = 256,
        nm: int = 32,
        low_proj_channels: int = 48,
        nc: int = 80,
    ):
        """Initializes DeepLabV3PlusProto decoder module."""
        super().__init__()
        self.c_p2 = c_p2
        self.c_high = c_high
        self.c_mid = c_mid
        self.nm = nm

        # 1. Low-level P2 projection
        self.low_proj = Conv(c_p2, low_proj_channels, k=1)

        # 2. CARAFE upsampler: high-level feature (stride 8) to stride 4 (scale 2)
        self.carafe_up = CARAFE(c_high, scale=2, kernel_size=5)

        # 3. Decoder fusion and refinement
        fuse_channels = low_proj_channels + c_high
        self.refine1 = Conv(fuse_channels, c_mid, k=3)
        self.refine2 = Conv(c_mid, c_mid, k=3)

        # 4. Final prototype projection (stride 4, nm=32 masks)
        self.proto_out = Conv(c_mid, nm, k=1)

        # Semantic segmentation auxiliary branch for YOLO26 training
        self.semseg = nn.Sequential(
            Conv(c_mid, c_mid, k=3),
            nn.Conv2d(c_mid, nc, 1),
        )

    def forward(self, p2: torch.Tensor, high_feat: torch.Tensor, return_semantic: bool = True):
        """Forward pass fusing low-level P2 detail and high-level context via CARAFE."""
        p2_low = self.low_proj(p2)
        high_up = self.carafe_up(high_feat)

        if p2_low.shape[2:] != high_up.shape[2:]:
            high_up = F.interpolate(high_up, size=p2_low.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([p2_low, high_up], dim=1)
        refined = self.refine2(self.refine1(fused))
        proto = self.proto_out(refined)

        if self.training and return_semantic:
            semantic = self.semseg(refined)
            return (proto, semantic)
        return proto


class DeepLabV3PlusSegment26(Segment26):
    """YOLO26 Segment head with DeepLabV3+-style low-level decoder and CARAFE prototype refinement.

    Takes [P2, P3, P4, P5] features:
    - P2 (stride 4) is used for low-level boundary refinement in DeepLabV3PlusProto.
    - P3, P4, P5 (strides 8, 16, 32) are passed to the 3 detection and mask-coefficient heads,
      maintaining 100% parameter compatibility with baseline YOLO26s-Seg detection weights.
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize DeepLabV3PlusSegment26 with [P2, P3, P4, P5] features."""
        c_p2 = ch[0]
        det_ch = tuple(ch[1:])
        super().__init__(nc, nm, npr, reg_max, end2end, det_ch)
        self.proto = DeepLabV3PlusProto(c_p2=c_p2, c_high=det_ch[0], c_mid=self.npr, nm=self.nm, nc=nc)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Forward pass: [P2, P3, P4, P5] -> detection on [P3, P4, P5] + DeepLabV3+ protos."""
        p2, det_feats = x[0], x[1:]
        outputs = Detect.forward(self, det_feats)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(p2, det_feats[0])
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


