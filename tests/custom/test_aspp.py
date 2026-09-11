# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for ASPP context module in YOLO26s-Seg."""

import pytest
import torch

from ultralytics.nn.modules.custom_seg import ASPP


def test_aspp_shapes_and_even_odd_spatial():
    """Verify ASPP output shapes and spatial handling across even and odd dimensions."""
    test_cases = [
        # (B, C, H, W)
        (1, 64, 20, 20),
        (2, 128, 40, 40),
        (2, 256, 15, 23),  # odd H and W
        (1, 512, 20, 20),  # typical P5 shape at 640x640 input
    ]

    for b, c, h, w in test_cases:
        aspp = ASPP(c1=c, rates=(3, 6, 9))
        x = torch.randn(b, c, h, w)
        out = aspp(x)
        assert out.shape == (b, c, h, w), f"Expected {(b, c, h, w)}, got {out.shape}"


def test_aspp_gradient_flow():
    """Verify that gradients flow to all parallel branches and the input tensor."""
    b, c, h, w = 2, 128, 20, 20
    aspp = ASPP(c1=c, rates=(3, 6, 9))
    x = torch.randn(b, c, h, w, requires_grad=True)

    out = aspp(x)
    loss = out.mean()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "Input tensor x did not receive gradient"
    assert not torch.isnan(x.grad).any(), "NaN detected in input gradient"

    # Check every parameter in each branch
    for name, p in aspp.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaN"


def test_aspp_amp_fp16():
    """Verify ASPP execution under autocast / AMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aspp = ASPP(c1=128, rates=(3, 6, 9)).to(device)
    x = torch.randn(2, 128, 20, 20, device=device, requires_grad=True)

    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

    with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
        out = aspp(x)
        loss = out.mean()

    assert out.shape == (2, 128, 20, 20)
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_aspp_channel_customization():
    """Verify that c2 (output channels) can be customized if needed."""
    aspp = ASPP(c1=256, c2=128, rates=(3, 6, 9))
    x = torch.randn(1, 256, 16, 16)
    out = aspp(x)
    assert out.shape == (1, 128, 16, 16)


if __name__ == "__main__":
    test_aspp_shapes_and_even_odd_spatial()
    test_aspp_gradient_flow()
    test_aspp_amp_fp16()
    test_aspp_channel_customization()
    print("All ASPP unit tests passed successfully!")
