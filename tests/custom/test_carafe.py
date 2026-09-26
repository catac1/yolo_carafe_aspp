# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for CARAFE upsampling module in YOLO26s-Seg."""

import torch

from ultralytics.nn.modules.custom_seg import CARAFE


def test_carafe_shapes():
    """Verify CARAFE output shapes across required configurations."""
    test_cases = [
        # (B, C, H, W, scale, kernel_size, expected_shape)
        (1, 64, 20, 20, 2, 5, (1, 64, 40, 40)),
        (2, 128, 40, 40, 2, 5, (2, 128, 80, 80)),
        (2, 256, 80, 80, 2, 5, (2, 256, 160, 160)),
    ]

    for b, c, h, w, scale, k, expected in test_cases:
        carafe = CARAFE(c1=c, scale=scale, kernel_size=k)
        x = torch.randn(b, c, h, w)
        out = carafe(x)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"


def test_carafe_gradient_flow():
    """Verify that gradients flow cleanly to all learnable parameters and input."""
    b, c, h, w = 2, 64, 20, 20
    carafe = CARAFE(c1=c, scale=2, kernel_size=5)
    x = torch.randn(b, c, h, w, requires_grad=True)

    out = carafe(x)
    loss = out.mean()
    loss.backward()

    # Verify input tensor received gradient
    assert x.grad is not None, "Input tensor x did not receive gradient"
    assert not torch.isnan(x.grad).any(), "NaN detected in input gradient"

    # Verify all module parameters received gradients
    for name, p in carafe.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaN"


def test_carafe_amp_fp16():
    """Verify CARAFE forward and backward under autocast / AMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carafe = CARAFE(c1=64, scale=2, kernel_size=5).to(device)
    x = torch.randn(2, 64, 20, 20, device=device, requires_grad=True)

    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

    with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
        out = carafe(x)
        loss = out.mean()

    assert out.shape == (2, 64, 40, 40)
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_carafe_kernel_normalization():
    """Verify that the predicted kernel weights properly sum to 1 per target pixel."""
    b, c, h, w = 1, 32, 10, 10
    carafe = CARAFE(c1=c, scale=2, kernel_size=5)
    x = torch.ones(b, c, h, w)

    # When all inputs are 1.0, because kernel weights sum to 1.0,
    # non-boundary interior pixels should evaluate to 1.0!
    out = carafe(x)
    # Check interior pixels away from border padding
    interior = out[:, :, 5:15, 5:15]
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-4)


if __name__ == "__main__":
    test_carafe_shapes()
    test_carafe_gradient_flow()
    test_carafe_amp_fp16()
    test_carafe_kernel_normalization()
    print("All CARAFE unit tests passed successfully!")
