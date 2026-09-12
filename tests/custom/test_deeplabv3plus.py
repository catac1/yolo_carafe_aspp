# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for DeepLabV3+ low-level decoder and DeepLabV3PlusSegment26 head."""

import torch

from ultralytics.nn.modules.custom_seg import DeepLabV3PlusProto, DeepLabV3PlusSegment26


def test_deeplabv3plus_proto_shapes():
    """Verify DeepLabV3PlusProto output shapes and stride 4 prototype generation."""
    b = 2
    c_p2 = 64  # P2 stride 4
    c_p3 = 128  # P3 stride 8
    h, w = 640, 640

    proto_module = DeepLabV3PlusProto(c_p2=c_p2, c_high=c_p3, c_mid=256, nm=32)
    p2 = torch.randn(b, c_p2, h // 4, w // 4)
    p3 = torch.randn(b, c_p3, h // 8, w // 8)

    # In eval mode
    proto_module.eval()
    with torch.no_grad():
        out = proto_module(p2, p3, return_semantic=False)
    assert out.shape == (b, 32, h // 4, w // 4), f"Expected {(b, 32, h // 4, w // 4)}, got {out.shape}"


def test_deeplabv3plus_proto_gradient_flow():
    """Verify gradient backpropagation to both P2 and high-level features."""
    proto_module = DeepLabV3PlusProto(c_p2=64, c_high=128, c_mid=128, nm=32)
    p2 = torch.randn(2, 64, 40, 40, requires_grad=True)
    p3 = torch.randn(2, 128, 20, 20, requires_grad=True)

    out, sem = proto_module(p2, p3, return_semantic=True)
    loss = out.mean() + sem.mean()
    loss.backward()

    assert p2.grad is not None and not torch.isnan(p2.grad).any(), "P2 input did not receive valid gradient"
    assert p3.grad is not None and not torch.isnan(p3.grad).any(), "P3 input did not receive valid gradient"

    for name, p in proto_module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"


def test_deeplabv3plus_proto_amp():
    """Verify execution under CUDA/CPU AMP autocast."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto_module = DeepLabV3PlusProto(c_p2=64, c_high=128, c_mid=128, nm=32).to(device)
    p2 = torch.randn(2, 64, 40, 40, device=device, requires_grad=True)
    p3 = torch.randn(2, 128, 20, 20, device=device, requires_grad=True)

    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

    with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
        out = proto_module(p2, p3, return_semantic=False)
        loss = out.mean()

    assert out.shape == (2, 32, 40, 40)
    loss.backward()
    assert p2.grad is not None


def test_deeplabv3plus_segment26_head():
    """Verify DeepLabV3PlusSegment26 initializes and performs forward pass."""
    ch = (64, 128, 256, 512)  # (P2, P3, P4, P5)
    head = DeepLabV3PlusSegment26(nc=80, nm=32, npr=256, reg_max=1, end2end=True, ch=ch)
    head.eval()

    feats = [
        torch.randn(1, 64, 160, 160),  # P2
        torch.randn(1, 128, 80, 80),  # P3
        torch.randn(1, 256, 40, 40),  # P4
        torch.randn(1, 512, 20, 20),  # P5
    ]

    with torch.no_grad():
        outputs = head(feats)

    # Validate output format
    assert isinstance(outputs, (tuple, list)), "Head output should be a tuple/list"


if __name__ == "__main__":
    test_deeplabv3plus_proto_shapes()
    test_deeplabv3plus_proto_gradient_flow()
    test_deeplabv3plus_proto_amp()
    test_deeplabv3plus_segment26_head()
    print("All DeepLabV3+ decoder tests passed successfully!")
