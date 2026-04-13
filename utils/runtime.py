from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from models.enet.ENet import ENet
from models.erfnet.erfnet import ERFNet


SUPPORTED_BACKBONES = ("dla34", "erfnet", "enet")


def normalize_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    normalized: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key in model_keys:
            normalized[key] = value
            continue

        # Older DLA checkpoints stored DCNv2-specific tensors that no longer
        # exist now that the upsampling path uses plain Conv2d blocks.
        if key.endswith(".conv.bias") or ".conv.conv_offset_mask." in key:
            continue

        normalized[key] = value

    return normalized


def resolve_device(requested: str = "auto", no_cuda: bool = False) -> str:
    requested = requested.lower()
    if no_cuda and requested == "auto":
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return requested

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this machine.")
        return requested

    if requested == "cpu":
        return requested

    raise ValueError(f"Unsupported device '{requested}'. Use one of: auto, cpu, cuda, mps.")


def move_sample_to_device(sample: Iterable[object], device: torch.device) -> tuple[object, ...]:
    non_blocking = device.type == "cuda"
    output = []
    for item in sample:
        if torch.is_tensor(item):
            output.append(item.to(device=device, non_blocking=non_blocking))
        else:
            output.append(item)
    return tuple(output)


def build_model(backbone: str, heads: dict[str, int], device: torch.device) -> torch.nn.Module:
    backbone = backbone.lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Incorrect model backbone provided: {backbone}")

    if backbone == "dla34":
        from models.dla.pose_dla_dcn import get_pose_net
        return get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    if backbone == "erfnet":
        return ERFNet(heads=heads)

    return ENet(heads=heads)


def load_snapshot(model: torch.nn.Module, snapshot_path: str | Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    state_dict = normalize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)
