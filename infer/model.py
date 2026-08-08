"""ResNet3D-50 + 2-D U-Net ink-detection model, rebuilt for local (CUDA-free) use.

The architecture here is a faithful re-implementation of
``modeling_inkdetection.py`` as published with the
``scrollprize/PHerc.1667-iteration-*`` checkpoints, with two deliberate
deviations, both of which are numerically exact:

1. ``PreTrainedModel`` is dropped. The published remote code targets
   ``transformers==4.57.6``; ``transformers>=5`` raises
   ``AttributeError: 'InkDetectionModel' object has no attribute
   'all_tied_weights_keys'`` during ``from_pretrained``. We build plain
   ``nn.Module`` trees and load ``model.safetensors`` directly, so the
   ``transformers`` version becomes irrelevant.

2. ``nn.MaxPool3d((1,3,3),(1,2,2),(0,1,1))`` is replaced by
   ``MaxPool2dOverDepth``, because ``aten::max_pool3d_with_indices`` is not
   implemented for the MPS backend in torch 2.8.0. Because the kernel and
   stride along depth are both 1 and the depth padding is 0, the 3-D pool is
   *by construction* an independent 2-D pool per depth slice. Verified
   bit-exact (max abs diff 0.0) against ``nn.MaxPool3d`` on CPU.

Neither deviation touches a parameter, so the checkpoint loads with zero
missing and zero unexpected keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# MPS-safe replacement for MaxPool3d(kernel=(1,3,3), stride=(1,2,2), pad=(0,1,1))
# --------------------------------------------------------------------------
class MaxPool2dOverDepth(nn.Module):
    """Spatial-only max pool over a 5-D (B, C, D, H, W) tensor.

    Equivalent to ``nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))``. The depth
    axis is folded into the channel axis, which is valid because
    ``nn.MaxPool2d`` acts independently per channel.
    """

    def __init__(self, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5-D (B,C,D,H,W), got {tuple(x.shape)}")
        b, c, d, h, w = x.shape
        y = self.pool(x.reshape(b, c * d, h, w))
        return y.reshape(b, c, d, y.shape[-2], y.shape[-1])


# --------------------------------------------------------------------------
# Vendored ResNet3D-50 (Hara, Kataoka & Satoh, 2018)
# --------------------------------------------------------------------------
def _conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = _conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _ResNet3D(nn.Module):
    """ResNet3D-50 backbone returning the 4 intermediate feature maps."""

    def __init__(
        self,
        n_input_channels: int = 1,
        block_inplanes: Sequence[int] = (64, 128, 256, 512),
        layers: Sequence[int] = (3, 4, 6, 3),
        conv1_t_size: int = 7,
        conv1_t_stride: int = 1,
    ):
        super().__init__()
        self.in_planes = block_inplanes[0]
        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # Deviation 2 (see module docstring): bit-exact stand-in for MaxPool3d.
        self.maxpool = MaxPool2dOverDepth(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_inplanes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block_inplanes[3], layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1x1(self.in_planes, planes * _Bottleneck.expansion, stride),
                nn.BatchNorm3d(planes * _Bottleneck.expansion),
            )
        layers = [_Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# --------------------------------------------------------------------------
# 2-D U-Net decoder
# --------------------------------------------------------------------------
class _Decoder(nn.Module):
    def __init__(self, encoder_dims: Sequence[int], upscale: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        encoder_dims[i] + encoder_dims[i - 1],
                        encoder_dims[i - 1],
                        3,
                        1,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(encoder_dims[i - 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )
        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            feature_maps[i - 1] = self.convs[i - 1](f)
        return self.up(self.logit(feature_maps[0]))


# --------------------------------------------------------------------------
# Top-level model
# --------------------------------------------------------------------------
_LAYERS_MAP = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


class InkDetectionModel(nn.Module):
    """``backbone`` + ``decoder``; parameter names match the published checkpoint."""

    def __init__(
        self,
        in_channels: int = 1,
        backbone_depth: int = 50,
        backbone_channels: Sequence[int] = (256, 512, 1024, 2048),
        decoder_upscale: int = 1,
    ):
        super().__init__()
        if backbone_depth not in _LAYERS_MAP:
            raise ValueError(
                f"unsupported backbone_depth={backbone_depth}; expected one of {sorted(_LAYERS_MAP)}"
            )
        self.backbone = _ResNet3D(
            n_input_channels=in_channels,
            block_inplanes=(64, 128, 256, 512),
            layers=_LAYERS_MAP[backbone_depth],
        )
        self.decoder = _Decoder(encoder_dims=list(backbone_channels), upscale=decoder_upscale)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return raw logits at quarter resolution, shape ``(B, 1, H/4, W/4)``."""
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must be 4-D (B, D, H, W) or 5-D (B, 1, D, H, W); "
                f"got shape {tuple(pixel_values.shape)}"
            )
        feats = self.backbone(pixel_values)
        pooled = [torch.max(f, dim=2)[0] for f in feats]
        return self.decoder(pooled)


def load_model(checkpoint_dir: str | Path) -> tuple[InkDetectionModel, dict]:
    """Build the model from ``config.json`` and load ``model.safetensors``.

    Returns ``(model_in_eval_mode, config_dict)``. Raises if the checkpoint does
    not map onto the architecture with zero missing and zero unexpected keys —
    a silent partial load would invalidate any downstream measurement.
    """
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())

    model = InkDetectionModel(
        in_channels=config["in_channels"],
        backbone_depth=config["backbone_depth"],
        backbone_channels=config["backbone_channels"],
        decoder_upscale=config["decoder_upscale"],
    )

    state = load_file(str(checkpoint_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match architecture: "
            f"{len(missing)} missing key(s) {list(missing)[:5]}, "
            f"{len(unexpected)} unexpected key(s) {list(unexpected)[:5]}"
        )

    model.eval()
    return model, config
