"""
FoundationStereo adapter for the Depth-Anything-3 (DA3) backbone.

This module provides `FeatureAdapter`, a dense feature extractor that matches the convention used by
FoundationStereo's `core/extractor_da3.py::Feature_da3`:

- Input: `imgs` shaped [B, 2, 3, H, W] (paired stereo views).
- Output: dict with key "out": stacked output [2B, 128, H', W']:
    {"out": Tensor[2B, 128, H', W']}

Key design points / fixes vs the previous implementation:
- No import-time dependency on `depth_anything_3`. DA3 is imported lazily in `__init__`.
- Correct token->dense conversion using DA3 head (DPT) up to `scratch.output_conv1` (128 channels).
- Correct handling of input layout (stacked LR vs paired interleaved).
- Runs on the module's device (so parent `.to(device)` works), while returning output to input device.
"""

from __future__ import annotations

import importlib
import sys
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from stereo.modeling.models.foundationstereo.Utils import freeze_model

def _ensure_depth_anything_3_on_syspath() -> None:
    """
    Prefer an installed `depth_anything_3`. If it's not importable, add the local repo path:
      <workspace_ws>/src/Depth-Anything-3/src
    """
    try:
        importlib.import_module("depth_anything_3")
        return
    except Exception:
        pass

    file_path = Path(__file__).resolve()
    parts = file_path.parts
    if "workspace_ws" not in parts:
        raise FileNotFoundError(f"`workspace_ws` not found in path: {file_path}")

    idx = parts.index("workspace_ws")
    workspace_root = Path(*parts[: idx + 1])  # .../workspace_ws
    da3_src = workspace_root / "src" / "Depth-Anything-3" / "src"
    if not da3_src.exists():
        raise FileNotFoundError(f"Depth-Anything-3 src path not found: {da3_src}")

    da3_src_str = str(da3_src)
    if da3_src_str not in sys.path:
        sys.path.insert(0, da3_src_str)


def _import_depth_anything_3():
    _ensure_depth_anything_3_on_syspath()
    from depth_anything_3.api import DepthAnything3  # type: ignore
    return DepthAnything3

class FeatureAdapter(nn.Module):
    """
    Dense feature adapter around DA3 that returns FoundationStereo-friendly output.

    Args:
        da3_model_dir: DA3 pretrained directory (HF layout).
        gpu: CUDA device index. Use -1 for CPU.
        input_layout:
            - "stacked_lr" (default): input is [L_batch; R_batch]
            - "paired": input is interleaved [L0,R0,L1,R1,...]
            - "image_stacked"/"image_paired": accepted aliases for backward compatibility.
        ref_view_strategy: passed to DA3 backbone.
        process_res/process_res_method: kept for CLI/API compatibility (not used in tensor->tensor forward).
        use_autocast: enable autocast on CUDA.
        allow_pad_to_divisible: pad H/W up to patch_size multiple before feeding DA3.
        trainable: if False (default), freeze DA3 backbone+head.
    """

    def __init__(
        self,
        da3_model_dir: str = "/DATA/disk0/zhaobojun/depthanything_model/DA3MONO-LARGE",
        gpu: int = 0,
        input_layout: str = "stacked_lr",
        ref_view_strategy: str = "saddle_balanced",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        use_autocast: bool = True,
        allow_pad_to_divisible: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        if gpu >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device("cpu")

        self.trainable = bool(trainable)

        DepthAnything3 = _import_depth_anything_3()
        api = DepthAnything3.from_pretrained(da3_model_dir).to(device)
        # Only keep the submodules we actually use to avoid double-registering the same
        # parameters under multiple names (api_da3.* and backbone/head.*).
        self.backbone = api.model.backbone
        self.dpt_head = api.model.head

        # vit mono large patch size 14
        self.patch_size = int(getattr(self.dpt_head, "patch_size", 14))
        self.down_ratio = int(getattr(self.dpt_head, "down_ratio", 1))
        # FoundationStereo and DA3 preprocessing require H,W divisible by both 16 and patch_size.
        self.required_divider = 16
        self.required_lcm = int(math.lcm(int(self.required_divider), int(self.patch_size)))

        self.ref_view_strategy = ref_view_strategy
        self.use_autocast = bool(use_autocast)
        self.allow_pad_to_divisible = bool(allow_pad_to_divisible)
        self.process_res = process_res
        self.process_res_method = process_res_method

        # Normalize input_layout aliases
        if input_layout in ("image_stacked", "stacked", "stacked_lr"):
            self.input_layout = "stacked_lr"
        elif input_layout in ("image_paired", "paired"):
            self.input_layout = "paired"
        else:
            raise ValueError(f"Unknown input_layout={input_layout!r}")
        # fix parameter and not train
        if not self.trainable:
            self.backbone = freeze_model(self.backbone).eval()
            self.dpt_head = freeze_model(self.dpt_head).eval()

    def forward(
        self,
        imgs: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: Optional[list[int]] = None, # depthAnything 
        ref_view_strategy: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        if extrinsics is not None or intrinsics is not None:
            raise NotImplementedError(
                "extrinsics/intrinsics are not supported in this DA3 FeatureAdapter yet."
            )

        if export_feat_layers is None:
            export_feat_layers = []
        if ref_view_strategy is None:
            ref_view_strategy = self.ref_view_strategy

        input_device = imgs.device

        # Run on module's current device (so parent `.to(...)` works)
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = input_device

        if imgs.ndim == 5:
            if imgs.shape[1] != 2 or imgs.shape[2] != 3:
                raise ValueError(
                    f"[FeatureAdapter.forward] Expected imgs as [B,2,3,H,W], got {tuple(imgs.shape)}"
                )
            x_pair = imgs
            H, W = imgs.shape[-2], imgs.shape[-1]
        else:
            raise ValueError(f"Unsupported imgs shape {tuple(imgs.shape)}")

        if (H % self.required_lcm) != 0 or (W % self.required_lcm) != 0:
            if not self.allow_pad_to_divisible:
                raise ValueError(
                    f"Input H,W must be divisible by lcm(16,patch_size)={self.required_lcm} "
                    f"(patch_size={self.patch_size}). Got H={H}, W={W}. "
                    f"Consider resizing like `core/da3_input_processor.InputProcessorV2` does."
                )
            pad_h = (self.required_lcm - (H % self.required_lcm)) % self.required_lcm
            pad_w = (self.required_lcm - (W % self.required_lcm)) % self.required_lcm
            x_pair = torch.nn.functional.pad(x_pair, (0, pad_w, 0, pad_h), mode="replicate")
            H, W = x_pair.shape[-2], x_pair.shape[-1]

        if x_pair.device != model_device:
            x_pair = x_pair.to(model_device, non_blocking=True)

        if x_pair.device.type == "cpu":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        grad_enabled = bool(self.trainable and self.training)
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            with torch.autocast(
                device_type=x_pair.device.type,
                dtype=autocast_dtype,
                enabled=bool(self.use_autocast and x_pair.device.type == "cuda"),
            ):
                feats, _aux_feats = self.backbone(
                    x_pair,
                    cam_token=None,
                    export_feat_layers=export_feat_layers,
                    ref_view_strategy=ref_view_strategy,
                )

        dense_pair = self._tokens_to_dense_features(feats, H=H, W=W)  # [Bpair,2,128,H',W']

        # Back to FoundationStereo convention: stacked [L_batch; R_batch]
        if self.input_layout == "paired":
            # Interleaved output: [L0,R0,L1,R1,...]
            Bpair = dense_pair.shape[0]
            dense = dense_pair.permute(0, 1, 2, 3, 4).contiguous().view(
                Bpair * 2, dense_pair.shape[2], dense_pair.shape[3], dense_pair.shape[4]
            )
        else:
            dense_left = dense_pair[:, 0]
            dense_right = dense_pair[:, 1]
            dense = torch.cat([dense_left, dense_right], dim=0)
        if dense.device != input_device:
            dense = dense.to(input_device, non_blocking=True)
        return {"out": dense}

    def _tokens_to_dense_features(self, feats, *, H: int, W: int) -> torch.Tensor:
        """
        Convert DA3 backbone tokens to dense [Bpair,2,128,H',W'] using DA3 DPT head.
        """
        from depth_anything_3.model.utils.head_utils import custom_interpolate
        #feature need dimension 4
        if len(feats) < 4:
            raise ValueError(f"Expected >=4 feature levels from DA3 backbone, got {len(feats)}")

        tokens0 = feats[0][0]
        if tokens0.ndim != 4:
            raise ValueError(f"Expected tokens shape [Bpair,2,N,C], got {tuple(tokens0.shape)}")
        Bpair, S, N, C = tokens0.shape
        if S != 2:
            raise ValueError(f"Expected S==2 stereo views, got S={S}")

        ph, pw = H // self.patch_size, W // self.patch_size
        if ph * pw != N:
            raise ValueError(
                f"Token length N does not match patch grid: N={N}, ph*pw={ph*pw} "
                f"(H={H}, W={W}, patch_size={self.patch_size})."
            )
        #feature flatten
        feats_flat = [f[0].reshape(Bpair * S, N, C) for f in feats]

        take_indices = list(getattr(self.dpt_head, "intermediate_layer_idx", (0, 1, 2, 3)))
        if len(take_indices) < 4:
            raise ValueError(f"Expected 4 indices in head.intermediate_layer_idx, got {take_indices}")
        if max(take_indices[:4]) >= len(feats_flat):
            raise ValueError(
                f"Head expects feature index {max(take_indices[:4])} but backbone returned only {len(feats_flat)} levels"
            )

        resized_feats = []
        for stage_idx, take_idx in enumerate(take_indices[:4]):
            x = feats_flat[take_idx]  # [BS, N, C]
            x = self.dpt_head.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(Bpair * S, C, ph, pw)  # [BS,C,ph,pw]
            x = self.dpt_head.projects[stage_idx](x)
            if bool(getattr(self.dpt_head, "pos_embed", False)):
                x = self.dpt_head._add_pos_embed(x, W, H)  # noqa: SLF001
            x = self.dpt_head.resize_layers[stage_idx](x)
            resized_feats.append(x)
        # finnest fusion layer
        fused = self.dpt_head._fuse(resized_feats)  # noqa: SLF001
        if isinstance(fused, (tuple, list)):
            fused = fused[0]

        # need resize channel 256 ->128
        if not (hasattr(self.dpt_head, "scratch") and hasattr(self.dpt_head.scratch, "output_conv1")):
            raise AttributeError("DA3 head does not expose scratch.output_conv1; cannot extract 128-ch dense feature")
        fused = self.dpt_head.scratch.output_conv1(fused)
        if fused.shape[1] != 128:
            raise ValueError(f"Expected 128-channel dense feature after output_conv1, got C={fused.shape[1]}")

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if bool(getattr(self.dpt_head, "pos_embed", False)):
            fused = self.dpt_head._add_pos_embed(fused, W, H)  # noqa: SLF001

        fused = fused.view(Bpair, S, fused.shape[1], fused.shape[2], fused.shape[3])
        return fused

