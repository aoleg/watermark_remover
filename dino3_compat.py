"""
dino3_compat.py - load DINOv3-enhanced YOLOv12 checkpoints with stock ultralytics.

Checkpoints trained with the Sompote/DINOV3-YOLOV12 fork (for example
``yolov12x-dino3-watermark-detection.pt``) pickle two ``DINO3Backbone`` layers under
``ultralytics.nn.modules.block``. Stock ultralytics has no such class, so ``torch.load``
fails with "Can't get attribute 'DINO3Backbone'". This module provides an inference-only
re-implementation of that layer and registers it in the namespace the pickle expects.

The forward pass mirrors the fork line for line, including its token-grid reshaping quirk
(it drops only the CLS token, keeps the four DINOv3 register tokens, and folds the resulting
200 tokens into a 10x20 grid). That behaviour is what the fusion layers were trained on, so
it must not be "fixed" here.

Requirements: ``transformers`` (the checkpoint embeds a ``DINOv3ViTModel``). No Hugging Face
download is needed at inference time; the DINOv3 weights are stored inside the checkpoint.

Usage:
    import dino3_compat
    dino3_compat.register()          # safe no-op for ordinary YOLO checkpoints
    model = YOLO("yolov12x-dino3-watermark-detection.pt")
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_INPUT_SIZE = 224  # the fork always resizes the pseudo-RGB map to 224x224 for DINOv3


def _remap_state_dict(old_sd, new_keys):
    """Translate DINOv3 weight names between transformers versions.

    transformers 5.x moved the ViT body into an inner ``model`` submodule, so a key such as
    ``layer.0.norm1.weight`` became ``model.layer.0.norm1.weight``. Handles both directions.
    """
    new_keys = set(new_keys)
    out = {}
    for k, v in old_sd.items():
        for cand in (k, f"model.{k}", k[len("model."):] if k.startswith("model.") else None):
            if cand is not None and cand in new_keys:
                out[cand] = v
                break
        else:
            raise KeyError(f"no matching parameter for checkpoint weight '{k}'")
    return out


class DINO3Backbone(nn.Module):
    """Inference-only replacement for the fork's DINO3Backbone layer.

    Instances are only ever created by unpickling a checkpoint. The pickled state restores
    ``dino_model`` (a transformers ``DINOv3ViTModel``), ``input_projection``, ``fusion_layer``,
    ``feature_adapter``, ``spatial_projection`` and ``freeze_backbone``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "dino3_compat.DINO3Backbone is an inference shim and is only created by loading a "
            "checkpoint trained with the DINOV3-YOLOV12 fork. Use that fork to build or train new models."
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        self._rebuild_dino_model()

    def _rebuild_dino_model(self):
        """Re-create the transformers ViT from its config and weights.

        The pickled ViT carries the internal attribute layout of whatever transformers version
        trained it. Rebuilding it with the installed version and copying the state dict keeps
        the forward code and the module attributes in sync. Falls back to the pickled module
        if the rebuild fails.
        """
        old = self._modules.get("dino_model")
        if old is None:
            return
        try:
            from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
            from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

            cfg = DINOv3ViTConfig(**old.config.to_dict())
            fresh = DINOv3ViTModel(cfg)
            fresh.load_state_dict(_remap_state_dict(old.state_dict(), fresh.state_dict().keys()), strict=True)
            fresh.to(next(old.parameters()).dtype)
            fresh.eval()
            for p in fresh.parameters():
                p.requires_grad_(False)
            self._modules["dino_model"] = fresh
        except Exception as e:  # depends on the installed transformers version
            warnings.warn(f"dino3_compat: could not rebuild DINOv3 ViT, using pickled module as-is ({e})")

    # ------------------------------------------------------------------ forward pass
    def _tokens_to_map(self, features: torch.Tensor) -> torch.Tensor:
        """Fold ViT tokens into a spatial map exactly the way the fork does."""
        if features.dim() == 2:
            features = features.unsqueeze(0)
        B, _, D = features.shape

        patch_features = features[:, 1:, :]  # fork drops only token 0 (CLS); register tokens stay
        n = patch_features.shape[1]

        patch_h = int(n**0.5)
        patch_w = patch_h
        if patch_h * patch_w != n:
            for h in range(patch_h, 0, -1):
                if n % h == 0:
                    patch_h, patch_w = h, n // h
                    break
            else:
                patch_h = patch_w = int(n**0.5)
                if patch_h * patch_w < n:
                    patch_h += 1
                    patch_w = patch_h

        min_dim = 4
        if patch_h < min_dim or patch_w < min_dim:
            aspect = patch_w / patch_h if patch_h > 0 else 1
            if aspect >= 1:
                patch_h = min_dim
                patch_w = max(min_dim, int(patch_h * aspect))
            else:
                patch_w = min_dim
                patch_h = max(min_dim, int(patch_w / aspect))

        target = patch_h * patch_w
        if target < n:
            patch_features = patch_features[:, :target, :]
        elif target > n:
            last = patch_features[:, -1:, :].expand(-1, target - n, -1)
            patch_features = torch.cat([patch_features, last], dim=1)

        fmap = patch_features.view(B, patch_h, patch_w, D)  # [B, H, W, D]
        fmap = self.feature_adapter(fmap)  # Linear + LayerNorm + GELU on the channel axis
        fmap = fmap.permute(0, 3, 1, 2)  # [B, C, H, W]
        return self.spatial_projection(fmap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.input_projection is None:
            raise RuntimeError("DINO3Backbone projection layers missing; the checkpoint was never trained")

        pseudo_rgb = self.input_projection(x)  # [B, 3, H, W], tanh-bounded
        pseudo_rgb = F.interpolate(
            pseudo_rgb, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False
        )

        with torch.set_grad_enabled(not getattr(self, "freeze_backbone", True)):
            out = self.dino_model(pseudo_rgb)
            if hasattr(out, "last_hidden_state"):
                tokens = out.last_hidden_state
            elif isinstance(out, torch.Tensor):
                tokens = out
            elif isinstance(out, (list, tuple)):
                tokens = out[0]
            else:
                raise ValueError(f"Unsupported DINOv3 output type: {type(out)}")

        dino = self._tokens_to_map(tokens)
        dino = F.interpolate(dino, size=(H, W), mode="bilinear", align_corners=False)
        return self.fusion_layer(torch.cat([x, dino], dim=1))


# Pickles produced by the fork reference this exact qualified name.
DINO3Backbone.__module__ = "ultralytics.nn.modules.block"


def _legacy_aattn_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Area-attention forward for the fork's AAttn layout.

    The fork (and the original YOLOv12 repo) build AAttn with separate ``qk`` and ``v``
    1x1 convolutions and a 5x5 depthwise positional conv applied to ``v`` before attention.
    Stock ultralytics uses one fused ``qkv`` conv and a 7x7 positional conv applied after,
    so its forward cannot run on a fork-pickled instance. Math follows the fork's non-flash
    path, using scaled_dot_product_attention for the softmax(q k^T) v product.
    """
    B, C, H, W = x.shape
    N = H * W

    qk = self.qk(x).flatten(2).transpose(1, 2)  # [B, N, 2C]
    v = self.v(x)  # [B, C, H, W]
    pp = self.pe(v)  # positional term, computed on v before attention
    v = v.flatten(2).transpose(1, 2)  # [B, N, C]

    if self.area > 1:
        qk = qk.reshape(B * self.area, N // self.area, C * 2)
        v = v.reshape(B * self.area, N // self.area, C)
        B, N, _ = qk.shape
    q, k = qk.split([C, C], dim=2)

    h, d = self.num_heads, self.head_dim
    q = q.view(B, N, h, d).transpose(1, 2)  # [B, h, N, d]
    k = k.view(B, N, h, d).transpose(1, 2)
    v = v.view(B, N, h, d).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v)  # [B, h, N, d]
    x = x.transpose(1, 2).reshape(B, N, C)  # channel = head-major, matches the fork's view()

    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, C)
        B, N, _ = x.shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return self.proj(x + pp)


_registered = False


def register() -> None:
    """Make stock ultralytics able to load and run the fork's checkpoints. Idempotent.

    1. Exposes DINO3Backbone where the pickles expect to find it.
    2. Teaches AAttn.forward to recognise the fork's qk/v layout. Stock instances (which have a
       fused ``qkv`` conv) keep the stock forward, so ordinary YOLO checkpoints are unaffected.
    """
    global _registered
    if _registered:
        return
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks

    for ns in (block, modules, tasks):
        if not hasattr(ns, "DINO3Backbone"):
            ns.DINO3Backbone = DINO3Backbone

    aattn = block.AAttn
    if not getattr(aattn, "_dino3_compat_patched", False):
        stock_forward = aattn.forward

        def forward(self, x):
            if hasattr(self, "qkv"):
                return stock_forward(self, x)
            return _legacy_aattn_forward(self, x)

        aattn.forward = forward
        aattn._dino3_compat_patched = True
    _registered = True
