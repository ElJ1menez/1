# Copyright (c) 2024 Tripo AI & Stability AI. MIT License (see LICENSE).
# Modified 2026: embedded config, dataclass configs, scikit-image marching
# cubes, numpy mesh output, local checkpoint loading only.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from .models import (DINOSingleImageTokenizer, NeRFMLP, Triplane1DTokenizer,
                     TriplaneNeRFRenderer, TriplaneUpsampleNetwork)
from .transformer_1d import Transformer1D
from .utils import BaseModule, scale_tensor

# stabilityai/TripoSR config.yaml
TRIPOSR_CONFIG = {
    "cond_image_size": 512,
    "image_tokenizer": {"pretrained_model_name_or_path": "facebook/dino-vitb16"},
    "tokenizer": {"plane_size": 32, "num_channels": 1024},
    "backbone": {"in_channels": 1024, "num_attention_heads": 16, "attention_head_dim": 64,
                 "num_layers": 16, "cross_attention_dim": 768},
    "post_processor": {"in_channels": 1024, "out_channels": 40},
    "decoder": {"in_channels": 120, "n_neurons": 64, "n_hidden_layers": 9, "activation": "silu"},
    "renderer": {"radius": 0.87, "feature_reduction": "concat", "density_activation": "exp",
                 "density_bias": -1.0, "num_samples_per_ray": 128},
}


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        image_tokenizer: dict
        tokenizer: dict
        backbone: dict
        post_processor: dict
        decoder: dict
        renderer: dict

    cfg: Config

    @classmethod
    def from_checkpoint(cls, weight_path, config=None):
        model = cls(config or TRIPOSR_CONFIG)
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13
            ckpt = torch.load(weight_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # The ViT pooler is unused; tolerate checkpoints saved without it.
        missing = [k for k in missing if ".pooler." not in k]
        if missing or unexpected:
            raise RuntimeError("Checkpoint TripoSR incompatible: faltan %s, sobran %s"
                               % (missing[:5], unexpected[:5]))
        return model

    def configure(self):
        self.image_tokenizer = DINOSingleImageTokenizer(self.cfg.image_tokenizer)
        self.tokenizer = Triplane1DTokenizer(self.cfg.tokenizer)
        self.backbone = Transformer1D(self.cfg.backbone)
        self.post_processor = TriplaneUpsampleNetwork(self.cfg.post_processor)
        self.decoder = NeRFMLP(self.cfg.decoder)
        self.renderer = TriplaneNeRFRenderer(self.cfg.renderer)

    def forward(self, image: np.ndarray, device) -> torch.FloatTensor:
        """image: (H, W, 3) float32 in [0, 1], object on a 0.5 gray background.
        Returns scene codes (triplanes), shape (1, 3, 40, 64, 64)."""
        size = self.cfg.cond_image_size
        rgb = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None]
        rgb = F.interpolate(rgb.permute(0, 3, 1, 2), (size, size), mode="bilinear",
                            align_corners=False, antialias=True)
        rgb_cond = rgb[:, None].to(device)  # B Nv C H W
        batch_size = rgb_cond.shape[0]

        input_image_tokens = self.image_tokenizer(rgb_cond)
        input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
        tokens = self.tokenizer(batch_size)
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)
        return self.post_processor(self.tokenizer.detokenize(tokens))

    def extract_mesh(self, scene_code, resolution=256, threshold=25.0):
        """Marching cubes on the density field of one scene code.

        Returns (vertices (N, 3) float32 in TripoSR space, faces (M, 3) int32,
        colors (N, 3) float32 in [0, 1]).
        """
        from skimage import measure

        radius = self.renderer.cfg.radius
        lin = torch.linspace(0.0, 1.0, resolution)
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(-1, 3)
        with torch.no_grad():
            density = self.renderer.query_triplane(
                self.decoder, scale_tensor(grid, (0, 1), (-radius, radius)), scene_code,
            )["density_act"]
        del grid
        level = (density - threshold).view(resolution, resolution, resolution).numpy()
        if level.max() <= 0 or level.min() >= 0:
            raise RuntimeError("La IA no encontró ninguna superficie: prueba otro umbral o imagen")
        # Level > 0 inside; "ascent" winds faces with normals pointing outwards.
        verts, faces, _normals, _values = measure.marching_cubes(level, 0.0, gradient_direction="ascent")
        verts = scale_tensor(verts / (resolution - 1.0), (0, 1), (-radius, radius)).astype(np.float32)
        with torch.no_grad():
            color = self.renderer.query_triplane(
                self.decoder, torch.from_numpy(verts), scene_code,
            )["color"]
        return verts, faces.astype(np.int32), color.numpy().astype(np.float32)
