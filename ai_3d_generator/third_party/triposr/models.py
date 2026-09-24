# Copyright (c) 2024 Tripo AI & Stability AI. MIT License (see LICENSE).
# Modified 2026: merged tsr/models/{tokenizers,network_utils,nerf_renderer}.py;
# the image tokenizer uses the local ViT (vit.py); only query_triplane is kept
# from the NeRF renderer (the add-on extracts meshes, it does not render).
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .utils import BaseModule, chunk_batch, get_activation, scale_tensor
from .vit import ViTModel


class DINOSingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        self.model = ViTModel()
        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        local_features, _global = self.model(rearrange(images, "B N C H W -> (B N) C H W"))
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size)
        if packed:
            local_features = local_features.squeeze(1)
        return local_features


class Triplane1DTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int
        num_channels: int

    cfg: Config

    def configure(self) -> None:
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        return rearrange(
            repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.cfg.plane_size**2 * 3
        assert Ct == self.cfg.num_channels
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.cfg.plane_size,
            Wp=self.cfg.plane_size,
        )


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        out_channels: int

    cfg: Config

    def configure(self) -> None:
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
        )

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        return rearrange(
            self.upsample(rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )


class NeRFMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        n_neurons: int
        n_hidden_layers: int
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        act = nn.SiLU if self.cfg.activation == "silu" else nn.ReLU
        layers = [nn.Linear(self.cfg.in_channels, self.cfg.n_neurons, bias=self.cfg.bias),
                  act(inplace=True)]
        for _ in range(self.cfg.n_hidden_layers - 1):
            layers += [nn.Linear(self.cfg.n_neurons, self.cfg.n_neurons, bias=self.cfg.bias),
                       act(inplace=True)]
        layers += [nn.Linear(self.cfg.n_neurons, 4, bias=self.cfg.bias)]  # density + rgb
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        inp_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        features = self.layers(x).reshape(*inp_shape, -1)
        return {"density": features[..., 0:1], "features": features[..., 1:4]}


class TriplaneNeRFRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float
        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0
        color_activation: str = "sigmoid"
        num_samples_per_ray: int = 128
        randomized: bool = False

    cfg: Config

    def configure(self) -> None:
        assert self.cfg.feature_reduction in ["concat", "mean"]
        self.chunk_size = 0
        self.check = None

    def set_chunk_size(self, chunk_size: int):
        assert chunk_size >= 0, "chunk_size must be a non-negative integer (0 for no chunking)."
        self.chunk_size = chunk_size

    def query_triplane(
        self,
        decoder: torch.nn.Module,
        positions: torch.Tensor,
        triplane: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)

        # positions in (-radius, radius), normalized to (-1, 1) for grid sample
        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))

        def _query_chunk(x):
            x = x.to(triplane.device)
            indices2D: torch.Tensor = torch.stack(
                (x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]),
                dim=-3,
            )
            out: torch.Tensor = F.grid_sample(
                rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
                rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
                align_corners=False,
                mode="bilinear",
            )
            if self.cfg.feature_reduction == "concat":
                out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            else:
                out = out.mean(dim=0)[:, 0].transpose(0, 1)
            net_out = decoder(out)
            # modified: activations applied per chunk, results kept on the CPU
            # so very large grids fit in GPU memory.
            return {
                "density_act": get_activation(self.cfg.density_activation)(
                    net_out["density"] + self.cfg.density_bias).cpu(),
                "color": get_activation(self.cfg.color_activation)(net_out["features"]).cpu(),
            }

        net_out = chunk_batch(_query_chunk, self.chunk_size, positions, self.check)
        return {k: v.view(*input_shape, -1) for k, v in net_out.items()}
