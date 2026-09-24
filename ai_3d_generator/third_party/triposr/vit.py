# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal ViT-B/16 (DINO) image encoder for TripoSR.

Same module and parameter names as transformers' ViTModel (4.35, the version
TripoSR was trained with), so the official TripoSR checkpoint loads into it
unchanged, but without depending on the transformers package. Only inference
with interpolated position embeddings is supported.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# facebook/dino-vitb16 config.json
DINO_VITB16 = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                   intermediate_size=3072, image_size=224, patch_size=16,
                   num_channels=3, layer_norm_eps=1e-12, qkv_bias=True)


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.projection = nn.Conv2d(c.num_channels, c.hidden_size,
                                    kernel_size=c.patch_size, stride=c.patch_size)

    def forward(self, pixel_values):
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_size = c.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(c)
        n = (c.image_size // c.patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, n + 1, c.hidden_size))

    def interpolate_pos_encoding(self, embeddings, height, width):
        # transformers 4.35 behaviour (scale_factor with the +0.1 offset).
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_size + 0.1
        w0 = width // self.patch_size + 0.1
        side = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic", align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values):
        batch, _, height, width = pixel_values.shape
        x = self.patch_embeddings(pixel_values)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, height, width)


class ViTSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.query = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)
        self.key = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)
        self.value = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)

    def forward(self, x):
        b, n, d = x.shape

        def split(t):
            return t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(self.query(x)), split(self.key(x)),
                                             split(self.value(x)))
        return out.transpose(1, 2).reshape(b, n, d)


class ViTSelfOutput(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x):
        return self.dense(x)


class ViTAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = ViTSelfAttention(c)
        self.output = ViTSelfOutput(c)

    def forward(self, x):
        return self.output(self.attention(x))


class ViTIntermediate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.intermediate_size)

    def forward(self, x):
        return F.gelu(self.dense(x))


class ViTOutput(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x, residual):
        return self.dense(x) + residual


class ViTLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = ViTAttention(c)
        self.intermediate = ViTIntermediate(c)
        self.output = ViTOutput(c)
        self.layernorm_before = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, x):
        x = self.attention(self.layernorm_before(x)) + x
        return self.output(self.intermediate(self.layernorm_after(x)), x)


class ViTEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(c) for _ in range(c.num_hidden_layers)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class ViTPooler(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x):
        return torch.tanh(self.dense(x[:, 0]))


class ViTModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = SimpleNamespace(**(config or DINO_VITB16))
        self.embeddings = ViTEmbeddings(c)
        self.encoder = ViTEncoder(c)
        self.layernorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.pooler = ViTPooler(c)

    def forward(self, pixel_values):
        """Returns (last_hidden_state, pooler_output)."""
        x = self.layernorm(self.encoder(self.embeddings(pixel_values)))
        return x, self.pooler(x)
