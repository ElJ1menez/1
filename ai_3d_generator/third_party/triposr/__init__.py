# TripoSR (MIT, Tripo AI & Stability AI), vendored from
# https://github.com/VAST-AI-Research/TripoSR at the commit in UPSTREAM_COMMIT.
#
# Modifications (2026, AI 3D Generator add-on):
# - Config is built from dataclasses instead of omegaconf; the model config
#   is embedded in system.py instead of downloaded.
# - The DINO ViT-B/16 image encoder is a minimal re-implementation with the
#   same parameter names as transformers' ViTModel (vit.py), so the official
#   checkpoint loads without the transformers package.
# - Marching cubes uses scikit-image instead of torchmcubes, and meshes are
#   returned as numpy arrays instead of trimesh objects.
# - NeRF rendering, video export, rembg and texture baking were removed
#   (the add-on bakes textures in Blender).
# attention.py, basic_transformer_block.py and transformer_1d.py are unchanged
# apart from one import path.
from .system import TSR, TRIPOSR_CONFIG  # noqa: F401
