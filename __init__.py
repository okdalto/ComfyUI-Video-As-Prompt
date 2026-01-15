"""ComfyUI-Video-As-Prompt package entry for ComfyUI custom nodes.

Exposes `NODE_CLASS_MAPPINGS` so ComfyUI can discover the nodes in `node.py`.
"""
from .node import NODE_CLASS_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS"]
