"""Lightweight Bandit core package initializer for Comfy-MSST inference.

The original MSST-WEBUI package initializer imports training utilities, losses,
metrics, and augmentation modules. That pulls in audiomentations,
torch_audiomentations, and asteroid before inference code even needs them.
Comfy-MSST only imports model submodules for checkpoint inference, so keeping
this initializer lightweight avoids unnecessary dependency conflicts with
ComfyUI.
"""
