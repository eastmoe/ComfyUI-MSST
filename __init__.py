import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
plugin_dir_str = str(PLUGIN_DIR)
if plugin_dir_str not in sys.path:
    sys.path.insert(0, plugin_dir_str)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:
    print(f"[Comfy-MSST] Failed to import nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
