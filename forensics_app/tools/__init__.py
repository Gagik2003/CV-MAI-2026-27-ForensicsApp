"""Register course functionality here so it appears in the sidebar."""

from .contrast_stretching import ContrastStretchingTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            MaskingTool(),
            ContrastStretchingTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
