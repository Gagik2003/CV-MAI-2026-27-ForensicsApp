"""Register course functionality here so it appears in the sidebar."""

from .contrast_stretching import ContrastStretchingTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .split_channel import SplitChannelTool
from .swap_channels import SwapChannelsTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            *(SplitChannelTool(channel) for channel in "RGB"),
            SwapChannelsTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchingTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
