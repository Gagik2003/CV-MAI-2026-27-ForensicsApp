"""Show a single colour channel of the working image as a grayscale image."""

from __future__ import annotations

import tkinter as tk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


NAMES = {"R": "Red", "G": "Green", "B": "Blue"}
COLORS = {"R": "#c0392b", "G": "#27ae60", "B": "#2e6fd1"}


class SplitChannelTool(ForensicsTool):
    """One sidebar button per channel; create an instance for each of "R", "G", "B"."""

    category = "Channels"

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.tool_id = f"split_channel_{channel.lower()}"
        self.title = f"{NAMES[channel]} channel"
        self.description = f"Show the {NAMES[channel].lower()} channel as a grayscale image."
        self.button_color = COLORS[channel]

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        output = document.current.convert("RGB").getchannel(self.channel)
        return ToolResult(
            image=output,
            message=f"Showing the {NAMES[self.channel].lower()} channel.",
            details={"Operation": "Split channel", "Channel": self.channel},
        )
