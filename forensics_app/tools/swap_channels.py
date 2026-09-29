"""Swap two colour channels of the working image."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def swap_channels(image: Image.Image, first: str, second: str) -> Image.Image:
    """Return an RGB copy of ``image`` with channels ``first`` and ``second`` swapped, e.g. "R", "B"."""
    pixels = np.array(image.convert("RGB"))  # shape (height, width, 3)
    i, j = "RGB".index(first), "RGB".index(second)
    pixels[..., [i, j]] = pixels[..., [j, i]]
    return Image.fromarray(pixels)


def ask_channels(parent: tk.Misc) -> tuple[str, str] | None:
    """Let the user pick two channels; return ``None`` if the window is closed."""
    dialog = tk.Toplevel(parent)
    dialog.title("Swap channels")
    first = ttk.Combobox(dialog, values=list("RGB"), state="readonly", width=4)
    second = ttk.Combobox(dialog, values=list("RGB"), state="readonly", width=4)
    first.set("R")
    second.set("B")
    first.pack(padx=20, pady=(12, 4))
    second.pack(padx=20, pady=4)

    result = []

    def ok() -> None:
        result.append((first.get(), second.get()))
        dialog.destroy()

    ttk.Button(dialog, text="OK", command=ok).pack(pady=(4, 12))

    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class SwapChannelsTool(ForensicsTool):
    tool_id = "swap_channels"
    title = "Swap channels"
    category = "Channels"
    description = "Swap two colour channels, e.g. red and blue."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        channels = ask_channels(parent)
        if channels is None:
            return None
        assert document.current is not None
        output = swap_channels(document.current, *channels)
        return ToolResult(
            image=output,
            message=f"Swapped channels {channels[0]} and {channels[1]}.",
            details={"Operation": "Swap channels", "Swapped": f"{channels[0]} ↔ {channels[1]}"},
        )
