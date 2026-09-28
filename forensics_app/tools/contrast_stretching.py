"""Expand grayscale contrast using configurable percentile limits."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def stretch_contrast(
    image: Image.Image,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> tuple[Image.Image, float, float]:
    """Return an 8-bit grayscale result and its input intensity limits.

    Values below the lower limit become black, values above the upper limit
    become white, and values in between are mapped linearly to 0–255.
    Percentiles 0 and 100 use the image's actual minimum and maximum.
    """

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 ≤ lower < upper ≤ 100.")
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for contrast stretching.")

    grayscale = ImageOps.grayscale(image.convert("RGB"))
    pixels = np.asarray(grayscale, dtype=np.float64)
    lower, upper = np.percentile(pixels, (lower_percentile, upper_percentile))
    if upper <= lower:
        raise ValueError(
            "The selected percentiles have the same intensity, so contrast cannot "
            "be stretched. Try percentiles 0 and 100, or a non-uniform image."
        )

    stretched = (pixels - lower) / (upper - lower) * 255.0
    output = np.rint(np.clip(stretched, 0, 255)).astype(np.uint8)
    return Image.fromarray(output), float(lower), float(upper)


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Enhancement"
    description = "Expand grayscale contrast using percentile limits (default: 5 and 95)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        lower_percentile = simpledialog.askfloat(
            self.title,
            "Lower percentile (0–100):\nUse 0 and 100 for min/max stretching.\n"
            "The output will be an 8-bit grayscale image.",
            parent=parent,
            minvalue=0.0,
            maxvalue=100.0,
            initialvalue=5.0,
        )
        if lower_percentile is None:
            return None

        upper_percentile = simpledialog.askfloat(
            self.title,
            f"Upper percentile (greater than {lower_percentile:g}, up to 100):",
            parent=parent,
            minvalue=lower_percentile,
            maxvalue=100.0,
            initialvalue=max(95.0, (lower_percentile + 100.0) / 2.0),
        )
        if upper_percentile is None:
            return None

        assert document.current is not None
        output, lower, upper = stretch_contrast(
            document.current, lower_percentile, upper_percentile
        )
        return ToolResult(
            image=output,
            message=(
                "Applied grayscale contrast stretching "
                f"(percentiles {lower_percentile:g}–{upper_percentile:g})."
            ),
            details={
                "Operation": "Contrast stretching",
                "Percentiles": f"{lower_percentile:g}–{upper_percentile:g}",
                "Input limits": f"{lower:.2f}–{upper:.2f}",
                "Mapping": "Input limits → 0–255 (clipped)",
                "Result range": f"{output.getextrema()[0]}–{output.getextrema()[1]}",
                "Size": f"{output.width} × {output.height}",
                "Output mode": "L (8-bit grayscale)",
            },
        )
