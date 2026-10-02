"""Plot the intensity histogram of the working image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


COLORS = {"R": "red", "G": "green", "B": "blue"}


def histogram_image(image: Image.Image, channels: str = "RGB") -> Image.Image:
    """Return a plot of the histogram of ``channels`` (one black line for grayscale images).

    The plot keeps the source image in ``info["histogram_source"]`` so other tools can redraw it.
    """
    if image.mode in ("1", "L"):
        lines = [(np.array(image.convert("L")), "black")]
    else:
        pixels = np.array(image.convert("RGB"))
        lines = [(pixels[..., "RGB".index(c)], COLORS[c]) for c in channels]

    figure = Figure(figsize=(6, 4), layout="constrained")  # keeps the axis labels inside the image
    axes = figure.add_subplot()
    for values, color in lines:
        counts, _ = np.histogram(values, bins=256, range=(0, 256))
        axes.plot(counts, color=color)
    axes.set_xlabel("Intensity")
    axes.set_ylabel("Pixel count")

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    plot = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    plot.info["histogram_source"] = image
    return plot


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Histogram"
    description = "Show the intensity histogram of the image (one line per channel)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        output = histogram_image(document.current)
        return ToolResult(
            image=output,
            message="Showing the image histogram.",
            details={"Operation": "Histogram", "Source mode": document.current.mode},
        )
