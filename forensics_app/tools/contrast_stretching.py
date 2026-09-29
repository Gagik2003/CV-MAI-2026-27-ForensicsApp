"""Grayscale contrast stretching and histogram enhancement operations."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


OPERATIONS = (
    "Contrast stretching",
    "Histogram equalization",
    "Histogram matching",
    "Adaptive histogram equalization (CLAHE)",
)


def _grayscale_pixels(image: Image.Image) -> np.ndarray:
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for histogram operations.")
    return np.asarray(ImageOps.grayscale(image.convert("RGB")))


def _uint8_image(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(pixels, 0, 255)).astype(np.uint8))


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
    pixels = _grayscale_pixels(image).astype(np.float64)
    lower, upper = np.percentile(pixels, (lower_percentile, upper_percentile))
    if upper <= lower:
        raise ValueError(
            "The selected percentiles have the same intensity, so contrast cannot "
            "be stretched. Try percentiles 0 and 100, or a non-uniform image."
        )

    stretched = (pixels - lower) / (upper - lower) * 255.0
    return _uint8_image(stretched), float(lower), float(upper)


def equalize_histogram(image: Image.Image) -> Image.Image:
    """Equalize the global grayscale histogram; keep uniform images unchanged."""
    from skimage import exposure

    pixels = _grayscale_pixels(image)
    if pixels.min() == pixels.max():
        return Image.fromarray(pixels.copy())
    return _uint8_image(exposure.equalize_hist(pixels) * 255.0)


def match_histogram(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Match a reference's grayscale distribution without changing image size."""
    from skimage import exposure

    pixels = _grayscale_pixels(image)
    reference_pixels = _grayscale_pixels(reference)
    matched = exposure.match_histograms(pixels, reference_pixels, channel_axis=None)
    return _uint8_image(matched)


def adaptive_equalize_histogram(
    image: Image.Image, clip_limit: float = 0.01
) -> Image.Image:
    """Apply CLAHE with automatic tiles of approximately 1/8 of each dimension."""
    from skimage import exposure

    if not 0 < clip_limit <= 1:
        raise ValueError("The CLAHE clip limit must be greater than 0 and at most 1.")
    pixels = _grayscale_pixels(image)
    if pixels.min() == pixels.max():
        return Image.fromarray(pixels.copy())
    return _uint8_image(exposure.equalize_adapthist(pixels, clip_limit=clip_limit) * 255.0)


class _OperationDialog(simpledialog.Dialog):
    """Keep this feature's operation picker out of the shared main window."""

    def body(self, master: tk.Misc) -> ttk.Combobox:
        ttk.Label(master, text="Choose an operation (8-bit grayscale output):").pack(
            anchor="w", pady=(0, 8)
        )
        self.selection = tk.StringVar(master, value=OPERATIONS[0])
        picker = ttk.Combobox(
            master, textvariable=self.selection, values=OPERATIONS,
            state="readonly", width=45,
        )
        picker.pack(fill="x")
        return picker

    def apply(self) -> None:
        self.result = self.selection.get()


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Enhancement"
    description = "Choose stretching, histogram equalization, matching, or adaptive equalization."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        operation = _OperationDialog(parent, title="Contrast and histogram operations").result
        if operation is None:
            return None
        if operation == OPERATIONS[0]:
            return self._run_stretching(parent, document)

        assert document.current is not None
        details: dict[str, object] = {"Operation": operation}
        if operation == OPERATIONS[1]:
            output = equalize_histogram(document.current)
        elif operation == OPERATIONS[2]:
            reference_filename = filedialog.askopenfilename(
                title="Choose a histogram reference image (any dimensions)",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                    ("All files", "*.*"),
                ],
                parent=parent,
            )
            if not reference_filename:
                return None
            reference_path = Path(reference_filename)
            with Image.open(reference_path) as reference:
                output = match_histogram(document.current, reference)
                details["Reference"] = reference_path.name
                details["Reference size"] = f"{reference.width} × {reference.height}"
        elif operation == OPERATIONS[3]:
            clip_limit = simpledialog.askfloat(
                operation,
                "Clip limit (greater than 0, up to 1):\n"
                "Higher values allow stronger local contrast.",
                parent=parent, minvalue=0.0, maxvalue=1.0, initialvalue=0.01,
            )
            if clip_limit is None:
                return None
            output = adaptive_equalize_histogram(document.current, clip_limit)
            details["Clip limit"] = clip_limit
            details["Tile size"] = (
                f"{max(output.width // 8, 1)} × {max(output.height // 8, 1)} (automatic)"
            )
        else:
            raise ValueError("Unknown histogram operation.")

        details.update({
            "Result range": f"{output.getextrema()[0]}–{output.getextrema()[1]}",
            "Size": f"{output.width} × {output.height}",
            "Output mode": "L (8-bit grayscale)",
        })
        return ToolResult(
            image=output,
            message=f"Applied {operation.lower()} in grayscale.",
            details=details,
        )

    def _run_stretching(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
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
