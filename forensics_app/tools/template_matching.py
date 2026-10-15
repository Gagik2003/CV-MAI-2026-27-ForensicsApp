"""Normalized cross-correlation image with skimage.feature.match_template."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def _grayscale_pixels(image: Image.Image) -> np.ndarray:
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for template matching.")
    return np.asarray(image.convert("L"), dtype=float) / 255


def correlation_map(image: Image.Image, template: Image.Image) -> np.ndarray:
    """Normalized cross-correlation of the template at every pixel, between -1 and 1.

    pad_input=True gives a map with the image's dimensions, centred on each pixel.
    """
    pixels, template_pixels = _grayscale_pixels(image), _grayscale_pixels(template)
    if template_pixels.shape[0] > pixels.shape[0] or template_pixels.shape[1] > pixels.shape[1]:
        raise ValueError("The template must not be larger than the image.")
    if template_pixels.size == 0:
        raise ValueError("The template must contain pixels.")
    return feature.match_template(pixels, template_pixels, pad_input=True)


def correlation_image(correlation: np.ndarray) -> Image.Image:
    """Stretch the map's minimum to black and its maximum to white, like imshow."""
    low, high = correlation.min(), correlation.max()
    scaled = (correlation - low) / (high - low) if high > low else np.zeros_like(correlation)
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8))


THRESHOLD = 0.7  # fixed for now; on the Set 4 images only the eye's few pixels pass it


def thresholded_image(
    image: Image.Image, correlation: np.ndarray, threshold: float = THRESHOLD,
) -> Image.Image:
    """Dimmed grayscale image with the positions scoring at least the threshold in red."""
    pixels = np.asarray(image.convert("L"), dtype=float) * 0.4
    rgb = np.repeat(pixels[..., None], 3, axis=2)
    rgb[correlation >= threshold] = (255, 0, 0)
    return Image.fromarray(np.rint(rgb).astype(np.uint8))


PANELS = ("Original image", "Template", "Normalized cross-correlation", "Thresholded image")
APPLY_OPTIONS = PANELS[2:]


class _TemplateMatchingDialog(simpledialog.Dialog):
    """Shows the four images of the slide; OK applies the selected result."""

    def __init__(
        self, parent: tk.Misc, image: Image.Image, template: Image.Image, template_name: str,
    ) -> None:
        self.source, self.template, self.template_name = image, template, template_name
        self.correlation = correlation_map(image, template)
        self.results = {
            APPLY_OPTIONS[0]: correlation_image(self.correlation),
            APPLY_OPTIONS[1]: thresholded_image(image, self.correlation),
        }
        super().__init__(parent, title="Normalized cross-correlation")

    def body(self, master: tk.Misc) -> ttk.Combobox:
        self._photos = []
        images = (self.source, self.template, *self.results.values())
        for index, (title, image) in enumerate(zip(PANELS, images)):
            row, column = divmod(index, 2)
            ttk.Label(master, text=title).grid(row=row * 2, column=column, pady=(6, 2))
            canvas = tk.Canvas(master, width=330, height=260, background="white",
                               highlightthickness=0)
            canvas.grid(row=row * 2 + 1, column=column, padx=4)
            self._photos.append(ImageTk.PhotoImage(_fit(image, 320, 250), master=self))
            canvas.create_image(165, 130, image=self._photos[-1])

        row, column = np.unravel_index(np.argmax(self.correlation), self.correlation.shape)
        self.best = int(column), int(row)
        self.matches = int((self.correlation >= THRESHOLD).sum())
        ttk.Label(
            master, wraplength=660,
            text=f"Best match at (x, y) = {self.best} with score "
            f"{self.correlation[row, column]:.3f}. {self.matches} position(s) score at least "
            f"{THRESHOLD} and are shown in red. Scores range from -1 to 1; the correlation "
            "image shows its lowest score as black and its highest as white.",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=8)
        options = ttk.Frame(master)
        options.grid(row=5, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Label(options, text="OK applies:").grid(row=0, column=0, padx=(0, 6))
        self.choice = tk.StringVar(master, value=APPLY_OPTIONS[1])
        combobox = ttk.Combobox(options, textvariable=self.choice, values=APPLY_OPTIONS,
                                state="readonly", width=28)
        combobox.grid(row=0, column=1)
        return combobox

    def apply(self) -> None:
        low, high = self.correlation.min(), self.correlation.max()
        self.result = ToolResult(
            image=self.results[self.choice.get()],
            message=f"Applied the {self.choice.get().lower()} for {self.template_name}.",
            details={
                "Template": self.template_name,
                "Template size": f"{self.template.width} × {self.template.height}",
                "Best match (x, y)": f"{self.best}",
                "Best score": round(float(high), 4),
                "Score range": f"{low:.4f} to {high:.4f}",
                "Threshold": THRESHOLD,
                "Positions at or above threshold": self.matches,
            },
        )


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale for display only; small images such as templates are enlarged pixel by pixel."""
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS
    return image.convert("RGB").resize(size, resample)


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Normalized cross-correlation"
    category = "Template matching"
    description = "Show how well a template matches each position of the image."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Choose the template to look for",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")], parent=parent,
        )
        if not filename:
            return None
        with Image.open(filename) as template_file:
            template = template_file.copy()
        return _TemplateMatchingDialog(
            parent, document.current, template, Path(filename).name,
        ).result
