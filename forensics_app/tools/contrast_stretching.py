"""Grayscale contrast stretching and histogram enhancement operations."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

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
    return np.asarray(image.convert("L"))


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


def _histogram(image: Image.Image) -> np.ndarray:
    """Percentage of full-resolution grayscale pixels at each intensity."""
    pixels = np.asarray(image)
    return np.bincount(pixels.ravel(), minlength=256) / pixels.size * 100.0


class _ContrastDialog(simpledialog.Dialog):
    """Preview on a fixed source; only OK returns a result to the document."""

    def __init__(self, parent: tk.Misc, image: Image.Image) -> None:
        self.source = Image.fromarray(_grayscale_pixels(image))
        self.source_histogram = _histogram(self.source)
        self.reference: Image.Image | None = None
        self.reference_path: Path | None = None
        self.reference_histogram: np.ndarray | None = None
        self._preview_result: ToolResult | None = None
        self._preview_after: str | None = None
        self._syncing_sliders = False
        super().__init__(parent, title="Contrast and histogram operations")

    def body(self, master: tk.Misc) -> ttk.Combobox:
        self.selection = tk.StringVar(master, value=OPERATIONS[0])
        picker = ttk.Combobox(
            master, textvariable=self.selection, values=OPERATIONS,
            state="readonly", width=55,
        )
        picker.pack(fill="x", pady=(0, 8))
        self.lower = tk.StringVar(master, value="5")
        self.upper = tk.StringVar(master, value="95")
        self.clip_limit = tk.StringVar(master, value="0.01")
        self._sliders: list[tuple[tk.StringVar, ttk.Scale]] = []

        parameters = ttk.Frame(master, height=90)
        parameters.pack(fill="x")
        parameters.grid_propagate(False)
        parameters.columnconfigure(0, weight=1)
        self.stretch_controls = ttk.Frame(parameters)
        self._add_slider(self.stretch_controls, "Lower percentile", self.lower, 0, 100)
        self._add_slider(self.stretch_controls, "Upper percentile", self.upper, 0, 100)
        ttk.Label(self.stretch_controls, text="0 / 100 = min/max stretching").pack(anchor="w")
        self.clahe_controls = ttk.Frame(parameters)
        self._add_slider(self.clahe_controls, "CLAHE clip limit", self.clip_limit, 0, 1)
        ttk.Label(self.clahe_controls, text="0 < limit ≤ 1; higher values allow stronger contrast.").pack(anchor="w")
        self.match_controls = ttk.Frame(parameters)
        ttk.Button(self.match_controls, text="Choose reference image…", command=self._choose_reference).pack(anchor="w")
        self.reference_label = ttk.Label(self.match_controls, text="No reference selected", wraplength=620)
        self.reference_label.pack(anchor="w", pady=4)

        views = ttk.Frame(master)
        views.pack()
        self.image_canvases = []
        self.histogram_canvases = []
        self._photos: list[ImageTk.PhotoImage | None] = [None, None]
        for column, title in enumerate(("Input (grayscale)", "Preview (grayscale)")):
            ttk.Label(views, text=title).grid(row=0, column=column, pady=4)
            image_canvas = tk.Canvas(views, width=330, height=260, background="white", highlightthickness=0)
            image_canvas.grid(row=1, column=column, padx=4)
            self.image_canvases.append(image_canvas)
            histogram_canvas = tk.Canvas(views, width=330, height=140, background="white", highlightthickness=0)
            histogram_canvas.grid(row=2, column=column, padx=4, pady=4)
            self.histogram_canvases.append(histogram_canvas)

        ttk.Label(master, text="Histogram: % of pixels per intensity, same axes on both sides.\n"
                  "Stretching: red = lower limit, orange = upper. Matching: orange = reference.").pack(anchor="w", pady=4)
        self.status = tk.StringVar(master)
        ttk.Label(master, textvariable=self.status, wraplength=660).pack(anchor="w")
        self._show_image(0, self.source)
        for variable, _ in self._sliders:
            variable.trace_add("write", self._schedule_preview)
        self.selection.trace_add("write", self._operation_changed)
        self._operation_changed()
        return picker

    def _add_slider(self, master: tk.Misc, label: str, variable: tk.StringVar,
                    minimum: float, maximum: float) -> None:
        row = ttk.Frame(master)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=20).pack(side="left")
        slider = ttk.Scale(row, from_=minimum, to=maximum,
                           command=lambda value: self._on_slider(variable, value))
        slider.set(float(variable.get()))
        slider.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Entry(row, textvariable=variable, width=10).pack(side="left")
        self._sliders.append((variable, slider))

    def _on_slider(self, variable: tk.StringVar, value: str) -> None:
        if not self._syncing_sliders:
            variable.set(f"{float(value):.3f}")

    def _operation_changed(self, *_: object) -> None:
        for operation, controls in ((OPERATIONS[0], self.stretch_controls),
                                    (OPERATIONS[2], self.match_controls),
                                    (OPERATIONS[3], self.clahe_controls)):
            controls.grid(row=0, column=0, sticky="ew")
            if self.selection.get() != operation:
                controls.grid_remove()
        self._schedule_preview()

    def _schedule_preview(self, *_: object) -> None:
        # Keep numeric entry and slider synchronized, including fractional values.
        self._syncing_sliders = True
        try:
            for variable, slider in self._sliders:
                try:
                    value = float(variable.get())
                except ValueError:
                    continue  # Allow partial input while typing.
                if float(slider["from"]) <= value <= float(slider["to"]):
                    slider.set(value)
        finally:
            self._syncing_sliders = False
        self._preview_result = None
        self.status.set("Updating preview…")
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        # Wait for a brief pause in slider movement; never process a thumbnail.
        self._preview_after = self.after(250, self._update_preview)

    def _choose_reference(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose a histogram reference image (any dimensions)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")], parent=self,
        )
        if not filename:
            return
        try:
            with Image.open(filename) as image:
                reference = Image.fromarray(_grayscale_pixels(image))
        except (OSError, ValueError) as error:
            messagebox.showerror("Histogram reference", str(error), parent=self)
            return
        self.reference = reference
        self.reference_path = Path(filename)
        self.reference_histogram = _histogram(reference)
        self.reference_label.configure(text=f"{self.reference_path.name} ({reference.width} × {reference.height})")
        self._schedule_preview()

    def _compute_result(self) -> tuple[ToolResult, tuple[float, ...]]:
        operation = self.selection.get()
        details: dict[str, object] = {"Operation": operation}
        limits: tuple[float, ...] = ()
        if operation == OPERATIONS[0]:
            lower_percentile, upper_percentile = float(self.lower.get()), float(self.upper.get())
            output, lower, upper = stretch_contrast(self.source, lower_percentile, upper_percentile)
            limits = (lower, upper)
            details.update({"Percentiles": f"{lower_percentile:g}–{upper_percentile:g}",
                            "Input limits": f"{lower:.2f}–{upper:.2f}",
                            "Mapping": "Input limits → 0–255 (clipped)"})
        elif operation == OPERATIONS[1]:
            output = equalize_histogram(self.source)
        elif operation == OPERATIONS[2]:
            if self.reference is None or self.reference_path is None:
                raise ValueError("Choose a reference image to preview histogram matching.")
            output = match_histogram(self.source, self.reference)
            details["Reference"] = self.reference_path.name
            details["Reference size"] = f"{self.reference.width} × {self.reference.height}"
        elif operation == OPERATIONS[3]:
            clip_limit = float(self.clip_limit.get())
            output = adaptive_equalize_histogram(self.source, clip_limit)
            details["Clip limit"] = clip_limit
            details["Tile size"] = f"{max(output.width // 8, 1)} × {max(output.height // 8, 1)} (automatic)"
        else:
            raise ValueError("Unknown histogram operation.")
        details.update({"Result range": f"{output.getextrema()[0]}–{output.getextrema()[1]}",
                        "Size": f"{output.width} × {output.height}",
                        "Output mode": "L (8-bit grayscale)"})
        return ToolResult(image=output, message=f"Applied {operation.lower()} in grayscale.",
                          details=details), limits

    def _show_image(self, column: int, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((320, 260), Image.Resampling.LANCZOS)
        self._photos[column] = ImageTk.PhotoImage(preview, master=self)
        canvas = self.image_canvases[column]
        canvas.delete("all")
        canvas.create_image(165, 130, image=self._photos[column])

    def _draw_histogram(self, column: int, histogram: np.ndarray, peak: float,
                        limits: tuple[float, ...] = (), reference: np.ndarray | None = None) -> None:
        canvas = self.histogram_canvases[column]
        canvas.delete("all")
        left, right, top, bottom = 45, 315, 20, 110
        points = [left, bottom]
        for intensity, percentage in enumerate(histogram):
            x = left + intensity / 256 * (right - left)
            y = bottom - percentage / peak * (bottom - top)
            points.extend((x, y, x + (right - left) / 256, y))
        points.extend((right, bottom))
        canvas.create_polygon(points, fill="#4477aa", outline="")
        if reference is not None:
            line = []
            for intensity, percentage in enumerate(reference):
                line.extend((left + intensity / 255 * (right - left),
                             bottom - percentage / peak * (bottom - top)))
            canvas.create_line(line, fill="#cc7722", width=2)
        for limit, color in zip(limits, ("#bb3333", "#cc7722")):
            x = left + limit / 255 * (right - left)
            canvas.create_line(x, top, x, bottom, fill=color, dash=(3, 2), width=2)
            canvas.create_text(x, top - 9, text=f"{limit:.1f}", fill=color)
        canvas.create_line(left, top, left, bottom, right, bottom, fill="#555555")
        canvas.create_text(left - 4, top, text=f"{peak:.1f}%", anchor="e")
        canvas.create_text(left - 4, bottom, text="0", anchor="e")
        for intensity in (0, 128, 255):
            canvas.create_text(left + intensity / 255 * (right - left), bottom + 12, text=str(intensity))

    def _update_preview(self) -> bool:
        self._preview_after = None
        try:
            result, limits = self._compute_result()
        except (ValueError, OSError) as error:
            self._preview_result = None
            self.image_canvases[1].delete("all")
            self._photos[1] = None
            self.histogram_canvases[1].delete("all")
            self._draw_histogram(0, self.source_histogram, float(self.source_histogram.max()))
            self.status.set(str(error))
            return False
        assert result.image is not None
        histogram = _histogram(result.image)
        reference = self.reference_histogram if self.selection.get() == OPERATIONS[2] else None
        peak = max(float(self.source_histogram.max()), float(histogram.max()),
                   float(reference.max()) if reference is not None else 0.0)
        self._show_image(1, result.image)
        self._draw_histogram(0, self.source_histogram, peak, limits)
        self._draw_histogram(1, histogram, peak, reference=reference)
        self._preview_result = result
        self.status.set("Preview uses the full image. OK applies it; Cancel leaves the image unchanged.")
        return True

    def validate(self) -> bool:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if self._preview_result is None and not self._update_preview():
            messagebox.showerror("Contrast parameters", self.status.get(), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._preview_result

    def cancel(self, event: tk.Event | None = None) -> None:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        super().cancel(event)


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Enhancement"
    description = "Preview stretching, histogram equalization, matching, or adaptive equalization."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        return _ContrastDialog(parent, document.current).result
