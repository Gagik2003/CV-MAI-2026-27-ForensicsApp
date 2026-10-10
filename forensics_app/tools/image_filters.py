"""Basic spatial filters from skimage.filters."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk
from skimage import filters
from skimage.filters import rank

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


# Rank filters keep median and mean fast for large neighborhoods and treat
# image borders the same way: only neighbors inside the image are used.
NEIGHBORHOOD_FILTERS = {
    "Median filter": rank.median,
    "Mean filter": rank.mean,
}
EDGE_FILTERS = {
    "Sobel edges": filters.sobel,
    "Prewitt edges": filters.prewitt,
    "Roberts edges": filters.roberts,
}
FILTERS = ("Gaussian blur", *NEIGHBORHOOD_FILTERS, *EDGE_FILTERS)
SHAPES = ("Square", "Disk", "Cross")
EXPLANATIONS = {
    "Gaussian blur": "Weighted smoothing. A sigma of 0 skips that direction. Separating copies the shared sigma; re-linking uses X for both axes.",
    "Median filter": "Local median over the selected neighborhood shape, useful for isolated bright/dark noise.",
    "Mean filter": "Local average over the selected neighborhood shape. Larger sizes produce more smoothing.",
    "Sobel edges": "Grayscale edge magnitude: brighter pixels indicate stronger intensity changes. No parameters.",
    "Prewitt edges": "Grayscale edge magnitude using the Prewitt operator. No parameters.",
    "Roberts edges": "Grayscale edge magnitude using small diagonal difference kernels. No parameters.",
}


def _validate_settings(
    operation: str, sigma: float | tuple[float, float], size: int, shape: str
) -> None:
    if operation not in FILTERS:
        raise ValueError("Choose one of the available filters.")
    if operation == "Gaussian blur":
        sigmas = np.asarray(sigma, dtype=float)
        if sigmas.shape not in ((), (2,)) or not np.all(
            np.isfinite(sigmas) & (sigmas >= 0) & (sigmas <= 20)
        ):
            raise ValueError("Each sigma must be a finite number between 0 and 20 pixels.")
    if operation in NEIGHBORHOOD_FILTERS:
        if not isinstance(size, int) or not 1 <= size <= 31 or size % 2 == 0:
            raise ValueError("Neighborhood size must be an odd integer between 1 and 31.")
        if shape not in SHAPES:
            raise ValueError("Choose Square, Disk, or Cross for the neighborhood shape.")


def _footprint(size: int, shape: str) -> np.ndarray:
    """Choose neighbors within the same odd-sized bounding square."""
    if shape == "Square":
        return np.ones((size, size), dtype=bool)
    offsets = np.arange(size) - size // 2
    if shape == "Disk":
        return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= (size // 2) ** 2
    return (offsets[:, None] == 0) | (offsets[None, :] == 0)


def apply_filter(
    image: Image.Image, operation: str, sigma: float | tuple[float, float] = 1.0,
    size: int = 3, shape: str = "Square",
) -> Image.Image:
    """Filter spatial dimensions without mixing RGB bands or changing image size.

    Smoothing preserves color; edge filters compute a grayscale magnitude.
    Sigma is shared by both spatial axes, or a (sigma_y, sigma_x) pair.
    An existing alpha channel is copied unchanged, not filtered.
    """
    _validate_settings(operation, sigma, size, shape)
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for filtering.")
    if image.width == 0 or image.height == 0:
        raise ValueError("The image must contain pixels.")

    grayscale = image.mode in ("1", "L", "LA") or operation in EDGE_FILTERS
    # Rank filters require a writable array; PIL's np.asarray view is read-only.
    pixels = np.array(image.convert("L" if grayscale else "RGB"))
    if operation == "Gaussian blur":
        output = filters.gaussian(
            pixels, sigma=sigma, preserve_range=True,
            channel_axis=None if grayscale else -1,
        )
    elif operation in NEIGHBORHOOD_FILTERS:
        footprint = _footprint(size, shape)
        filter_function = NEIGHBORHOOD_FILTERS[operation]
        if grayscale:
            output = filter_function(pixels, footprint=footprint)
        else:
            output = np.stack(
                [filter_function(pixels[..., channel], footprint=footprint) for channel in range(3)],
                axis=-1,
            )
    else:
        # skimage scales the uint8 input to 0–1; map its magnitude back to 0–255.
        output = EDGE_FILTERS[operation](pixels) * 255.0

    result = Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8))
    if "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info):
        result.putalpha(image.convert("RGBA").getchannel("A"))
    return result


class _FilterDialog(simpledialog.Dialog):
    """Preview on a fixed source; only OK returns a result to the document."""

    def __init__(self, parent: tk.Misc, image: Image.Image) -> None:
        self.source = image.copy()
        self._preview_result: ToolResult | None = None
        self._preview_after: str | None = None
        super().__init__(parent, title="Image filters")

    def body(self, master: tk.Misc) -> ttk.Combobox:
        self.selection = tk.StringVar(master, value=FILTERS[0])
        self.sigma = tk.StringVar(master, value="1.0")
        self.sigma_x = tk.StringVar(master, value="1.0")
        self.sigma_y = tk.StringVar(master, value="1.0")
        self.linked_sigmas = tk.BooleanVar(master, value=True)
        self.size = tk.StringVar(master, value="3")
        self.shape = tk.StringVar(master, value=SHAPES[0])
        picker = ttk.Combobox(
            master, textvariable=self.selection, values=FILTERS, state="readonly", width=30,
        )
        picker.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.gaussian_controls = ttk.Frame(master)
        self.gaussian_controls.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            self.gaussian_controls, text="Use the same sigma for X and Y",
            variable=self.linked_sigmas, command=self._sigma_mode_changed,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.shared_sigma_controls = ttk.Frame(self.gaussian_controls)
        self.separate_sigma_controls = ttk.Frame(self.gaussian_controls)
        for controls, fields in (
            (self.shared_sigma_controls, (("Sigma X = Y (0–20 pixels):", self.sigma),)),
            (self.separate_sigma_controls, (
                ("Sigma X — horizontal (0–20):", self.sigma_x),
                ("Sigma Y — vertical (0–20):", self.sigma_y),
            )),
        ):
            controls.grid(row=1, column=0, sticky="w")
            for row, (label, variable) in enumerate(fields):
                ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
                ttk.Spinbox(
                    controls, textvariable=variable, from_=0, to=20, increment=0.1, width=10,
                ).grid(row=row, column=1, padx=5, pady=5)

        self.neighborhood_controls = ttk.Frame(master)
        self.neighborhood_controls.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(self.neighborhood_controls, text="Neighborhood size (odd, 1–31):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5,
        )
        self.size_entry = ttk.Spinbox(
            self.neighborhood_controls, textvariable=self.size, from_=1, to=31, increment=2, width=10,
        )
        self.size_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.neighborhood_controls, text="Neighborhood shape:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5,
        )
        ttk.Combobox(
            self.neighborhood_controls, textvariable=self.shape, values=SHAPES,
            state="readonly", width=12,
        ).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(self.neighborhood_controls, text="The shape fits inside a size × size square.").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5,
        )
        self.explanation = ttk.Label(master, wraplength=660)
        self.explanation.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=8)
        views = ttk.Frame(master)
        views.grid(row=3, column=0, columnspan=2)
        self.image_canvases = []
        self._photos: list[ImageTk.PhotoImage | None] = [None, None]
        for column, title in enumerate(("Input", "Preview")):
            ttk.Label(views, text=title).grid(row=0, column=column, pady=4)
            canvas = tk.Canvas(
                views, width=330, height=260, background="white", highlightthickness=0,
            )
            canvas.grid(row=1, column=column, padx=4)
            self.image_canvases.append(canvas)
        self.status = tk.StringVar(master)
        ttk.Label(master, textvariable=self.status, wraplength=660).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=8,
        )
        self._show_image(0, self.source)
        for variable in (self.sigma, self.sigma_x, self.sigma_y, self.size, self.shape):
            variable.trace_add("write", self._schedule_preview)
        self.selection.trace_add("write", self._operation_changed)
        self._operation_changed()
        return picker

    def _operation_changed(self, *_args: str) -> None:
        operation = self.selection.get()
        self.gaussian_controls.grid_remove()
        self.neighborhood_controls.grid_remove()
        if operation == "Gaussian blur":
            self.gaussian_controls.grid()
            self.shared_sigma_controls.grid_remove()
            self.separate_sigma_controls.grid_remove()
            if self.linked_sigmas.get():
                self.shared_sigma_controls.grid()
            else:
                self.separate_sigma_controls.grid()
        elif operation in NEIGHBORHOOD_FILTERS:
            self.neighborhood_controls.grid()
        self.explanation.configure(text=EXPLANATIONS[operation])
        self._schedule_preview()

    def _sigma_mode_changed(self) -> None:
        if self.linked_sigmas.get():
            # Re-link using X; separating starts both axes at the shared value.
            self.sigma.set(self.sigma_x.get())
        else:
            self.sigma_x.set(self.sigma.get())
            self.sigma_y.set(self.sigma.get())
        self._operation_changed()

    def _settings(self) -> tuple[str, float | tuple[float, float], int, str]:
        operation = self.selection.get()
        sigma: float | tuple[float, float] = 1.0
        size, shape = 3, SHAPES[0]
        if operation == "Gaussian blur":
            if self.linked_sigmas.get():
                sigma = float(self.sigma.get())
            else:
                # NumPy spatial axes are ordered Y (rows), X (columns).
                sigma = (float(self.sigma_y.get()), float(self.sigma_x.get()))
        elif operation in NEIGHBORHOOD_FILTERS:
            size, shape = int(self.size.get()), self.shape.get()
        _validate_settings(operation, sigma, size, shape)
        return operation, sigma, size, shape

    def _schedule_preview(self, *_args: str) -> None:
        self._preview_result = None
        self.status.set("Updating preview…")
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        # Allow a brief pause while typing; filter the full image, not a thumbnail.
        self._preview_after = self.after(250, self._update_preview)

    def _show_image(self, column: int, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((320, 260), Image.Resampling.LANCZOS)
        self._photos[column] = ImageTk.PhotoImage(preview, master=self)
        canvas = self.image_canvases[column]
        canvas.delete("all")
        canvas.create_image(165, 130, image=self._photos[column])

    def _update_preview(self) -> bool:
        self._preview_after = None
        try:
            operation, sigma, size, shape = self._settings()
            output = apply_filter(self.source, operation, sigma, size, shape)
        except (ValueError, OSError) as error:
            self._preview_result = None
            self.image_canvases[1].delete("all")
            self._photos[1] = None
            self.status.set(str(error))
            return False
        details = {"Operation": operation, "Output mode": output.mode}
        if operation == "Gaussian blur":
            if isinstance(sigma, tuple):
                details.update({"Sigma X (pixels)": sigma[1], "Sigma Y (pixels)": sigma[0]})
            else:
                details["Sigma X = Y (pixels)"] = sigma
        elif operation in NEIGHBORHOOD_FILTERS:
            details.update({"Neighborhood size": f"{size} × {size}", "Neighborhood shape": shape})
        self._preview_result = ToolResult(
            image=output, message=f"Applied {operation}.", details=details,
        )
        self._show_image(1, output)
        self.status.set("Preview uses the full image. OK applies it; Cancel leaves the image unchanged.")
        return True

    def validate(self) -> bool:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if self._preview_result is None and not self._update_preview():
            messagebox.showerror(self.title(), self.status.get(), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._preview_result

    def cancel(self, event: tk.Event | None = None) -> None:
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        super().cancel(event)


class ImageFiltersTool(ForensicsTool):
    tool_id = "image_filters"
    title = "Image filters"
    category = "Filtering"
    description = "Preview Gaussian, median, mean, Sobel, Prewitt, or Roberts filtering before applying."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        return _FilterDialog(parent, document.current).result
