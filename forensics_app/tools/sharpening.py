"""Image sharpening with unsharp masking from skimage.filters."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def _validate_settings(radius: float, amount: float) -> None:
    if not np.isfinite(radius) or not 0 < radius <= 20:
        raise ValueError("Radius must be a number greater than 0 and at most 20 pixels.")
    if not np.isfinite(amount) or not 0 <= amount <= 10:
        raise ValueError("Amount must be a number between 0 and 10.")


VIEWS = ("Sharpened", "Blurred", "Detail")


def unsharp_stages(
    image: Image.Image, radius: float = 2.0, amount: float = 1.5,
) -> dict[str, Image.Image]:
    """Unsharp masking steps: sharpened = original + amount * (original - blurred).

    Radius is the Gaussian sigma; it sets how fine the boosted details are.
    Detail is the signed amount * (original - blurred), shown around mid-gray 128.
    Each RGB band is processed separately; an existing alpha channel is copied unchanged.
    """
    _validate_settings(radius, amount)
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Use an 8-bit grayscale or RGB image for sharpening.")
    if image.width == 0 or image.height == 0:
        raise ValueError("The image must contain pixels.")

    grayscale = image.mode in ("1", "L", "LA")
    pixels = np.asarray(image.convert("L" if grayscale else "RGB"), dtype=float)
    # Same Gaussian as skimage.filters.unsharp_mask, so Sharpened matches it.
    blurred = filters.gaussian(
        pixels, sigma=radius, mode="reflect", preserve_range=True,
        channel_axis=None if grayscale else -1,
    )
    detail = amount * (pixels - blurred)
    stages = {"Sharpened": pixels + detail, "Blurred": blurred, "Detail": 128 + detail}

    alpha = None
    if "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").getchannel("A")
    for view, output in stages.items():
        stages[view] = Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8))
        if alpha is not None:
            stages[view].putalpha(alpha)
    return stages


def sharpen(image: Image.Image, radius: float = 2.0, amount: float = 1.5) -> Image.Image:
    return unsharp_stages(image, radius, amount)["Sharpened"]


class _SharpeningDialog(simpledialog.Dialog):
    """Preview on a fixed source; only OK returns a result to the document."""

    def __init__(self, parent: tk.Misc, image: Image.Image) -> None:
        self.source = image.copy()
        self._preview_result: ToolResult | None = None
        self._preview_after: str | None = None
        self._stages: dict[str, Image.Image] = {}
        super().__init__(parent, title="Image sharpening")

    def body(self, master: tk.Misc) -> ttk.Spinbox:
        self.radius = tk.StringVar(master, value="2.0")
        self.amount = tk.StringVar(master, value="1.5")
        self.view = tk.StringVar(master, value=VIEWS[0])
        spinboxes = []
        for row, (label, variable, low, high) in enumerate((
            ("Radius — blur sigma (0–20 pixels):", self.radius, 0.1, 20),
            ("Amount — detail strength (0–10):", self.amount, 0, 10),
        )):
            ttk.Label(master, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
            spinbox = ttk.Spinbox(
                master, textvariable=variable, from_=low, to=high, increment=0.1, width=10,
            )
            spinbox.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            spinboxes.append(spinbox)
        ttk.Label(
            master, wraplength=660,
            text="Adds back the detail removed by a Gaussian blur. A larger radius boosts coarser "
            "edges; a larger amount makes them stronger. Amount 0 leaves the image unchanged.",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=8)
        views = ttk.Frame(master)
        views.grid(row=3, column=0, columnspan=2)
        self.image_canvases = []
        self._photos: list[ImageTk.PhotoImage | None] = [None, None]
        ttk.Label(views, text="Input").grid(row=0, column=0, pady=4)
        preview_title = ttk.Frame(views)
        preview_title.grid(row=0, column=1, pady=4)
        ttk.Label(preview_title, text="Preview:").grid(row=0, column=0, padx=4)
        ttk.Combobox(
            preview_title, textvariable=self.view, values=VIEWS, state="readonly", width=12,
        ).grid(row=0, column=1)
        for column in range(2):
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
        for variable in (self.radius, self.amount):
            variable.trace_add("write", self._schedule_preview)
        self.view.trace_add("write", self._show_view)
        self._schedule_preview()
        return spinboxes[0]

    def _settings(self) -> tuple[float, float]:
        try:
            radius, amount = float(self.radius.get()), float(self.amount.get())
        except ValueError:
            raise ValueError("Radius and amount must be numbers.") from None
        _validate_settings(radius, amount)
        return radius, amount

    def _schedule_preview(self, *_args: str) -> None:
        self._preview_result = None
        self.status.set("Updating preview…")
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        # Allow a brief pause while typing; sharpen the full image, not a thumbnail.
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
            radius, amount = self._settings()
            self._stages = unsharp_stages(self.source, radius, amount)
        except (ValueError, OSError) as error:
            self._preview_result = None
            self._stages = {}
            self.image_canvases[1].delete("all")
            self._photos[1] = None
            self.status.set(str(error))
            return False
        # OK always applies the sharpened image; the other views only explain it.
        output = self._stages["Sharpened"]
        self._preview_result = ToolResult(
            image=output, message="Applied unsharp masking.",
            details={"Operation": "Unsharp masking", "Radius (pixels)": radius,
                     "Amount": amount, "Output mode": output.mode},
        )
        self._show_view()
        return True

    def _show_view(self, *_args: str) -> None:
        if not self._stages:
            return
        self._show_image(1, self._stages[self.view.get()])
        self.status.set({
            "Sharpened": "Sharpened = original + detail.",
            "Blurred": "Gaussian blur with the chosen radius: the fine detail is removed.",
            "Detail": "Detail = amount × (original − blurred). Mid-gray is no change; "
            "lighter pixels get brighter and darker pixels get darker.",
        }[self.view.get()] + " Preview uses the full image. OK applies the sharpened image; "
            "Cancel leaves the image unchanged.")

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


class SharpeningTool(ForensicsTool):
    tool_id = "sharpening"
    title = "Image sharpening"
    category = "Filtering"
    description = "Preview unsharp masking with configurable radius and amount before applying."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        return _SharpeningDialog(parent, document.current).result
