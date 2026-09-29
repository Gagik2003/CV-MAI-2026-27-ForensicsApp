"""Create a shape-preserving binary mask and composite a foreground image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DEFAULT_THRESHOLD = 0
BinaryMask = NDArray[np.bool_]


def make_threshold_mask(
    image: Image.Image, threshold: int, direction: str = "UPPER"
) -> BinaryMask:
    """Return a 2D boolean array, with True marking the foreground.

    The array shape is ``(height, width)`` so each mask value addresses the
    corresponding pixel in another image of the same dimensions.
    UPPER keeps brightness > threshold; LOWER keeps brightness < threshold.
    """

    if not 0 <= threshold <= 255:
        raise ValueError("The threshold must be between 0 and 255.")
    if direction not in ("UPPER", "LOWER"):
        raise ValueError("The direction must be UPPER or LOWER.")

    grayscale = np.asarray(image.convert("L"))
    mask = grayscale > threshold if direction == "UPPER" else grayscale < threshold
    if "A" in image.getbands():
        mask &= np.asarray(image.getchannel("A")) > 0
    return mask


def composite_with_mask(
    foreground: Image.Image,
    background: Image.Image,
    mask: BinaryMask,
) -> Image.Image:
    """Place foreground pixels over the background wherever mask is True."""

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("The mask must be a two-dimensional boolean array.")
    expected_shape = (foreground.height, foreground.width)
    if mask.shape != expected_shape or background.size != foreground.size:
        raise ValueError(
            "The coat, mask, and target image must have the same dimensions."
        )

    foreground_pixels = np.asarray(foreground.convert("RGBA"))
    background_pixels = np.asarray(background.convert("RGBA"))
    output = Image.fromarray(
        np.where(mask[..., None], foreground_pixels, background_pixels)
    )
    if output.getchannel("A").getextrema() == (255, 255):
        return output.convert("RGB")
    return output


class _MaskDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str, source: Image.Image) -> None:
        self.source = source
        super().__init__(parent, title)

    def body(self, master: tk.Misc) -> tk.Widget:
        self.direction = tk.StringVar(master, value="UPPER")
        self.threshold = tk.StringVar(master, value=str(DEFAULT_THRESHOLD))
        ttk.Label(master, text="Keep pixels:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(
            master, textvariable=self.direction, values=("UPPER", "LOWER"),
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(master, text="Brightness threshold (0–255):").grid(
            row=1, column=0, sticky="w", padx=5, pady=5,
        )
        entry = ttk.Spinbox(master, textvariable=self.threshold, from_=0, to=255, width=12)
        entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.slider = tk.Scale(
            master, from_=0, to=255, resolution=1, orient="horizontal",
            showvalue=False, command=self._on_slider,
        )
        self.slider.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(
            master,
            text="UPPER: brightness > threshold (dark background).\n"
                 "LOWER: brightness < threshold (light background).\n"
                 "Pixels equal to the threshold are excluded.",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(master, text="Original image").grid(row=4, column=0, pady=5)
        ttk.Label(master, text="Binary mask").grid(row=4, column=1, pady=5)
        original_preview = self.source.convert("RGB")
        original_preview.thumbnail((300, 300), Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(original_preview, master=master)
        ttk.Label(master, image=self.original_photo).grid(row=5, column=0, padx=5, pady=5)
        self.mask_preview = ttk.Label(master)
        self.mask_preview.grid(row=5, column=1, padx=5, pady=5)
        self.preview_status = ttk.Label(master)
        self.preview_status.grid(row=6, column=0, columnspan=2, pady=5)
        self.threshold.trace_add("write", self._update_preview)
        self.direction.trace_add("write", self._update_preview)
        self._update_preview()
        return entry

    def _on_slider(self, value: str) -> None:
        threshold = str(int(float(value)))
        if self.threshold.get() != threshold:
            self.threshold.set(threshold)

    def _update_preview(self, *_args: str) -> None:
        try:
            threshold = int(self.threshold.get())
            mask = make_threshold_mask(self.source, threshold, self.direction.get())
        except ValueError:
            self.preview_status.configure(text="Enter a whole number between 0 and 255.")
            return
        self.slider.set(threshold)
        preview = Image.fromarray(mask.astype(np.uint8) * 255)
        # Resize only for display, after computing the full-resolution binary mask.
        preview.thumbnail((300, 300), Image.Resampling.NEAREST)
        self.mask_photo = ImageTk.PhotoImage(preview, master=self.mask_preview)
        self.mask_preview.configure(image=self.mask_photo)
        self.preview_status.configure(
            text=f"White = kept ({mask.mean() * 100:.1f}%). Black = excluded."
        )

    def validate(self) -> bool:
        try:
            threshold = int(self.threshold.get())
            if not 0 <= threshold <= 255:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                self.title(), "Enter a whole number between 0 and 255.", parent=self,
            )
            return False
        return True

    def apply(self) -> None:
        self.result = (self.direction.get(), int(self.threshold.get()))


class MaskingTool(ForensicsTool):
    tool_id = "mask_dark_background"
    title = "Apply mask to another image"
    category = "Masking"
    description = "Place the coat or an optional texture over another same-size image using a binary mask."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current
        settings = _MaskDialog(parent, self.title, source).result
        if settings is None:
            return None
        direction, threshold = settings

        target_filename = filedialog.askopenfilename(
            title="Choose the image to apply the mask to",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not target_filename:
            return None

        mask = make_threshold_mask(source, threshold, direction)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            adjustment = "lower" if direction == "UPPER" else "higher"
            raise ValueError(f"No foreground was found. Try a {adjustment} brightness threshold.")

        target_path = Path(target_filename)
        with Image.open(target_path) as target_file:
            target = target_file.copy()
        if target.size != source.size:
            raise ValueError("The original image and target image must have the same dimensions.")

        use_texture = messagebox.askyesnocancel(
            "Coat texture",
            "Replace the original image's pixels with a texture?\n\n"
            "Yes: choose a same-size texture image.\n"
            "No: keep the original image.",
            parent=parent,
        )
        if use_texture is None:
            return None

        foreground = source
        texture_path = None
        if use_texture:
            texture_filename = filedialog.askopenfilename(
                title="Choose a texture with the same dimensions as the original image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                    ("All files", "*.*"),
                ],
                parent=parent,
            )
            if not texture_filename:
                return None
            texture_path = Path(texture_filename)
            with Image.open(texture_path) as texture_file:
                foreground = texture_file.copy()
            if foreground.size != source.size:
                raise ValueError(
                    "The texture must have the same dimensions as the original image "
                    f"({source.width} × {source.height} pixels)."
                )

        # Always use the original image's silhouette, not the texture's brightness.
        output = composite_with_mask(foreground, target, mask)

        foreground_pixels = int(mask.sum())
        total_pixels = mask.size
        foreground_percentage = foreground_pixels / total_pixels * 100
        left, top = int(xs.min()), int(ys.min())
        right, bottom = int(xs.max()) + 1, int(ys.max()) + 1

        return ToolResult(
            image=output,
            document_path=target_path,
            message=(
                f"Applied {texture_path.name} using the coat mask to {target_path.name}."
                if texture_path is not None
                else f"Applied the coat mask to {target_path.name}."
            ),
            details={
                "Operation": "Binary foreground mask",
                "Keep pixels": f"{direction} (brightness {'>' if direction == 'UPPER' else '<'} threshold)",
                "Threshold": threshold,
                "Mask shape": f"{mask.shape[0]} × {mask.shape[1]} (height × width)",
                "Foreground": f"{foreground_percentage:.1f}% of image",
                "Bounding box": f"({left}, {top})–({right}, {bottom})",
                "Target": target_path.name,
                "Texture": texture_path.name if texture_path is not None else "None (original coat)",
            },
        )
