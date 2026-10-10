"""Canny edge detection, optionally superimposed on the image."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage
from skimage import feature, measure

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over


# Number setting: (label, default, minimum, maximum). Canny defaults are skimage's for an image scaled to 0–1.
PARAMETERS = {
    "sigma": ("Sigma (smoothing)", 1.0, 0.0, 10.0),
    "low": ("Low threshold", 0.1, 0.0, 1.0),
    "high": ("High threshold", 0.2, 0.0, 1.0),
}
STYLE_SETTINGS = {
    "thickness": ("Thickness (px)", 1, 1, 10),
    "opacity": ("Opacity", 0.5, 0.0, 1.0),
    "brightness": ("Image brightness", 0.4, 0.0, 1.0),
}
INTEGER_SETTINGS = {"thickness"}
COLOURS = {"Red": (1.0, 0.0, 0.0), "Green": (0.0, 1.0, 0.0), "Blue": (0.0, 0.0, 1.0), "White": (1.0, 1.0, 1.0)}
BASES = ("Colour image", "Grayscale")  # what the edges are drawn on; Canny itself always uses grayscale
VIEWS = ("Off (edges only)", "Coloured edges", "Transparent overlay", "Bright edges, dim image", "Contour overlay")
# The settings each view offers, with its starting colour; the numbers start at STYLE_SETTINGS defaults.
VIEW_SETTINGS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    VIEWS[0]: (None, ()),
    VIEWS[1]: ("Red", ("thickness",)),
    VIEWS[2]: ("Red", ("thickness", "opacity")),
    VIEWS[3]: ("White", ("thickness", "brightness")),
    VIEWS[4]: ("Red", ("thickness",)),
}
PREVIEW_SIZE = (420, 315)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing or sliding before running Canny on the full image


def detect_edges(image: Image.Image, sigma: float = 1.0, low: float = 0.1, high: float = 0.2) -> np.ndarray:
    """Return a boolean edge map from ``skimage.feature.canny`` on the grayscale image (scaled to 0–1)."""
    if low > high:
        raise ValueError("The low threshold must not be above the high threshold.")
    return feature.canny(_gray(image), sigma=sigma, low_threshold=low, high_threshold=high)


def show_edges(
    image: Image.Image,
    edges: np.ndarray,
    view: str = VIEWS[0],
    colour: str | None = None,
    thickness: int = 1,
    opacity: float = 0.5,
    brightness: float = 0.4,
    base: str = BASES[0],
) -> Image.Image:
    """Return the edge map alone ("Off") or drawn over the image (``base``: in colour or grayscale).

    ``colour`` defaults to the view's own (white for bright edges, otherwise red); ``thickness`` widens
    the edges (or, for contours, the line); ``opacity`` is the overlay's strength and ``brightness``
    how much of the image remains around bright edges.
    """
    if view == VIEWS[0]:
        return Image.fromarray(edges.astype(np.uint8) * 255)
    if view not in VIEW_SETTINGS:
        raise ValueError(f"Unknown view: {view}")
    colour = colour or VIEW_SETTINGS[view][0]
    if base == BASES[0]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    else:
        rgb = np.repeat(_gray(image)[..., np.newaxis], 3, axis=2)
    if view == "Contour overlay":
        result = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(result)
        fill = tuple(int(255 * c) for c in COLOURS[colour])
        contours = measure.find_contours(edges.astype(np.float64), 0.5) if min(edges.shape) >= 2 else []
        for contour in contours:
            draw.line([(x, y) for y, x in contour], fill=fill, width=thickness, joint="curve")
        return result
    if thickness > 1:
        edges = ndimage.binary_dilation(edges, structure=np.ones((thickness, thickness), dtype=bool))
    if view == "Coloured edges":
        rgb[edges] = COLOURS[colour]
    elif view == "Transparent overlay":  # a black layer with coloured edges, blended over the image
        overlay = np.zeros_like(rgb)
        overlay[edges] = COLOURS[colour]
        rgb = (1 - opacity) * rgb + opacity * overlay
    elif view == "Bright edges, dim image":
        rgb *= brightness
        rgb[edges] = COLOURS[colour]
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def setting_label(view: str, name: str) -> str:
    """How a view setting is named on screen; for contours the thickness is a line width."""
    if name in ("colour", "base"):
        return {"colour": "Colour", "base": "Drawn on"}[name]
    return "Line width (px)" if view == "Contour overlay" and name == "thickness" else STYLE_SETTINGS[name][0]


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def _is_decimal_prefix(text: str) -> bool:
    return re.fullmatch(r"\d*\.?\d*", text) is not None


def _read_number(variable: tk.StringVar, label: str, minimum: float, maximum: float, integer: bool) -> float:
    """The typed value; raises ``ValueError`` with a message for the status line."""
    try:
        number = int(variable.get()) if integer else float(variable.get())
    except ValueError:
        kind = "a whole number" if integer else "a number"
        raise ValueError(f"Enter {kind} for {label.lower()}.") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return number


def ask_edges(
    parent: tk.Misc, image: Image.Image
) -> tuple[dict[str, float], str, dict[str, Any], np.ndarray, Image.Image] | None:
    """Let the user tune Canny and the view while previewing on ``image``; ``None`` if cancelled.

    Returns the Canny parameters, the view, its style settings, the edge map and the image to show.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Edge detection (Canny)")
    dialog.resizable(False, False)
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    controls = ttk.Frame(body)
    controls.grid(row=0, column=0, sticky="n")
    preview_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    preview_side.grid(row=0, column=1, sticky="n")

    ttk.Label(preview_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(preview_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(preview_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w")

    preview_job: str | None = None
    edges_cache: tuple[tuple[float, ...], np.ndarray] | None = None  # Canny is slow; restyling is not
    preview: tuple[tuple[Any, ...], Image.Image] | None = None
    photo: ImageTk.PhotoImage | None = None
    only_decimals = (dialog.register(_is_decimal_prefix), "%P")
    syncing = False

    def add_number(parent: tk.Misc, label: str, default: float, minimum: float, maximum: float,
                   integer: bool = False) -> tk.StringVar:
        """A slider with a box for typing the exact value; either one updates the other."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=label, width=17).pack(side="left")
        variable = tk.StringVar(dialog, value=f"{default:g}")
        slider = ttk.Scale(row, from_=minimum, to=maximum, length=160, value=default)
        slider.pack(side="left", padx=8)
        ttk.Entry(row, textvariable=variable, width=7, justify="center",
                  validate="key", validatecommand=only_decimals).pack(side="left")

        def slider_moved(value: str) -> None:
            if not syncing:
                variable.set(f"{round(float(value))}" if integer else f"{float(value):.2f}")

        def text_changed(*_: object) -> None:
            nonlocal syncing
            try:
                number = float(variable.get())
            except ValueError:
                number = None
            if number is not None and minimum <= number <= maximum:
                syncing = True
                slider.set(number)
                syncing = False
            schedule_preview()

        slider.configure(command=slider_moved)
        variable.trace_add("write", text_changed)
        return variable

    ttk.Label(controls, text="Canny parameters").pack(anchor="w")
    canny_values = {name: add_number(controls, *spec) for name, spec in PARAMETERS.items()}

    def reset() -> None:
        for name, (_, default, _, _) in PARAMETERS.items():
            canny_values[name].set(f"{default:g}")

    ttk.Button(controls, text="Reset to skimage defaults", command=reset).pack(anchor="w", pady=(8, 0))

    ttk.Label(controls, text="Superimpose on the image", padding=(0, 16, 0, 0)).pack(anchor="w")
    view = tk.StringVar(dialog, value=VIEWS[0])
    for option in VIEWS:
        ttk.Radiobutton(controls, text=option, value=option, variable=view).pack(anchor="w", pady=(2, 0))
    base_row = ttk.Frame(controls)
    base_row.pack(fill="x", pady=(8, 0))
    ttk.Label(base_row, text="Draw on", width=17).pack(side="left")
    base = tk.StringVar(dialog, value=BASES[0])
    for option in BASES:
        ttk.Radiobutton(base_row, text=option, value=option, variable=base,
                        style="Toolbutton").pack(side="left", padx=(0, 4))
    base.trace_add("write", lambda *_: schedule_preview())

    # Each view keeps its own settings in its own frame; all frames share one grid cell so the
    # window keeps its size, and the selected view's frame is raised on top.
    ttk.Label(controls, text="View settings", padding=(0, 16, 0, 0)).pack(anchor="w")
    settings_area = ttk.Frame(controls)
    settings_area.pack(fill="x")
    settings_frames: dict[str, ttk.Frame] = {}
    view_values: dict[str, dict[str, tk.Variable]] = {}
    for option, (colour, numbers) in VIEW_SETTINGS.items():
        frame = ttk.Frame(settings_area)
        frame.grid(row=0, column=0, sticky="nsew")
        settings_frames[option] = frame
        view_values[option] = {}
        if colour is None:
            ttk.Label(frame, text="None: the result is the black-and-white edge map.",
                      foreground="#666666").pack(anchor="w", pady=(6, 0))
            continue
        colour_row = ttk.Frame(frame)
        colour_row.pack(fill="x", pady=(6, 0))
        ttk.Label(colour_row, text=setting_label(option, "colour"), width=17).pack(side="left")
        chosen = tk.StringVar(dialog, value=colour)
        for name in COLOURS:
            ttk.Radiobutton(colour_row, text=name, value=name, variable=chosen,
                            style="Toolbutton").pack(side="left", padx=(0, 4))
        chosen.trace_add("write", lambda *_: schedule_preview())
        view_values[option]["colour"] = chosen
        for name in numbers:
            _, default, minimum, maximum = STYLE_SETTINGS[name]
            view_values[option][name] = add_number(frame, setting_label(option, name), default, minimum, maximum,
                                                   integer=name in INTEGER_SETTINGS)

    def view_changed(*_: object) -> None:
        settings_frames[view.get()].tkraise()
        schedule_preview()

    view.trace_add("write", view_changed)
    settings_frames[VIEWS[0]].tkraise()

    def canny_parameters() -> dict[str, float]:
        current = {}
        for name, variable in canny_values.items():
            label, _, minimum, maximum = PARAMETERS[name]
            current[name] = _read_number(variable, label, minimum, maximum, integer=False)
        if current["low"] > current["high"]:
            raise ValueError("The low threshold must not be above the high threshold.")
        return current

    def style() -> dict[str, Any]:
        settings: dict[str, Any] = {} if view.get() == VIEWS[0] else {"base": base.get()}
        for name, variable in view_values[view.get()].items():
            if name == "colour":
                settings[name] = variable.get()
            else:
                _, _, minimum, maximum = STYLE_SETTINGS[name]
                settings[name] = _read_number(variable, setting_label(view.get(), name), minimum, maximum,
                                              name in INTEGER_SETTINGS)
        return settings

    def compute() -> tuple[tuple[Any, ...], dict[str, float], dict[str, Any], np.ndarray, Image.Image]:
        nonlocal edges_cache
        current, settings = canny_parameters(), style()
        canny_key = tuple(current.values())
        if edges_cache is None or edges_cache[0] != canny_key:
            edges_cache = (canny_key, detect_edges(image, **current))
        key = (canny_key, view.get(), tuple(settings.items()))
        if preview is not None and preview[0] == key:  # nothing changed since the preview
            return key, current, settings, edges_cache[1], preview[1]
        return key, current, settings, edges_cache[1], show_edges(image, edges_cache[1], view.get(), **settings)

    def schedule_preview(*_: object) -> None:
        nonlocal preview_job
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        status.configure(text="Updating preview…")
        preview_job = dialog.after(PREVIEW_DELAY_MS, update_preview)

    def update_preview() -> None:
        nonlocal preview_job, preview, photo
        preview_job = None
        canvas.delete("all")
        try:
            key, _, _, edges, shown = compute()
        except ValueError as error:
            preview = photo = None
            status.configure(text=str(error))
            return
        preview = (key, shown)
        thumbnail = shown.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text=f"{edges.mean():.2%} of pixels are edges. Apply keeps exactly this result.")

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        try:
            _, current, settings, edges, shown = compute()  # reuses the preview when nothing changed
        except ValueError as error:
            messagebox.showerror("Edge detection", str(error), parent=dialog)
            return
        result.append((current, view.get(), settings, edges, shown))
        close()

    buttons = ttk.Frame(preview_side)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    schedule_preview()
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class EdgeDetectionTool(ForensicsTool):
    tool_id = "edge_detection"
    title = "Edge detection"
    category = "Edge detection"
    description = "Canny edges with adjustable sigma and thresholds, optionally drawn over the image."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_edges(parent, document.current)
        if choice is None:
            return None
        current, view, settings, edges, shown = choice
        return ToolResult(
            image=shown,
            message="Detected edges with Canny.",
            details={
                "Operation": "Canny edge detection",
                "Sigma": f"{current['sigma']:g}",
                "Low threshold": f"{current['low']:g}",
                "High threshold": f"{current['high']:g}",
                "Edge pixels": f"{int(edges.sum())} ({edges.mean():.2%})",
                "Superimposed": view,
                **{setting_label(view, name): f"{value:g}" if isinstance(value, float) else value
                   for name, value in settings.items()},
            },
        )
