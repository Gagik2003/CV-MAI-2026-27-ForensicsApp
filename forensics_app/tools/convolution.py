"""Convolve the working image with a hand-written square kernel (3 × 3, 5 × 5 or 7 × 7)."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


SIZES = (3, 5, 7)
PRESETS = ("Horizontal", "Vertical")
OUTPUTS = ("Absolute value", "Clip to 0–255")
OUTPUT_HINTS = {
    OUTPUTS[0]: "Negative responses become positive (shows edges of both directions).",
    OUTPUTS[1]: "Negative responses become black.",
}
CHANNELS = ("Mean of channels", "Grayscale")
CHANNEL_HINTS = {
    CHANNELS[0]: "Convolve R, G and B separately and average the results (also finds edges between colours).",
    CHANNELS[1]: "Convert to grayscale (luminance) first, then convolve once.",
}
ARROWS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
MISSING_VALUE = "Fill every kernel cell with an integer."
PREVIEW_SIZE = (320, 240)
PREVIEW_DELAY_MS = 250  # wait for a pause in typing before convolving the full image


def sobel(direction: str, size: int = 3) -> np.ndarray:
    """Return the ``size`` × ``size`` Sobel kernel (the same ones OpenCV uses).

    "Horizontal" responds to changes along x (so it finds vertical edges), "Vertical" along y.
    Each is a binomial smoothing row times a derivative column, e.g. [1 2 1] and [-1 0 1] for 3 × 3.
    """
    smoothing = _binomial(size - 1)
    derivative = np.convolve(_binomial(size - 3), [-1, 0, 1])
    kernel = np.outer(smoothing, derivative)
    return kernel if direction == "Horizontal" else kernel.T


def _binomial(order: int) -> np.ndarray:
    row = np.array([1.0])
    for _ in range(order):
        row = np.convolve(row, [1, 1])
    return row


def convolve(
    image: Image.Image, kernel: np.ndarray, output: str = OUTPUTS[0], channels: str = CHANNELS[0]
) -> Image.Image:
    """Return a grayscale image of ``image`` convolved with ``kernel`` (odd size).

    ``scipy.ndimage.convolve`` flips the kernel, as convolution requires; borders repeat the edge pixels.
    Negative responses become positive with "Absolute value" and black with "Clip to 0–255".
    "Mean of channels" convolves R, G and B separately and averages the three 0–255 results;
    "Grayscale" converts the image to luminance first.
    """
    grayscale = channels == CHANNELS[1] or image.mode in ("1", "L")
    pixels = np.asarray(image.convert("L" if grayscale else "RGB"), dtype=np.float64)
    weights = kernel if pixels.ndim == 2 else kernel[..., np.newaxis]  # never mix the colour channels
    result = ndimage.convolve(pixels, weights, mode="nearest")
    if output == OUTPUTS[0]:
        result = np.abs(result)
    result = np.clip(result, 0, 255)
    if result.ndim == 3:
        result = result.mean(axis=2)
    return Image.fromarray(np.rint(result).astype(np.uint8))


def _configure_styles(widget: tk.Misc) -> None:
    style = ttk.Style(widget)
    style.configure("Preset.TButton", padding=(14, 6))
    style.configure("SelectedPreset.TButton", padding=(14, 6), background="#3b6fb6", foreground="white")
    style.map("SelectedPreset.TButton", background=[("active", "#335f9c")], foreground=[("active", "white")])


def _is_integer_prefix(text: str) -> bool:
    """Accept what a kernel cell may hold while typing: an integer, or "" / "-" on the way to one."""
    return re.fullmatch(r"-?\d*", text) is not None


def _without_leading_zeros(text: str) -> str:
    """"01" -> "1", "00" -> "0", "-05" -> "-5"; "-0" -> "-" since minus zero is not a value."""
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    if digits:
        digits = digits.lstrip("0") or ("" if sign else "0")
    return sign + digits


def _toggle_sign(text: str) -> str:
    """What the minus key does to a cell: flip the sign, wherever the caret is ("0" becomes "-")."""
    if text.startswith("-"):
        return text[1:]
    return "-" if text in ("", "0") else "-" + text


def _on_cell_key(event: tk.Event) -> str | None:
    entry: ttk.Entry = event.widget
    if event.char == "-":
        text = entry.get()
        whole_value_selected = (entry.selection_present() and entry.index("sel.first") == 0
                                and entry.index("sel.last") == len(text))
        entry.delete(0, "end")  # never rejected: "" and "-n" are both valid cell contents
        entry.insert(0, "-" if whole_value_selected else _toggle_sign(text))
        return "break"
    if event.char and event.char.isprintable() and not event.char.isdigit():
        return "break"  # stop it before Tk deletes the selected value to make room for it
    return None


def _center_over(dialog: tk.Toplevel, window: tk.Misc) -> None:
    dialog.update_idletasks()
    x = window.winfo_rootx() + (window.winfo_width() - dialog.winfo_reqwidth()) // 2
    title_bar = window.winfo_rooty() - window.winfo_y()  # "+x+y" places the frame, not the content
    y = window.winfo_rooty() + (window.winfo_height() - dialog.winfo_reqheight()) // 2 - title_bar
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def ask_kernel(parent: tk.Misc, image: Image.Image) -> tuple[np.ndarray, str, str, Image.Image] | None:
    """Let the user pick a size and type a kernel or load a preset while previewing the result on ``image``.

    Returns the kernel, output mode, colour mode and the convolved image, or ``None`` if cancelled.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()  # shown once centred, so it never flashes at the default position
    dialog.title("Convolution kernel")
    dialog.resizable(False, False)
    dialog.transient(parent)
    _configure_styles(dialog)
    body = ttk.Frame(dialog, padding=16)
    body.pack()
    kernel_side = ttk.Frame(body)  # size, presets and grid on the left; options on the right, so a
    kernel_side.grid(row=0, column=0, sticky="n")  # 7 × 7 kernel makes the window wider, not taller
    options_side = ttk.Frame(body, padding=(20, 0, 0, 0))
    options_side.grid(row=0, column=1, sticky="ns")

    ttk.Label(kernel_side, text="Kernel size").pack(anchor="w")
    size = tk.IntVar(dialog, value=SIZES[0])
    sizes = ttk.Frame(kernel_side)
    sizes.pack(anchor="w", pady=(4, 12))

    ttk.Label(kernel_side, text="Presets (Sobel)").pack(anchor="w")
    presets = ttk.Frame(kernel_side)
    presets.pack(fill="x", pady=(4, 12))
    preset_buttons: dict[str, ttk.Button] = {}

    ttk.Label(kernel_side, text="Kernel (integers; arrow keys move between cells)").pack(anchor="w")
    grid = ttk.Frame(kernel_side)
    grid.pack(pady=(4, 12))
    cells: list[list[tk.StringVar]] = []
    entries: list[list[ttk.Entry]] = []
    only_integers = (dialog.register(_is_integer_prefix), "%P")  # %P: the text the edit would produce

    ttk.Label(options_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(options_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(options_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w", pady=(0, 12))
    preview_job: str | None = None
    preview: tuple[tuple[np.ndarray, str, str], Image.Image] | None = None  # (settings, full-size result)
    photo: ImageTk.PhotoImage | None = None

    def kernel_values() -> np.ndarray | None:
        try:
            return np.array([[int(v.get()) for v in row] for row in cells], dtype=np.float64)
        except ValueError:
            return None

    def settings() -> tuple[np.ndarray, str, str] | None:
        kernel = kernel_values()
        return None if kernel is None else (kernel, output.get(), channels.get())

    def highlight_matching_preset() -> None:
        kernel = kernel_values()
        for name, button in preset_buttons.items():
            selected = kernel is not None and np.array_equal(kernel, sobel(name, len(cells)))
            button.configure(style="SelectedPreset.TButton" if selected else "Preset.TButton")

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
        current = settings()
        if current is None:
            preview = photo = None
            status.configure(text=MISSING_VALUE)
            return
        result = convolve(image, *current)
        preview = (current, result)
        thumbnail = result.copy()
        thumbnail.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail, master=dialog)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo)
        status.configure(text="Preview of the full image; Apply keeps exactly this result.")

    def on_cell_change(variable: tk.StringVar | None = None) -> None:
        if variable is not None and (tidy := _without_leading_zeros(variable.get())) != variable.get():
            variable.set(tidy)  # traces are paused inside a trace, so this does not re-enter
        highlight_matching_preset()
        schedule_preview()

    def load(kernel: np.ndarray) -> None:
        for row, variables in enumerate(cells):
            for column, variable in enumerate(variables):
                variable.set(f"{kernel[row, column]:g}")
        focused = dialog.focus_get()
        if isinstance(focused, ttk.Entry):  # setting the values cleared its selection
            focused.select_range(0, "end")

    def focus_cell(row: int, column: int) -> None:
        entries[row][column].focus_set()
        entries[row][column].icursor("end")

    def on_arrow(event: tk.Event, row: int, column: int) -> str | None:
        entry: ttk.Entry = event.widget
        if event.keysym in ("Left", "Right") and not entry.selection_present():
            # Inside a value the caret moves first; at its start/end the next cell is reached.
            at_edge = entry.index("insert") == (0 if event.keysym == "Left" else len(entry.get()))
            if not at_edge:
                return None
        d_row, d_column = ARROWS[event.keysym]
        last = len(cells) - 1
        focus_cell(min(max(row + d_row, 0), last), min(max(column + d_column, 0), last))
        return "break"

    def build_grid(n: int) -> None:
        """Show an ``n`` × ``n`` grid, keeping the old values centred (zeros around, or cropped)."""
        old = [[variable.get() for variable in row] for row in cells]
        offset = (len(old) - n) // 2
        for widget in grid.winfo_children():
            widget.destroy()
        cells.clear()
        entries.clear()
        for row in range(n):
            cells.append([])
            entries.append([])
            for column in range(n):
                r, c = row + offset, column + offset
                value = old[r][c] if 0 <= r < len(old) and 0 <= c < len(old) else "0"
                variable = tk.StringVar(dialog, value=value)
                entry = ttk.Entry(grid, textvariable=variable, width=6, justify="center", font=("TkDefaultFont", 14),
                                  validate="key", validatecommand=only_integers)
                entry.grid(row=row, column=column, padx=3, pady=3, ipady=4)
                for key in ARROWS:
                    entry.bind(f"<{key}>", lambda event, r=row, c=column: on_arrow(event, r, c))
                entry.bind("<FocusIn>", lambda event: event.widget.select_range(0, "end"))  # typing replaces it
                entry.bind("<KeyPress>", _on_cell_key)
                variable.trace_add("write", lambda *_, v=variable: on_cell_change(v))
                cells[row].append(variable)
                entries[row].append(entry)
        on_cell_change()  # new cells start with values, so nothing else fires

    def resize() -> None:
        build_grid(size.get())
        _center_over(dialog, parent.winfo_toplevel())
        focus_cell(0, 0)

    for n in SIZES:
        ttk.Radiobutton(sizes, text=f"{n} × {n}", value=n, variable=size, command=resize,
                        style="Toolbutton").pack(side="left", padx=(0, 6))
    for name in PRESETS:
        button = ttk.Button(presets, text=name, style="Preset.TButton",
                            command=lambda name=name: load(sobel(name, len(cells))))
        button.pack(side="left", expand=True, fill="x", padx=3)
        preset_buttons[name] = button
    build_grid(SIZES[0])

    def add_choice(title: str, options: tuple[str, ...], hints: dict[str, str]) -> tk.StringVar:
        ttk.Label(options_side, text=title).pack(anchor="w")
        choice = tk.StringVar(dialog, value=options[0])
        hint = ttk.Label(options_side, text=hints[options[0]], foreground="#666666", wraplength=PREVIEW_SIZE[0])
        for option in options:
            ttk.Radiobutton(
                options_side, text=option, value=option, variable=choice,
                command=lambda: hint.configure(text=hints[choice.get()]),
            ).pack(anchor="w", pady=(2, 0))
        hint.pack(anchor="w", pady=(2, 12))
        choice.trace_add("write", schedule_preview)
        return choice

    channels = add_choice("Colour images", CHANNELS, CHANNEL_HINTS)
    output = add_choice("Negative values", OUTPUTS, OUTPUT_HINTS)

    result = []

    def close() -> None:
        if preview_job is not None:
            dialog.after_cancel(preview_job)
        dialog.destroy()

    def ok(_event: tk.Event | None = None) -> None:
        current = settings()
        if current is None:
            messagebox.showerror("Convolution", MISSING_VALUE, parent=dialog)
            return
        up_to_date = preview is not None and np.array_equal(preview[0][0], current[0]) and preview[0][1:] == current[1:]
        result.append((*current, preview[1] if up_to_date else convolve(image, *current)))
        close()

    buttons = ttk.Frame(options_side)
    buttons.pack(side="bottom", fill="x")
    ttk.Button(buttons, text="Cancel", command=close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=ok, default="active").pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", ok)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.protocol("WM_DELETE_WINDOW", close)

    load(sobel("Horizontal", SIZES[0]))
    _center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    focus_cell(0, 0)
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Convolution"
    description = "Convolve the image with a hand-written kernel (3 × 3 to 7 × 7), e.g. Sobel edges."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_kernel(parent, document.current)
        if choice is None:
            return None
        kernel, output_mode, channel_mode, output = choice
        rows = {f"Kernel row {i + 1}": "  ".join(f"{value:g}" for value in row) for i, row in enumerate(kernel)}
        return ToolResult(
            image=output,
            message="Applied the convolution kernel.",
            details={
                "Operation": "Convolution",
                "Kernel size": f"{len(kernel)} × {len(kernel)}",
                **rows,
                "Colour images": channel_mode,
                "Output": output_mode,
            },
        )
