"""Convolve the working image with a hand-written kernel of any size up to 15 × 15."""

from __future__ import annotations

import math
import re
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import center_over


SIZES = tuple(range(1, 16))  # rows and columns each
DEFAULT_SIZE = 3
# Cell look by the larger kernel side, so a 15 × 15 grid still fits on screen:
# (largest side, entry width in characters, font size, padding, inner vertical padding)
CELL_STYLES = ((7, 6, 14, 3, 4), (11, 5, 12, 2, 2), (15, 4, 11, 1, 1))
SMALLEST_CELL_FONT = 8  # long values shrink the numbers down to this size before cells get wider
SCREEN_MARGIN = 40
EVEN_SIZE_NOTE = "Even sizes have no centre pixel, so the result shifts by half a pixel."
LISTED_KERNEL_ROWS = 7  # taller kernels are summarised instead of listed row by row in the results
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
    image: Image.Image,
    kernel: np.ndarray,
    output: str = OUTPUTS[0],
    channels: str = CHANNELS[0],
    normalize: bool = False,
) -> Image.Image:
    """Return a grayscale image of ``image`` convolved with ``kernel`` (any size; with an even side
    there is no centre pixel, so the result shifts by half a pixel).

    ``normalize`` divides the kernel by the sum of its weights, so all 1s give the neighbourhood's
    mean (a box blur). A kernel whose weights sum to 0, like Sobel, is divided by the sum of its
    positive weights instead, so a full black-to-white edge gives exactly 255 at any kernel size.

    ``scipy.ndimage.convolve`` flips the kernel, as convolution requires; borders repeat the edge pixels.
    Negative responses become positive with "Absolute value" and black with "Clip to 0–255".
    "Mean of channels" convolves R, G and B separately and averages the three 0–255 results;
    "Grayscale" converts the image to luminance first.
    """
    if normalize and (divisor := normalization_divisor(kernel)) is not None:
        kernel = kernel / divisor
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


def normalization_divisor(kernel: np.ndarray) -> float | None:
    """What Normalize divides by: the sum of the weights, or for zero-sum kernels (edges) the sum of
    the positive weights; ``None`` when every weight is 0."""
    total = kernel.sum()
    if total != 0:
        return float(total)
    positive = kernel[kernel > 0].sum()
    return float(positive) if positive > 0 else None


def _normalization_note(kernel: np.ndarray) -> str:
    divisor = normalization_divisor(kernel)
    if divisor is None:
        return "Every weight is 0, so there is nothing to normalize."
    if kernel.sum() != 0:
        return f"Kernel divided by {divisor:,.0f} (the sum of its weights)."
    return f"Weights sum to 0, so the kernel is divided by its positive weights ({divisor:,.0f})."


def _configure_styles(widget: tk.Misc) -> None:
    style = ttk.Style(widget)
    style.configure("Preset.TButton", padding=(14, 6))
    style.configure("SelectedPreset.TButton", padding=(14, 6), background="#3b6fb6", foreground="white")
    style.map("SelectedPreset.TButton", background=[("active", "#335f9c")], foreground=[("active", "white")])


def _is_kernel_size_prefix(text: str) -> bool:
    """What a size box may hold while typing: nothing yet, or a whole number from 1 to 15."""
    return text == "" or (re.fullmatch(r"[1-9]\d?", text) is not None and int(text) in SIZES)


def _block_non_digit_key(event: tk.Event) -> str | None:
    # Stops the key before Tk deletes the selected text to make room for it.
    if event.char and event.char.isprintable() and not event.char.isdigit():
        return "break"
    return None


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


def ask_kernel(parent: tk.Misc, image: Image.Image) -> tuple[np.ndarray, str, str, bool, Image.Image] | None:
    """Let the user pick a size and type a kernel or load a preset while previewing the result on ``image``.

    Returns the kernel, output mode, colour mode, whether to normalize and the convolved image,
    or ``None`` if cancelled.
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

    ttk.Label(kernel_side, text="Kernel size (rows × columns, 1–15)").pack(anchor="w")
    sizes = ttk.Frame(kernel_side)
    sizes.pack(anchor="w", pady=(4, 12))
    rows_count = tk.StringVar(dialog, value=str(DEFAULT_SIZE))
    columns_count = tk.StringVar(dialog, value=str(DEFAULT_SIZE))

    ttk.Label(kernel_side, text="Presets (Sobel; square, odd size of 3 or more)").pack(anchor="w")
    presets = ttk.Frame(kernel_side)
    presets.pack(fill="x", pady=(4, 12))
    preset_buttons: dict[str, ttk.Button] = {}

    grid = ttk.Frame(kernel_side)  # packed below the presets, once those exist
    cells: list[list[tk.StringVar]] = []
    entries: list[list[ttk.Entry]] = []
    only_integers = (dialog.register(_is_integer_prefix), "%P")  # %P: the text the edit would produce
    cell_font = tkfont.nametofont("TkDefaultFont").copy()  # shared by every cell, so one call resizes all
    cell_look = {"width": CELL_STYLES[0][1], "font": CELL_STYLES[0][2]}  # the grid's normal look

    def fit_values() -> None:
        """Keep every digit visible: shrink the numbers first (down to SMALLEST_CELL_FONT) so the cells
        keep their normal size, and widen the cells only when that is not enough."""
        longest = max(len(variable.get()) for row in cells for variable in row)
        cell_font.configure(size=cell_look["font"])
        cell_pixels = cell_font.measure("0") * cell_look["width"]
        while cell_font.measure("0" * longest) > cell_pixels and cell_font.cget("size") > SMALLEST_CELL_FONT:
            cell_font.configure(size=cell_font.cget("size") - 1)
        width = max(math.ceil(cell_pixels / cell_font.measure("0")), longest)  # entry width counts "0"s
        for entry in (entry for row in entries for entry in row):
            entry.configure(width=width)

    def fit_on_screen() -> None:
        """Shrink the cell font while the window is wider than the screen, then centre the window."""
        dialog.update_idletasks()
        while (dialog.winfo_reqwidth() > dialog.winfo_screenwidth() - SCREEN_MARGIN
               and cell_font.cget("size") > SMALLEST_CELL_FONT):
            cell_font.configure(size=cell_font.cget("size") - 1)
            dialog.update_idletasks()
        center_over(dialog, parent.winfo_toplevel())

    ttk.Label(options_side, text="Preview").pack(anchor="w")
    canvas = tk.Canvas(options_side, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
                       background="#20242b", highlightthickness=0)
    canvas.pack(pady=(4, 2))
    status = ttk.Label(options_side, foreground="#666666", wraplength=PREVIEW_SIZE[0])
    status.pack(anchor="w", pady=(0, 12))
    preview_job: str | None = None
    preview: tuple[tuple[np.ndarray, str, str, bool], Image.Image] | None = None  # (settings, full-size result)
    photo: ImageTk.PhotoImage | None = None

    def kernel_values() -> np.ndarray | None:
        try:
            return np.array([[int(v.get()) for v in row] for row in cells], dtype=np.float64)
        except ValueError:
            return None

    def settings() -> tuple[np.ndarray, str, str, bool] | None:
        kernel = kernel_values()
        return None if kernel is None else (kernel, output.get(), channels.get(), normalize.get())

    def sobel_size() -> int | None:
        """The size Sobel presets can use now: the grid must be square, odd and at least 3 × 3."""
        rows, columns = len(cells), len(cells[0]) if cells else 0
        return rows if rows == columns >= 3 and rows % 2 == 1 else None

    def highlight_matching_preset() -> None:
        kernel, n = kernel_values(), sobel_size()
        for name, button in preset_buttons.items():
            selected = kernel is not None and n is not None and np.array_equal(kernel, sobel(name, n))
            button.configure(style="SelectedPreset.TButton" if selected else "Preset.TButton",
                             state="normal" if n is not None else "disabled")

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
        notes = [_normalization_note(current[0])] if current[3] else []
        if any(side % 2 == 0 for side in current[0].shape):
            notes.append(EVEN_SIZE_NOTE)
        status.configure(text=" ".join([*notes, "Preview of the full image; Apply keeps exactly this result."]))

    def on_cell_change(variable: tk.StringVar | None = None) -> None:
        if variable is not None and (tidy := _without_leading_zeros(variable.get())) != variable.get():
            variable.set(tidy)  # traces are paused inside a trace, so this does not re-enter
        highlight_matching_preset()
        schedule_preview()

    def load(kernel: np.ndarray) -> None:
        for row, variables in enumerate(cells):
            for column, variable in enumerate(variables):
                variable.set(str(int(round(kernel[row, column]))))  # never "1.4e+06" or "-0"
        fit_values()  # e.g. large Sobel presets have up to 8 digits
        fit_on_screen()
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
        focus_cell(min(max(row + d_row, 0), len(cells) - 1), min(max(column + d_column, 0), len(cells[0]) - 1))
        return "break"

    def build_grid(rows: int, columns: int, values: list[list[str]] | None = None) -> None:
        """Show a ``rows`` × ``columns`` grid holding ``values``, or else the old values centred
        (zeros around them when growing, cropped when shrinking)."""
        if values is None:
            old = [[variable.get() for variable in row] for row in cells]
            row_offset = (len(old) - rows) // 2
            column_offset = (len(old[0]) - columns) // 2 if old else 0

            def old_value(r: int, c: int) -> str:
                return old[r][c] if 0 <= r < len(old) and 0 <= c < len(old[r]) else "0"

            values = [[old_value(row + row_offset, column + column_offset) for column in range(columns)]
                      for row in range(rows)]
        width, font_size, padding, inner_padding = next(style[1:] for style in CELL_STYLES
                                                        if max(rows, columns) <= style[0])
        cell_look.update(width=width, font=font_size)
        for widget in grid.winfo_children():
            widget.destroy()
        cells.clear()
        entries.clear()
        for row in range(rows):
            cells.append([])
            entries.append([])
            for column in range(columns):
                variable = tk.StringVar(dialog, value=values[row][column])
                entry = ttk.Entry(grid, textvariable=variable, width=width, justify="center",
                                  font=cell_font, validate="key", validatecommand=only_integers)
                entry.grid(row=row, column=column, padx=padding, pady=padding, ipady=inner_padding)
                for key in ARROWS:
                    entry.bind(f"<{key}>", lambda event, r=row, c=column: on_arrow(event, r, c))
                entry.bind("<FocusIn>", lambda event: event.widget.select_range(0, "end"))  # typing replaces it
                entry.bind("<KeyPress>", _on_cell_key)
                variable.trace_add("write", lambda *_, v=variable: on_cell_change(v))
                cells[row].append(variable)
                entries[row].append(entry)
        fit_values()
        on_cell_change()  # new cells start with values, so nothing else fires

    def show_grid(rows: int, columns: int, values: list[list[str]] | None = None) -> None:
        build_grid(rows, columns, values)
        fit_on_screen()
        focus_cell(0, 0)

    def apply_typed_size() -> bool:
        """Resize the grid to the size boxes; returns whether it changed. A typed size takes effect on
        Enter or when leaving the box, so typing "15" never passes through a 1-wide grid."""
        if not dialog.winfo_exists():
            return False
        if not rows_count.get():  # left empty: put the current size back
            rows_count.set(str(len(cells)))
        if not columns_count.get():
            columns_count.set(str(len(cells[0])))
        size = int(rows_count.get()), int(columns_count.get())
        if size == (len(cells), len(cells[0])):
            return False
        focused = dialog.focus_get()
        show_grid(*size)
        if focused is not None and focused.winfo_exists():  # e.g. still typing in a size box
            focused.focus_set()
        return True

    def transpose() -> None:
        """Swap rows and columns, e.g. a 1 × 15 kernel becomes 15 × 1 (the other direction)."""
        values = [list(column) for column in zip(*[[variable.get() for variable in row] for row in cells])]
        rows_count.set(str(len(values)))
        columns_count.set(str(len(values[0])))
        show_grid(len(values), len(values[0]), values)

    def fill_all(_event: tk.Event | None = None) -> str:
        if _is_integer_prefix(fill_value.get()) and fill_value.get() not in ("", "-"):
            load(np.full((len(cells), len(cells[0])), int(fill_value.get())))
        return "break"  # Enter here fills the kernel instead of applying it

    only_sizes = (dialog.register(_is_kernel_size_prefix), "%P")
    for variable, label in ((rows_count, "rows"), (columns_count, "columns")):
        box = ttk.Spinbox(sizes, values=SIZES, textvariable=variable, width=4, wrap=False,
                          command=apply_typed_size, validate="key", validatecommand=only_sizes)
        box.pack(side="left")
        box.bind("<KeyPress>", _block_non_digit_key)
        box.bind("<Return>", lambda _event: (apply_typed_size(), "break")[1])  # resize, do not Apply
        box.bind("<FocusOut>", lambda _event: dialog.after_idle(apply_typed_size))
        ttk.Label(sizes, text=" × " if label == "rows" else "").pack(side="left")
    ttk.Button(sizes, text="Transpose", command=transpose).pack(side="left", padx=(12, 0))
    for name in PRESETS:
        button = ttk.Button(presets, text=name, style="Preset.TButton",
                            command=lambda name=name: load(sobel(name, len(cells))))
        button.pack(side="left", expand=True, fill="x", padx=3)
        preset_buttons[name] = button

    ttk.Label(kernel_side, text="Kernel (integers; arrow keys move between cells)").pack(anchor="w")
    grid.pack(anchor="w", pady=(4, 8))
    fill_row = ttk.Frame(kernel_side)
    fill_row.pack(anchor="w", pady=(0, 12))
    ttk.Label(fill_row, text="Fill all cells with").pack(side="left")
    fill_value = tk.StringVar(dialog, value="1")
    fill_entry = ttk.Entry(fill_row, textvariable=fill_value, width=5, justify="center",
                           validate="key", validatecommand=only_integers)
    fill_entry.pack(side="left", padx=6)
    fill_entry.bind("<KeyPress>", _on_cell_key)
    fill_entry.bind("<Return>", fill_all)
    ttk.Button(fill_row, text="Fill", command=fill_all).pack(side="left")
    build_grid(DEFAULT_SIZE, DEFAULT_SIZE)

    normalize = tk.BooleanVar(dialog, value=False)
    ttk.Checkbutton(kernel_side, text="Normalize (÷ sum of weights)", variable=normalize).pack(anchor="w")
    ttk.Label(kernel_side, text="All 1s then give the neighbourhood's average (a blur). "
              "Edge kernels whose weights sum to 0, like Sobel, are divided by their positive weights.",
              foreground="#666666", wraplength=250).pack(anchor="w", pady=(2, 0))
    normalize.trace_add("write", schedule_preview)

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
        if apply_typed_size():  # a size was typed but not confirmed: show that grid before applying
            return
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

    load(sobel("Horizontal", DEFAULT_SIZE))
    center_over(dialog, parent.winfo_toplevel())
    dialog.deiconify()
    focus_cell(0, 0)
    dialog.grab_set()
    dialog.wait_window()
    return result[0] if result else None


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Convolution"
    description = "Convolve the image with a hand-written kernel (1 × 1 to 15 × 15), e.g. Sobel edges or a blur."
    works_on_histogram = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_kernel(parent, document.current)
        if choice is None:
            return None
        kernel, output_mode, channel_mode, normalize, output = choice
        if kernel.shape[0] <= LISTED_KERNEL_ROWS and kernel.size <= LISTED_KERNEL_ROWS ** 2:
            rows = {f"Kernel row {i + 1}": "  ".join(f"{value:g}" for value in row) for i, row in enumerate(kernel)}
        elif np.all(kernel == kernel.flat[0]):
            rows = {"Kernel values": f"all {kernel.flat[0]:g}"}
        else:
            rows = {"Kernel values": f"min {kernel.min():g}, max {kernel.max():g}, sum {kernel.sum():g}"}
        return ToolResult(
            image=output,
            message="Applied the convolution kernel.",
            details={
                "Operation": "Convolution",
                "Kernel size": f"{kernel.shape[0]} × {kernel.shape[1]}",
                **rows,
                "Colour images": channel_mode,
                "Output": output_mode,
                "Normalized": _normalization_note(kernel) if normalize else "No",
            },
        )
