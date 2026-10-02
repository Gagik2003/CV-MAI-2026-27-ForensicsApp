"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]
SIDEBAR_BACKGROUND = "#eef1f5"


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar(value="Ready. Open an image to begin.")

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=SIDEBAR_BACKGROUND)
        style.configure("Category.TLabel", background=SIDEBAR_BACKGROUND, font=("TkDefaultFont", 10, "bold"))
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", padding=12, width=235)
        sidebar.pack_propagate(False)
        body.add(sidebar, weight=0)
        ttk.Label(sidebar, text="Forensics tools", style="Title.TLabel", background=SIDEBAR_BACKGROUND).pack(
            anchor="w", pady=(0, 12)
        )
        tool_list, bind_scroll = self._build_scrollable_list(sidebar)
        self.tool_buttons: list[tuple[ForensicsTool, ttk.Button]] = []
        for category, tools in self.registry.categories():
            label = ttk.Label(tool_list, text=category, style="Category.TLabel")
            label.pack(anchor="w", pady=(9, 4))
            bind_scroll(label)
            for tool in tools:
                button = ttk.Button(
                    tool_list,
                    text=tool.title,
                    style=self._button_style(tool),
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", pady=2)
                self.tool_buttons.append((tool, button))
                button.bind("<Enter>", lambda _event, selected=tool: self.status.set(selected.description))
                button.bind("<Leave>", lambda _event: self.status.set("Ready."))
                bind_scroll(button)

        self.image_view = ImageView(body)
        body.add(self.image_view, weight=1)

        inspector = ttk.Frame(body, padding=12, width=250)
        inspector.pack_propagate(False)
        body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree headings", height=15)
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def _build_scrollable_list(self, parent: ttk.Frame):
        """Return a frame that scrolls vertically inside ``parent`` and a wheel-binding helper.

        The scrollbar only appears when the content is taller than the space available.
        """
        canvas = tk.Canvas(parent, background=SIDEBAR_BACKGROUND, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)

        content = ttk.Frame(canvas, style="Sidebar.TFrame")
        window = canvas.create_window((0, 0), window=content, anchor="nw")

        def overflows() -> bool:
            return content.winfo_reqheight() > canvas.winfo_height()

        def update_scrolling(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=(0, 0, content.winfo_reqwidth(), content.winfo_reqheight()))
            if overflows():
                scrollbar.pack(side="right", fill="y", padx=(6, 0), before=canvas)
            else:
                scrollbar.pack_forget()
                canvas.yview_moveto(0)

        def fit_width(event: tk.Event) -> None:
            canvas.itemconfigure(window, width=event.width)
            update_scrolling()

        def on_wheel(event: tk.Event) -> None:
            if not overflows():
                return
            if event.num in (4, 5):  # X11 reports the wheel as buttons
                step = -1 if event.num == 4 else 1
            else:
                step = -event.delta if abs(event.delta) < 120 else -event.delta // 120
            canvas.yview_scroll(step, "units")

        def on_touchpad(event: tk.Event) -> None:
            if not overflows():
                return
            low = event.delta & 0xFFFF  # Tk packs the pixel deltas as (dx << 16) | dy
            delta_y = low if low < 0x8000 else low - 0x10000
            canvas.yview_moveto(canvas.yview()[0] - delta_y / content.winfo_reqheight())

        def bind_scroll(widget: tk.Misc) -> None:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, on_wheel, add="+")
            try:  # Tk 9 reports trackpad gestures separately from the mouse wheel
                widget.bind("<TouchpadScroll>", on_touchpad, add="+")
            except tk.TclError:
                pass

        content.bind("<Configure>", update_scrolling)
        canvas.bind("<Configure>", fit_width)
        bind_scroll(canvas)
        bind_scroll(content)
        return content, bind_scroll

    def _button_style(self, tool: ForensicsTool) -> str:
        """Tools that set ``button_color`` get their own coloured button style."""
        if not tool.button_color:
            return "Tool.TButton"
        name = f"{tool.tool_id}.Tool.TButton"
        style = ttk.Style(self.root)
        style.configure(name, background=tool.button_color, foreground="white")
        style.map(name, background=[("active", tool.button_color)])
        return name

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        try:
            result = tool.run(self.root, self.document)
            if result is None:
                self.status.set(f"Cancelled {tool.title}.")
                return
            if result.document_path is not None:
                self.document.load(result.document_path)
            if result.image is not None:
                self.document.apply(result.image)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self.image_view.show(self.document.current)
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        showing_histogram = self.document.current is not None and "histogram_source" in self.document.current.info
        for tool, button in self.tool_buttons:
            button.configure(state="disabled" if showing_histogram and not tool.works_on_histogram else "normal")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
