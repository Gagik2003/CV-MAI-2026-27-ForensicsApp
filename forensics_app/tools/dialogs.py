"""Small helpers shared by the tools' dialog windows."""

from __future__ import annotations

import tkinter as tk


def center_over(dialog: tk.Toplevel, window: tk.Misc) -> None:
    """Place ``dialog`` so its centre matches the centre of ``window``."""
    dialog.update_idletasks()
    x = window.winfo_rootx() + (window.winfo_width() - dialog.winfo_reqwidth()) // 2
    title_bar = window.winfo_rooty() - window.winfo_y()  # "+x+y" places the frame, not the content
    y = window.winfo_rooty() + (window.winfo_height() - dialog.winfo_reqheight()) // 2 - title_bar
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")
