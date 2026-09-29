import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.split_channel import SplitChannelTool
from forensics_app.tools.swap_channels import swap_channels


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_split_channel_returns_selected_band(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2), (10, 20, 30))
        result = SplitChannelTool("G").run(None, document)
        self.assertEqual(result.image.getpixel((0, 0)), 20)

    def test_swap_channels_swaps_two_bands(self) -> None:
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        self.assertEqual(swap_channels(image, "R", "B").getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(swap_channels(image, "G", "R").getpixel((0, 0)), (20, 10, 30))

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
