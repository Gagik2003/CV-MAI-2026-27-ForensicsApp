import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import (
    _is_integer_prefix,
    _toggle_sign,
    _without_leading_zeros,
    convolve,
    sobel,
)
from forensics_app.tools.edge_detection import VIEWS, detect_edges, show_edges
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import histogram_image
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

    def test_histogram_image_is_a_plot(self) -> None:
        plot = histogram_image(Image.new("RGB", (4, 4), (10, 20, 30)))
        self.assertEqual(plot.mode, "RGB")

    def test_channel_button_on_histogram_redraws_that_channel(self) -> None:
        source = Image.new("RGB", (4, 4), (10, 20, 30))
        document = ImageDocument()
        document.current = source
        document.apply(histogram_image(source))
        result = SplitChannelTool("R").run(None, document)
        self.assertIs(result.image.info["histogram_source"], source)

    def test_convolution_flips_the_kernel(self) -> None:
        image = Image.frombytes("L", (3, 1), bytes([10, 20, 30]))
        left_cell = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        # convolution: out[x] = in[x + 1]; correlation (no flip) would give in[x - 1] = [10, 10, 20]
        self.assertEqual(list(convolve(image, left_cell).getdata()), [20, 30, 30])

    def test_convolution_output_modes_handle_negative_responses(self) -> None:
        step = Image.frombytes("L", (4, 1), bytes([0, 0, 10, 10]))
        horizontal = sobel("Horizontal")
        self.assertEqual(list(convolve(step, horizontal, "Absolute value").getdata()), [0, 40, 40, 0])
        self.assertEqual(list(convolve(step, horizontal, "Clip to 0–255").getdata()), [0, 0, 0, 0])
        self.assertEqual(list(convolve(step, sobel("Vertical")).getdata()), [0, 0, 0, 0])

    def test_convolution_colour_modes_return_one_channel(self) -> None:
        image = Image.new("RGB", (3, 3), (10, 20, 30))
        identity = np.zeros((3, 3))
        identity[1, 1] = 1
        self.assertEqual(convolve(image, identity, channels="Mean of channels").getpixel((1, 1)), 20)
        self.assertEqual(convolve(image, identity, channels="Grayscale").getpixel((1, 1)), 18)  # luminance

    def test_channel_mean_finds_edges_between_equally_bright_colours(self) -> None:
        image = Image.new("RGB", (4, 1), (0, 100, 0))
        image.paste((196, 0, 0), (2, 0, 4, 1))  # same luminance as the green half
        horizontal = sobel("Horizontal")
        self.assertEqual(convolve(image, horizontal, channels="Grayscale").getpixel((1, 0)), 0)
        self.assertEqual(convolve(image, horizontal, channels="Mean of channels").getpixel((1, 0)), 170)

    def test_normalized_ones_kernel_is_a_mean_blur(self) -> None:
        image = Image.frombytes("L", (5, 1), bytes([0, 0, 90, 0, 0]))
        ones = np.ones((3, 3))
        self.assertEqual(list(convolve(image, ones, normalize=True).getdata()), [0, 30, 30, 30, 0])
        self.assertEqual(list(convolve(image, ones).getdata()), [0, 255, 255, 255, 0])  # sum of 3 rows of 90
        flat = Image.new("RGB", (4, 4), (40, 80, 120))
        self.assertEqual(convolve(flat, np.ones((7, 7)), normalize=True).getpixel((2, 2)), 80)

    def test_normalize_leaves_zero_sum_kernels_alone(self) -> None:
        step = Image.frombytes("L", (4, 1), bytes([0, 0, 10, 10]))
        horizontal = sobel("Horizontal")
        self.assertEqual(list(convolve(step, horizontal, normalize=True).getdata()),
                         list(convolve(step, horizontal).getdata()))

    def test_sobel_kernels_match_opencv(self) -> None:
        self.assertEqual(sobel("Horizontal", 3).tolist(), [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.assertEqual(sobel("Vertical", 3).tolist(), [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.assertEqual(sobel("Horizontal", 5)[2].tolist(), [-6, -12, 0, 12, 6])
        self.assertEqual(sobel("Horizontal", 7)[0].tolist(), [-1, -4, -5, 0, 5, 4, 1])
        self.assertTrue(np.array_equal(sobel("Vertical", 5), sobel("Horizontal", 5).T))

    def test_convolution_accepts_larger_kernels(self) -> None:
        image = Image.frombytes("L", (5, 1), bytes([10, 20, 30, 40, 50]))
        two_left = np.zeros((5, 5))
        two_left[2, 0] = 1  # convolution: out[x] = in[x + 2]
        self.assertEqual(list(convolve(image, two_left).getdata()), [30, 40, 50, 50, 50])

    def test_kernel_cells_accept_only_integers(self) -> None:
        for text in ("", "-", "0", "7", "-12", "100"):
            self.assertTrue(_is_integer_prefix(text), text)
        for text in ("a", "1a", "0.5", "1.", "--1", "1-", "+1", " 1", "1e3"):
            self.assertFalse(_is_integer_prefix(text), text)

    def test_kernel_cells_drop_leading_zeros(self) -> None:
        cases = {"00": "0", "01": "1", "007": "7", "-05": "-5", "-0": "-", "0": "0", "100": "100", "-": "-", "": ""}
        for text, expected in cases.items():
            self.assertEqual(_without_leading_zeros(text), expected, text)

    def test_minus_key_toggles_the_sign(self) -> None:
        cases = {"3": "-3", "-3": "3", "12": "-12", "0": "-", "": "-", "-": ""}
        for text, expected in cases.items():
            self.assertEqual(_toggle_sign(text), expected, text)

    def test_canny_finds_the_border_of_a_square(self) -> None:
        image = Image.new("L", (40, 40), 0)
        image.paste(255, (10, 10, 30, 30))
        edges = detect_edges(image)
        self.assertTrue(edges[20, 9:12].any())  # left side of the square
        self.assertFalse(edges[20, 15:25].any())  # flat inside
        self.assertGreater(detect_edges(image, sigma=0.5).sum(), 0)
        with self.assertRaises(ValueError):
            detect_edges(image, low=0.3, high=0.2)

    def test_edge_views(self) -> None:
        image = Image.new("RGB", (4, 1), (100, 100, 100))
        edges = np.array([[False, True, False, False]])
        self.assertEqual(show_edges(image, edges).mode, "L")
        self.assertEqual(list(show_edges(image, edges).getdata()), [0, 255, 0, 0])
        red = show_edges(image, edges, "Coloured edges")
        self.assertEqual(red.getpixel((1, 0)), (255, 0, 0))
        self.assertEqual(red.getpixel((0, 0)), (100, 100, 100))
        overlay = show_edges(image, edges, "Transparent overlay")
        self.assertEqual(overlay.getpixel((1, 0)), (178, 50, 50))
        self.assertEqual(overlay.getpixel((0, 0)), (50, 50, 50))
        dim = show_edges(image, edges, "Bright edges, dim image")
        self.assertEqual(dim.getpixel((1, 0)), (255, 255, 255))
        self.assertEqual(dim.getpixel((0, 0)), (40, 40, 40))
        for view in VIEWS[1:]:
            self.assertEqual(show_edges(image, edges, view).size, image.size, view)
        self.assertEqual(show_edges(image, edges, "Coloured edges", colour="Blue").getpixel((1, 0)), (0, 0, 255))
        self.assertEqual(show_edges(image, edges, "Transparent overlay", colour="Green", opacity=1).getpixel((1, 0)),
                         (0, 255, 0))
        self.assertEqual(show_edges(image, edges, "Bright edges, dim image", brightness=0).getpixel((0, 0)), (0, 0, 0))
        square = np.zeros((9, 9), dtype=bool)
        square[4, 2:7] = True
        contour = show_edges(Image.new("L", (9, 9), 100), square, "Contour overlay")
        self.assertEqual(contour.getpixel((4, 3)), (255, 0, 0))  # outline just above the edge line
        self.assertEqual(contour.getpixel((0, 0)), (100, 100, 100))

    def test_edge_thickness_widens_edges_but_not_the_edge_map(self) -> None:
        image = Image.new("L", (9, 9), 100)
        edges = np.zeros((9, 9), dtype=bool)
        edges[4, 4] = True
        thick = np.asarray(show_edges(image, edges, "Coloured edges", colour="Green", thickness=3))
        self.assertEqual(int((thick == (0, 255, 0)).all(axis=2).sum()), 9)  # a 3 × 3 block
        self.assertEqual(int(np.asarray(show_edges(image, edges, thickness=3)).astype(bool).sum()), 1)
        wide = np.asarray(show_edges(image, edges, "Contour overlay", thickness=1)).copy()
        wider = np.asarray(show_edges(image, edges, "Contour overlay", thickness=4))
        self.assertGreater(int((wider == (255, 0, 0)).all(axis=2).sum()), int((wide == (255, 0, 0)).all(axis=2).sum()))

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
