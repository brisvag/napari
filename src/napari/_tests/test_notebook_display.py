import numpy as np
import pytest

from napari._tests.utils import skip_on_win_ci
from napari._version import __version__
from napari.utils import nbscreenshot


@skip_on_win_ci
def test_nbscreenshot(make_napari_viewer):
    """Test taking a screenshot."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)

    rich_display_object = nbscreenshot(viewer)
    assert hasattr(rich_display_object, '_repr_png_')
    # Trigger method that would run in jupyter notebook cell automatically
    png_bytes = rich_display_object._repr_png_()
    assert rich_display_object.image is not None
    # Test digital watermark is included in bytes of .png file
    version_byte_string = __version__.encode('utf-8')
    assert b'napari version' in png_bytes
    assert version_byte_string in png_bytes
