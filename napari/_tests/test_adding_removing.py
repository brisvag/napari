import numpy as np
import pytest

from napari._tests.utils import (
    layer_test_data,
    skip_local_popups,
    skip_on_win_ci,
)


@skip_on_win_ci
@skip_local_popups
@pytest.mark.parametrize('Layer, data, _', layer_test_data)
def test_add_all_layers(make_napari_viewer, Layer, data, _):
    """Make sure that all layers can show in the viewer."""
    viewer = make_napari_viewer(show=True)
    viewer.layers.append(Layer(data))


def test_layers_removed_on_close(make_napari_viewer):
    """Test layers removed on close."""
    viewer = make_napari_viewer()

    # add layers
    viewer.add_image(np.random.random((30, 40)))
    viewer.add_image(np.random.random((50, 20)))
    assert len(viewer.layers) == 2

    viewer.close()
    # check layers have been removed
    assert len(viewer.layers) == 0
