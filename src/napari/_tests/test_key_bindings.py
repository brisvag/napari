from unittest.mock import Mock

import numpy as np
import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent


@pytest.mark.key_bindings
def test_viewer_key_bindings(make_napari_viewer):
    """Test adding key bindings to the viewer"""
    np.random.seed(0)
    viewer = make_napari_viewer()

    mock_press = Mock()
    mock_release = Mock()
    mock_shift_press = Mock()
    mock_shift_release = Mock()

    @viewer.bind_key('F')
    def key_callback(v):
        assert viewer == v

        # on press
        mock_press.method()

        yield

        # on release
        mock_release.method()

    @viewer.bind_key('Shift-F')
    def key_shift_callback(v):
        assert viewer == v

        # on press
        mock_shift_press.method()

        yield

        # on release
        mock_shift_release.method()

    def send_key(event_type, key, modifiers=Qt.KeyboardModifier.NoModifier):
        qt_event = QKeyEvent(event_type, key, modifiers)
        if event_type == QEvent.Type.KeyPress:
            viewer.window._qt_viewer.keyPressEvent(qt_event)
        else:
            viewer.window._qt_viewer.keyReleaseEvent(qt_event)

    # Simulate press only
    send_key(QEvent.Type.KeyPress, Qt.Key.Key_F)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    send_key(QEvent.Type.KeyRelease, Qt.Key.Key_F)
    mock_press.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate press only
    send_key(QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.ShiftModifier)
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_called_once()
    mock_shift_press.reset_mock()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    send_key(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_F,
        Qt.KeyboardModifier.ShiftModifier,
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_called_once()
    mock_shift_release.reset_mock()


@pytest.mark.key_bindings
def test_layer_key_bindings(make_napari_viewer):
    """Test adding key bindings to a layer"""
    np.random.seed(0)
    viewer = make_napari_viewer()

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.add(layer)

    mock_press = Mock()
    mock_release = Mock()
    mock_shift_press = Mock()
    mock_shift_release = Mock()

    @layer.bind_key('F')
    def key_callback(_layer):
        assert layer == _layer
        # on press
        mock_press.method()
        yield
        # on release
        mock_release.method()

    @layer.bind_key('Shift-F')
    def key_shift_callback(_layer):
        assert layer == _layer

        # on press
        mock_shift_press.method()

        yield

        # on release
        mock_shift_release.method()

    # Simulate press only
    send_key(QEvent.Type.KeyPress, Qt.Key.Key_F)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    send_key(QEvent.Type.KeyRelease, Qt.Key.Key_F)
    mock_press.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate press only
    send_key(QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.ShiftModifier)
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_called_once()
    mock_shift_press.reset_mock()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    send_key(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_F,
        Qt.KeyboardModifier.ShiftModifier,
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_called_once()
    mock_shift_release.reset_mock()
