"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget
from superqt import ensure_main_thread

from napari._qt.widgets.qt_histogram_canvas import QtHistogramCanvasWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.settings import get_settings
from napari.utils.events import disconnect_events
from napari.utils.histogram import _make_empty
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from qtpy.QtGui import QShowEvent

    from napari.layers import Image, Surface
    from napari.utils.events import Event


class QtHistogramWidget(QWidget):
    """Shared histogram visualization and settings content.

    Display only: the layer's histogram model computes in the background
    and this widget renders whatever it publishes.

    Parameters
    ----------
    layer : Image | Surface
        The napari layer to visualize.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    histogram_canvas : QtHistogramCanvasWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for mode and log scale controls.
    """

    def __init__(
        self,
        layer: Image | Surface,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self.bin_edges, self.counts = _make_empty().values()
        self._constructed = False
        self._constructing_content = False
        self._appearance = get_settings().appearance

        self.layer.histogram.events.updated.connect(self._on_model_histogram)  # pyrefly: ignore [bad-argument-type]
        self.layer.histogram.events.completed.connect(self._on_model_histogram)  # pyrefly: ignore [bad-argument-type]
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(self.update_histogram)
        self.layer.events.gamma.connect(self.update_histogram)
        self._appearance.events.theme.connect(
            self._on_appearance_theme_changed
        )

    def _ensure_histogram_content(self) -> None:
        """Lazily create the histogram content widget (vispy canvas).

        Sometimes necessary because opengl requires a preexisting context.
        Also, reparenting the native canvas widget can double re-trigger showEvent
        on an already-visible widget, so guard against that.
        """
        if self._constructed or self._constructing_content:
            return
        self._constructing_content = True
        try:
            layout = QVBoxLayout()
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
            self.setLayout(layout)
            self.histogram_canvas = QtHistogramCanvasWidget(
                parent=self,
            )
            layout.addWidget(self.histogram_canvas)
            self.histogram_canvas.update_colormap(self.layer.colormap)

            self.settings_widget = QtHistogramSettingsWidget(
                self.layer.histogram,
                parent=self,
            )
            layout.addWidget(self.settings_widget)

            self._constructed = True
        finally:
            self._constructing_content = False

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._ensure_histogram_content()
        if (cached := self.layer.histogram.cached()) is not None:
            self.bin_edges, self.counts = cached
        self.update_histogram()

    @ensure_main_thread
    def _on_model_histogram(self, event: Event) -> None:
        # the model publishes from its worker thread
        self.bin_edges = event.bin_edges
        self.counts = event.counts
        self.update_histogram()

    def _on_colormap_change(self) -> None:
        if self._constructed:
            self.histogram_canvas.update_colormap(self.layer.colormap)

    def _on_appearance_theme_changed(self, event: Event) -> None:
        if not self._constructed:
            return
        self.histogram_canvas._on_theme_change(get_theme(event.value))

    def update_histogram(self) -> None:
        if self._constructed:
            self.histogram_canvas.histogram_visual.set_data(
                bin_edges=self.bin_edges,
                counts=self.counts,
                gamma=self.layer.gamma,
                clims=self.layer.contrast_limits,  # pyrefly: ignore [bad-argument-type]
                data_range=self.layer.contrast_limits_range,  # pyrefly: ignore [bad-argument-type]
            )
            self.histogram_canvas.canvas.update()

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self.layer.histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self.disconnect_widget_controls()
        if self._constructed:
            # updates already queued from the worker thread may still
            # arrive; they must not touch the destroyed canvas
            self._constructed = False
            self.settings_widget.cleanup()
            self.histogram_canvas.cleanup()
