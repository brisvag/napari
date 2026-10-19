"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from qtpy.QtCore import QEvent

    from napari.utils.histogram import HistogramSupportLayer


class QtHistogramThread(QThread):
    """Thread for computing histogram in the background."""

    histogram_computed = Signal(object, tuple)

    layer: HistogramSupportLayer | None
    next_layer: HistogramSupportLayer | None

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.layer = None
        self.next_layer = None
        self._need_to_stop = False
        self._last_result: tuple[np.ndarray, np.ndarray] | None = None

    def calculation_loop(self) -> bool:
        """Run the histogram calculation loop.

        Returns
        -------
        bool
            True if the loop finished naturally.
        """
        if self.layer is None:
            return False
        hist = self.layer.histogram
        layer = self.layer
        for result in hist._compute_async_no_events(layer):
            with self.lock:
                if self._need_to_stop:
                    return False
                if self.next_layer is not None:
                    # If the layer has changed, stop processing
                    return True
            self.histogram_computed.emit(layer, result)
        return False

    def run(self) -> None:
        """Compute the histogram in a separate thread."""
        while True:
            with self.lock:
                if self._need_to_stop:
                    return
            if self.calculation_loop():
                with self.lock:
                    self.layer = self.next_layer
                    self.next_layer = None
            else:
                with self.lock:
                    self._need_to_stop = True

    def start(
        self, priority: QThread.Priority = QThread.Priority.InheritPriority
    ) -> None:
        """Start the thread."""
        with self.lock:
            self._need_to_stop = False
        super().start(priority)

    def schedule_next_layer(self, layer: HistogramSupportLayer) -> None:
        need_start = False
        with self.lock:
            if not self.isRunning():
                self.layer = layer
                need_start = True
            else:
                self.next_layer = layer
        if need_start:
            self.start()

    def quit(self) -> None:
        """Stop the thread."""
        with self.lock:
            self._need_to_stop = True
        super().quit()


class QtHistogramContentWidget(QWidget):
    """Shared histogram visualization and settings content.

    Parameters
    ----------
    layer : HistogramSupportLayer
        The napari layer to visualize.
    histogram_thread : QtHistogramThread
        The thread used for computing histograms.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    histogram_widget : QtHistogramWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for mode and log scale controls.
    """

    def __init__(
        self,
        layer: HistogramSupportLayer,
        histogram_thread: QtHistogramThread,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self._constructed = False
        self.histogram_thread = histogram_thread
        histogram_thread.histogram_computed.connect(self._on_histogram_yield)

        for ev in (
            self.layer.histogram.events.bins,
            self.layer.histogram.events.max_samples,
            self.layer.histogram.events.mode,
            self.layer.histogram.events.log_scale,
            self.layer.events.data,
            self.layer.events.contrast_limits_range,
            self.layer.events.set_data,
        ):
            ev.connect(self._schedule_histogram_compute)

    def _ensure_histogram_content(self) -> None:
        """Lazily create the histogram content widget (vispy canvas).

        Sometimes necessary because opengl requires a preexisting context.
        """
        if self._constructed:
            return

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setLayout(layout)
        self.histogram_widget = QtHistogramWidget(
            self.layer,
            parent=self,
        )
        layout.addWidget(self.histogram_widget)

        self.settings_widget = QtHistogramSettingsWidget(
            self.layer.histogram,
            parent=self,
        )
        layout.addWidget(self.settings_widget)

        self._constructed = True

    def _yield_from_histogram(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        yield from self.layer.histogram._compute_async_no_events(self.layer)

    def _schedule_histogram_compute(self) -> None:
        """Run the async histogram compute."""
        self.histogram_thread.schedule_next_layer(self.layer)

    def _on_histogram_yield(self) -> None:
        self.layer.histogram.events.updated()

    def _on_histogram_done(self, event: QEvent | None = None) -> None:
        """Emit ``completed`` on the main thread once the worker finishes."""
        self.layer.histogram.events.completed()

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self.layer.histogram.events, self)
        disconnect_events(self.layer.events, self)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
