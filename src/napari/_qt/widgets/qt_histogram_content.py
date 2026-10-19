"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.qt import create_worker
from napari.utils.events import Event, disconnect_events

if TYPE_CHECKING:
    from collections.abc import Generator

    from qtpy.QtCore import QEvent

    from napari._qt.qthreading import FunctionWorker, GeneratorWorker
    from napari.layers.intensity_mixin import IntensityVisualizationMixin
    from napari.utils.histogram import HistogramSupportLayer


class QtHistogramThread(QThread):
    """Thread for computing histogram in the background.

    Parameters
    ----------
    layer : HistogramSupportLayer | None
        The napari Image layer to visualize.
    """

    histogram_computed = Signal(object, np.ndarray)

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
        for result in hist._compute_async_no_events(self.layer):
            with self.lock:
                if self._need_to_stop:
                    return False
            if self.next_layer is not None:
                # If the layer has changed, stop processing
                return True
            self.histogram_computed.emit(self.layer, result)
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
        with self.lock:
            if not self.isRunning():
                self.layer = layer
                self.start()
            else:
                self.next_layer = layer

    def quit(self) -> None:
        """Stop the thread."""
        with self.lock:
            self._need_to_stop = True
        super().quit()


class QtHistogramContentWidget(QWidget):
    """Shared histogram visualization and settings content.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
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
        layer: IntensityVisualizationMixin,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self._constructed = False
        self._histogram_worker: GeneratorWorker | FunctionWorker | None = None

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

        self.destroyed.connect(self._abort_histogram_worker)

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

    def _schedule_histogram_compute(self, event: Event | None = None) -> None:
        """Run the async histogram compute."""
        self._ensure_histogram_content()
        self._abort_histogram_worker()
        worker = create_worker(self._yield_from_histogram)  # pyrefly: ignore [bad-argument-type]
        self._histogram_worker = worker

        worker.yielded.connect(self._on_histogram_yield)
        worker.finished.connect(self._on_histogram_done)
        worker.start()

    def _on_histogram_yield(self, event: QEvent | None = None) -> None:
        self.layer.histogram.events.updated()

    def _on_histogram_done(self, event: QEvent | None = None) -> None:
        """Emit ``completed`` on the main thread once the worker finishes."""
        self.layer.histogram.events.completed()
        self._abort_histogram_worker()

    def _abort_histogram_worker(self) -> None:
        """Stop any in-flight compute worker."""
        worker = self._histogram_worker
        self._histogram_worker = None
        if worker is None:
            return
        worker.yielded.disconnect()
        worker.finished.disconnect()
        # progress bar should be stopped if present
        pbar = getattr(worker, 'pbar', None)
        if pbar is not None:
            pbar.close()
        worker.quit()

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self.layer.histogram.events, self)
        disconnect_events(self.layer.events, self)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self._abort_histogram_worker()
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
