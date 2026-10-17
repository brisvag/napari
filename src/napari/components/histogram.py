"""Histogram model for Image/Surface layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import PrivateAttr

from napari.utils.events import Event, EventedModel
from napari.utils.histogram import (
    _make_empty,
    compute_histogram,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from napari.layers.intensity_mixin import IntensityVisualizationMixin


class Histogram(EventedModel):
    """Histogram model with controls for histogram generation.

    Results are cached per mode with linear counts, so toggling
    `log_scale` or switching back to a previously computed `mode`
    does not recompute. The owning layer invalidates the cache through
    `_invalidate` (data / contrast limits range) and `_on_slice_change`.

    Events
    ------
    updated : Event
        Intermediate `(bin_edges, counts)` during a progressive compute.
    completed : Event
        Final `(bin_edges, counts)`, also fired when a cached result is
        restored.
    stale : Event
        Fired (only while `enabled`) when the cached result for the
        current mode is no longer valid and a recompute is needed.
    """

    enabled: bool = False
    bins: int = 256
    max_samples: int = 1_000_000
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False

    # mode -> (bin_edges, linear counts)
    _cache: dict[str, tuple[np.ndarray, np.ndarray]] = PrivateAttr(
        default_factory=dict
    )
    # mode -> latest progressive (bin_edges, linear counts) of an in-flight
    # compute, so a log toggle can rescale it instead of restarting
    _partial: dict[str, tuple[np.ndarray, np.ndarray]] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.events.add(completed=Event, updated=Event, stale=Event)
        self.events.bins.connect(self._invalidate)
        self.events.max_samples.connect(self._invalidate)
        self.events.mode.connect(self._restore_or_stale)
        self.events.log_scale.connect(self._on_log_scale_change)

    def cached(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the cached result for the current mode, if any."""
        if self.mode not in self._cache:
            return None
        bin_edges, counts = self._cache[self.mode]
        return bin_edges, self._scale(counts)

    def compute_async(
        self, layer: IntensityVisualizationMixin
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield (bin_edges, counts) for layer using the model settings.

        This is a generator that can be used to update a representation asyncronously.

        Do not use in a separate thread (use _compute_async_no_events instead).
        """
        # the generator may yield nothing (e.g. first chunk fails to load)
        bin_edges, counts = _make_empty().values()
        for bin_edges, counts in self._compute_async_no_events(layer):
            self._publish_partial(bin_edges, counts)
            yield bin_edges, self._scale(counts)
        self._publish_result(bin_edges, counts)

    def compute(
        self, layer: IntensityVisualizationMixin
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the full histogram for layer synchronously.

        Returns the cached result if still valid. Otherwise emits updated
        (once per chunk) and completed on the calling thread, so listeners
        (e.g. Qt controls) update accordingly.
        """
        if (cached := self.cached()) is not None:
            return cached
        bin_edges, counts = _make_empty().values()
        for res in self.compute_async(layer):
            bin_edges, counts = res
        return bin_edges, counts

    def _compute_async_no_events(
        self, layer: IntensityVisualizationMixin
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Non-evented generator yielding *linear* counts, for use in threads.

        Responsibility to fire events in the main thread lies on the caller,
        via `_publish_partial` and `_publish_result`.
        """
        yield from compute_histogram(
            layer,
            bin_count=self.bins,
            max_samples=self.max_samples,
            mode=self.mode,
            log_scale=False,
        )

    def _publish_partial(
        self,
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mode: str | None = None,
    ) -> None:
        """Emit an intermediate result (linear counts in, scaled out)."""
        mode = mode or self.mode
        self._partial[mode] = (bin_edges, counts)
        if mode != self.mode:
            return
        self.events.updated(bin_edges=bin_edges, counts=self._scale(counts))

    def _publish_result(
        self,
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mode: str | None = None,
    ) -> None:
        """Cache a final result for `mode` and emit it if it is current."""
        mode = mode or self.mode
        self._cache[mode] = (bin_edges, counts)
        self._partial.pop(mode, None)
        if mode != self.mode:
            return
        self.events.completed(bin_edges=bin_edges, counts=self._scale(counts))

    def _scale(self, counts: np.ndarray) -> np.ndarray:
        if self.log_scale:
            return np.log10(counts + 1).astype(np.float32)
        return counts.astype(np.float32)

    def _restore_or_stale(self) -> None:
        """Re-emit the cached result for the current mode, or ask for one."""
        if (cached := self.cached()) is not None:
            bin_edges, counts = cached
            self.events.completed(bin_edges=bin_edges, counts=counts)
        elif self.enabled:
            self.events.stale()

    def _on_log_scale_change(self) -> None:
        """Rescale the cached or in-progress result; never recompute.

        Counts are stored linearly, so an in-flight compute stays valid and
        its later yields pick up the new scale.
        """
        if (cached := self.cached()) is not None:
            self.events.completed(bin_edges=cached[0], counts=cached[1])
        elif self.mode in self._partial:
            bin_edges, counts = self._partial[self.mode]
            self.events.updated(
                bin_edges=bin_edges, counts=self._scale(counts)
            )

    def _clear(self, mode: str | None = None) -> None:
        """Drop cached and partial results for ``mode`` (or all modes)."""
        if mode is None:
            self._cache.clear()
            self._partial.clear()
        else:
            self._cache.pop(mode, None)
            self._partial.pop(mode, None)

    def _invalidate(self, event: Event | None = None) -> None:
        """Drop all cached results (data, range, bins or samples changed)."""
        self._clear()
        if self.enabled:
            self.events.stale()

    def _on_slice_change(self, event: Event | None = None) -> None:
        """A new slice only invalidates the displayed-data histogram."""
        self._clear('canvas')
        if self.enabled and self.mode == 'canvas':
            self.events.stale()

    def reset(self) -> None:
        """Reset settings to defaults and drop cached results."""
        super().reset()
        self._clear()
