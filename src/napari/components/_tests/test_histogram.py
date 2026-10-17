"""Tests for HistogramModel."""

from __future__ import annotations

import numpy as np
import pytest

from napari.layers import Image
from napari.utils.histogram import (
    _MAX_MATERIALIZE_ELEMENTS,
)

_DEFAULT_MAX_SAMPLES = 1_000_000


def _image(data, **kwargs):
    """Create an Image layer (sliced at construction time)."""
    return Image(data, **kwargs)


class TestDefaultState:
    """Test the default state of HistogramModel."""

    def test_default_mode_canvas(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.mode == 'canvas'

    def test_default_bins_256(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.bins == 256

    def test_default_log_scale_false(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert not model.log_scale

    def test_default_max_samples(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.max_samples == _DEFAULT_MAX_SAMPLES


class TestCompute:
    """Test synchronous compute and metadata writing."""

    def test_compute_returns_bin_edges_counts(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        bin_edges, counts = model.compute(layer)
        assert len(counts) == model.bins
        assert len(bin_edges) == model.bins + 1

    def test_compute_emits_events(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        events = []
        model.events.updated.connect(lambda: events.append('updated'))
        model.events.completed.connect(lambda: events.append('completed'))
        model.compute(layer)
        assert 'updated' in events
        assert 'completed' in events

    def test_compute_async_no_events_does_not_emit(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        fired = []
        model.events.updated.connect(lambda: fired.append('u'))
        model.events.completed.connect(lambda: fired.append('c'))
        list(model._compute_async_no_events(layer))
        # No events fire from the no-events variant...
        assert fired == []


class TestDataTypes:
    """Test histogram with various data types."""

    @pytest.mark.parametrize(
        'dtype',
        ['uint8', 'uint16', 'float32', 'float64', 'int16'],
    )
    def test_various_dtypes(self, dtype):
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 255, (20, 20), dtype=dtype)
        else:
            data = rng.random((20, 20)).astype(dtype)
        layer = _image(data)
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_all_zeros(self):
        layer = _image(np.zeros((10, 10), dtype=np.float32))
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_constant_data(self):
        layer = _image(np.full((10, 10), 42.0, dtype=np.float32))
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_with_nan(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.nan
        layer = _image(data.astype(np.float32))
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts))

    def test_with_inf(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.inf
        data[0, 1] = -np.inf
        layer = _image(data.astype(np.float32))
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts))


class TestCustomBins:
    """Test custom bin counts."""

    def test_custom_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 128
        bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 128
        assert len(bin_edges) == 129

    def test_large_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 4096
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 4096

    def test_small_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 2
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 2


class TestMode:
    """Test canvas vs full mode."""

    def test_canvas_mode(self):
        layer = _image(np.random.rand(5, 20, 20))
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_full_mode(self):
        layer = _image(np.random.rand(5, 20, 20))
        layer.histogram.mode = 'full'
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_full_mode_uses_coarsest_multiscale(self):
        data = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]
        layer = _image(data, multiscale=True)
        layer.histogram.mode = 'full'
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() == data[-1].size


class TestLogScale:
    """Test log scale."""

    def test_log_scale_differs(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        _, linear = model.compute(layer)
        model.log_scale = True
        _, logged = model.compute(layer)
        assert len(logged) == len(linear)
        assert not np.array_equal(linear, logged)

    def test_log_scale_values_smaller(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        _, linear = model.compute(layer)
        model.log_scale = True
        _, logged = model.compute(layer)
        assert logged.max() <= linear.max()


class TestRGB:
    """Test histogram with RGB data."""

    def test_rgb_image(self):
        data = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
        layer = _image(data, rgb=True)
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0


class TestNoneDataPath:
    """Tests for compute() with unavailable data."""

    def test_compute_with_no_display_data(self, monkeypatch):
        """When there is no displayed data, compute yields the empty result."""
        import napari.utils.histogram as hm

        layer = _image(np.random.rand(10, 10))
        monkeypatch.setattr(hm, '_get_data', lambda layer, mode: None)
        bin_edges, counts = layer.histogram.compute(layer)
        assert len(bin_edges) == 2
        assert len(counts) == 1
        assert counts[0] == 0


class TestChunked:
    """Tests for chunked (dask / zarr) data."""

    def test_dask_full_mode_progressive_yields(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=(50, 50)
        )
        layer = _image(data)
        layer.histogram.mode = 'full'
        results = list(layer.histogram.compute_async(layer))
        assert len(results) >= 1
        _bin_edges, counts = results[-1]
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_dask_canvas_mode(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(5, 50, 50).astype(np.float32), chunks=(1, 25, 25)
        )
        layer = _image(data)
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_zarr_full_mode(self):
        zarr = pytest.importorskip('zarr')
        data = zarr.array(
            np.random.rand(50, 50).astype(np.float32), chunks=(25, 25)
        )
        layer = _image(data)
        layer.histogram.mode = 'full'
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0


class TestMaterializationGuard:
    """Tests for the _get_full_data materialization guard."""

    def test_large_non_chunked_data_skipped_with_warning(self):
        from napari.utils.histogram import _get_full_data

        class Big:
            size = _MAX_MATERIALIZE_ELEMENTS + 1
            shape = (10_000, 5_001)
            dtype = np.dtype(np.float32)

        class FakeLayer:
            data = Big()

        with pytest.warns(UserWarning, match='Skipping full-data histogram'):
            result = _get_full_data(FakeLayer())
        assert result is None


@pytest.fixture
def no_background_compute(monkeypatch):
    """Keep cache tests deterministic: no worker thread is started."""
    from napari.components.histogram import Histogram

    monkeypatch.setattr(Histogram, '_request', lambda self, layer: None)


@pytest.mark.usefixtures('no_background_compute')
class TestCache:
    """Results are cached per mode with linear counts."""

    def _enabled_image(self):
        layer = _image(np.random.default_rng(0).random((20, 20)))
        layer.histogram.enabled = True
        return layer

    def test_log_scale_toggle_uses_cache(self, monkeypatch):
        layer = self._enabled_image()
        _, linear = layer.histogram.compute(layer)

        def fail(*args, **kwargs):
            raise AssertionError('should not recompute')

        monkeypatch.setattr(
            'napari.components.histogram.compute_histogram', fail
        )
        completed = []
        layer.histogram.events.completed.connect(
            lambda e: completed.append(e.counts)
        )
        layer.histogram.log_scale = True
        np.testing.assert_allclose(completed[-1], np.log10(linear + 1))
        layer.histogram.log_scale = False
        np.testing.assert_allclose(completed[-1], linear)

    def test_full_cache_survives_mode_round_trip(self):
        layer = self._enabled_image()
        layer.histogram.mode = 'full'
        _, full = layer.histogram.compute(layer)
        layer.histogram.mode = 'canvas'
        layer.histogram.compute(layer)
        layer.histogram.mode = 'full'
        cached = layer.histogram.cached()
        assert cached is not None
        np.testing.assert_array_equal(cached[1], full)

    def test_slice_change_only_drops_canvas(self):
        layer = self._enabled_image()
        layer.histogram.compute(layer)
        layer.histogram.mode = 'full'
        layer.histogram.compute(layer)
        layer.events.set_data()
        assert layer.histogram.cached() is not None
        layer.histogram.mode = 'canvas'
        assert layer.histogram.cached() is None

    @pytest.mark.parametrize(
        'change',
        [
            lambda layer: setattr(layer, 'data', layer.data * 2),
            lambda layer: setattr(layer, 'contrast_limits_range', (0, 5)),
            lambda layer: setattr(layer.histogram, 'bins', 16),
        ],
    )
    def test_invalidation(self, change):
        layer = self._enabled_image()
        layer.histogram.compute(layer)
        change(layer)
        assert layer.histogram.cached() is None

    def test_disable_drops_cache_and_disconnects(self):
        layer = self._enabled_image()
        layer.histogram.compute(layer)
        layer.histogram.enabled = False
        assert layer.histogram.cached() is None
        assert len(layer.events.set_data.callbacks) == 0

    def test_stale_only_when_enabled(self):
        layer = _image(np.random.rand(10, 10))
        stale = []
        layer.histogram.events.stale.connect(lambda: stale.append(1))
        layer.histogram.bins = 16
        assert not stale
        layer.histogram.enabled = True
        layer.histogram.bins = 32
        assert stale


def test_compute_with_empty_generator(monkeypatch):
    """A generator yielding nothing (failed first chunk) is not an error."""
    monkeypatch.setattr(
        'napari.components.histogram.compute_histogram',
        lambda *a, **k: iter(()),
    )
    layer = _image(np.random.rand(10, 10))
    bin_edges, counts = layer.histogram.compute(layer)
    assert len(bin_edges) == 2
    assert len(counts) == 1


def test_histogram_model_deprecated():
    from napari.components.histogram import Histogram, HistogramModel

    layer = _image(np.random.rand(10, 10))
    with pytest.warns(FutureWarning, match='HistogramModel is deprecated'):
        model = HistogramModel(layer, bins=8)
    assert isinstance(model, Histogram)
    assert model.bins == 8
    with pytest.warns(FutureWarning, match='counts is deprecated'):
        assert len(model.counts) == 1


@pytest.mark.usefixtures('no_background_compute')
def test_log_toggle_mid_compute_rescales_partial():
    """Toggling log during a progressive compute must not ask for a restart."""
    layer = _image(np.random.rand(10, 10))
    hist = layer.histogram
    hist.enabled = True
    hist.mode = 'full'
    stale, updated = [], []
    hist.events.stale.connect(lambda: stale.append(1))
    hist.events.updated.connect(lambda e: updated.append(e.counts))

    edges, linear = np.linspace(0, 1, 5), np.array([0.0, 9.0, 99.0, 999.0])
    hist._publish_partial(edges, linear, 'full')
    hist.log_scale = True

    assert not stale
    np.testing.assert_allclose(updated[-1], [0, 1, 2, 3])
    hist._publish_result(edges, linear, 'full')
    np.testing.assert_allclose(hist.cached()[1], [0, 1, 2, 3])


class TestBackgroundCompute:
    """The model computes in a daemon thread while enabled (no Qt needed)."""

    def _wait(self, predicate, timeout=5.0):
        import time

        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError('timed out')

    def test_enable_computes_in_background(self):
        layer = _image(np.random.rand(20, 20))
        completed = []
        layer.histogram.events.completed.connect(
            lambda e: completed.append(e.counts)
        )
        layer.histogram.enabled = True
        self._wait(lambda: completed)
        assert layer.histogram.cached() is not None
        layer.histogram.enabled = False

    def test_invalidation_drops_late_result(self):
        layer = _image(np.random.rand(20, 20))
        hist = layer.histogram
        generation = hist._generation['canvas']
        hist._clear()
        edges, counts = np.linspace(0, 1, 3), np.ones(2)
        assert not hist._publish_result(edges, counts, 'canvas', generation)
        assert hist.cached() is None

    def test_disable_stops_thread(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.enabled = True
        worker = layer.histogram._worker
        self._wait(lambda: layer.histogram.cached() is not None)
        layer.histogram.enabled = False
        assert layer.histogram._worker is None
        self._wait(lambda: not worker._thread.is_alive())


def test_removing_layer_stops_histogram_worker(monkeypatch):
    """A layer removed mid-compute must not keep reading chunks."""
    import time

    from napari.components import ViewerModel
    from napari.utils import histogram as hist_utils

    da = pytest.importorskip('dask.array')
    load_chunk = hist_utils._load_chunk

    def slow_load_chunk(data, flat_idx):
        time.sleep(0.02)
        return load_chunk(data, flat_idx)

    monkeypatch.setattr(hist_utils, '_load_chunk', slow_load_chunk)
    viewer = ViewerModel()
    layer = viewer.add_image(da.random.random((100, 8, 8), chunks=(1, 8, 8)))
    layer.histogram.mode = 'full'
    layer.histogram.enabled = True
    thread = layer.histogram._worker._thread

    viewer.layers.remove(layer)
    assert layer.histogram._worker is None
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert layer.histogram.cached() is None
