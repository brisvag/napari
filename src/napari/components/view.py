from __future__ import annotations

import inspect
import itertools
import logging
import os
import warnings
from collections.abc import (
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from urllib.parse import urlparse

import numpy as np

# This cannot be condition to TYPE_CHECKING or the stubgen fails
# with undefined Context.
from pydantic import Field, PrivateAttr
from typing_extensions import deprecated

from napari import layers
from napari.components._layer_slicer import _LayerSlicer
from napari.components._viewer_mouse_bindings import (
    dims_scroll,
    double_click_to_zoom,
    drag_to_zoom,
    layers_scroll,
)
from napari.components.canvas import Canvas
from napari.components.cursor import Cursor, CursorStyle
from napari.components.dims import Dims
from napari.components.layerlist import LayerList
from napari.components.scene import Scene
from napari.errors import (
    MultipleReaderError,
    NoAvailableReaderError,
    ReaderPluginError,
)
from napari.layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers._scalar_field import ScalarFieldBase
from napari.layers._source import Source, layer_source
from napari.layers.image._image_key_bindings import image_fun_to_mode
from napari.layers.image._image_utils import guess_labels
from napari.layers.labels._labels_key_bindings import labels_fun_to_mode
from napari.layers.points._points_key_bindings import points_fun_to_mode
from napari.layers.shapes._shapes_key_bindings import shapes_fun_to_mode
from napari.layers.surface._surface_key_bindings import surface_fun_to_mode
from napari.layers.tracks._tracks_key_bindings import tracks_fun_to_mode
from napari.layers.utils.stack_utils import split_channels
from napari.layers.vectors._vectors_key_bindings import vectors_fun_to_mode
from napari.plugins import _npe2
from napari.plugins.utils import get_preferred_reader
from napari.settings import get_settings
from napari.types import (
    FullLayerData,
    LayerData,
    LayerTypeName,
    PathLike,
    PathOrPaths,
    SampleData,
)
from napari.utils._register import create_func as create_add_method
from napari.utils.action_manager import action_manager
from napari.utils.colormaps import ensure_colormap
from napari.utils.events import (
    Event,
    EventedModel,
    disconnect_events,
)
from napari.utils.key_bindings import KeymapProvider
from napari.utils.misc import ensure_list_of_layer_data_tuple, is_sequence
from napari.utils.mouse_bindings import MousemapProviderPydantic
from napari.utils.progress import progress

if TYPE_CHECKING:
    from napari.window import Window
    from pathlib import Path
    from npe2.types import SampleDataCreator

    from napari.components.camera import Camera
    from napari.components.grid import GridCanvas
    from napari.components.overlays import (
        CanvasAxesOverlay,
        ScaleBarOverlay,
        SceneAxesOverlay,
        TextOverlay,
    )
import typing
from weakref import WeakSet
 
import magicgui as mgui
import numpy as np
 
from napari.utils import _magicgui
from napari.utils.events.event_utils import disconnect_events
 
 
EXCLUDE_JSON = {'layers', 'active_layer'}

__all__ = ['View', 'valid_add_kwargs']


logger = logging.getLogger(__name__)


def _validate_paths_exist(paths: list[PathLike]) -> None:
    """Raise FileNotFoundError if any local (non-URL) path does not exist."""
    for p in paths:
        p_str = str(p)
        parsed = urlparse(p_str)
        if not (parsed.scheme and parsed.netloc) and not Path(p_str).exists():
            raise FileNotFoundError(f'Path {p_str!r} does not exist.')

# KeymapProvider & MousemapProvider should eventually be moved off the View
@mgui.register_type(bind=_magicgui.proxy_viewer_ancestor)
class View(KeymapProvider, MousemapProviderPydantic, EventedModel):
    """View containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.

    Attributes
    ----------
    cursor: napari.components.cursor.Cursor
        The cursor object containing the position and properties of the cursor.
    dims : napari.components.dims.Dimensions
        Contains axes, indices, dimensions and sliders.
    help: str
        A help message of the viewer model
    layers : napari.components.layerlist.LayerList
        List of contained layers.
    mouse_over_canvas: bool
        Indicating whether the mouse cursor is on the viewer canvas.
    title: str
        The title of the viewer model
    tooltip: napari.components.tooltip.Tooltip
        A tooltip showing extra information on the cursor
    window : napari._qt.qt_main_window.Window
        Parent window.
    _ctx: Mapping
        View object context mapping.
    _layer_slicer: napari.components._layer_slicer._Layer_Slicer
        A layer slicer object controlling the creation of a slice
    """

    # Using frozen=True means these attributes aren't settable and don't
    # have an event emitter associated with them
    name: str | None = Field(None, frozen=True)
    canvas: Canvas = Field(default_factory=Canvas, frozen=True)
    scene: Scene = Field(default_factory=Scene, frozen=True)
    cursor: Cursor = Field(default_factory=Cursor, frozen=True)
    dims: Dims = Field(default_factory=Dims, frozen=True)
    layers: LayerList = Field(
        default_factory=LayerList, frozen=True
    )  # Need to create custom JSON encoder for layer!

    # Need to use default factory because slicer is not copyable which
    # is required for default values.
    _layer_slicer: _LayerSlicer = PrivateAttr(default_factory=_LayerSlicer)
    _layer_list_scroll_progress: float = 0

    def __init__(
        self, title='napari', ndisplay=2, order=(), axis_labels=(),
    ) -> None:
        # allow extra attributes during model initialization, useful for mixins
        self.model_config['extra'] = 'allow'
        super().__init__(
            title=title,
            dims={
                'ndim': max(2, len(axis_labels), ndisplay, len(order)),
                'axis_labels': axis_labels,
                'ndisplay': ndisplay,
                'order': order,
            },
        )
        self.model_config['extra'] = 'ignore'

        settings = get_settings()

        self._update_camera_orientation()
        settings.application.events.depth_axis_orientation.connect(
            self._update_camera_orientation
        )
        settings.application.events.vertical_axis_orientation.connect(
            self._update_camera_orientation
        )
        settings.application.events.horizontal_axis_orientation.connect(
            self._update_camera_orientation
        )
        self._update_synced_camera()
        settings.application.events.synced_camera.connect(
            self._update_synced_camera
        )

        settings.experimental.events.async_.connect(self._update_async)

        # Add extra reset_view event. Ideally this should be removed in the
        # future.
        self.events.add(
            reset_view=Event,
        )

        # Connect events
        self.dims.events.ndisplay.connect(self._update_layers)
        self.dims.events.ndisplay.connect(
            self._save_camera_state, position='first'
        )
        self.dims.events.ndisplay.connect(self._on_ndisplay_changed)
        self.dims.events.order.connect(self._update_layers)
        self.dims.events.order.connect(self.fit_to_view)
        self.dims.events.point.connect(self._update_layers)
        # FIXME: the next line is a temporary workaround. With #5522 and #5751 Dims.point became
        #        the source of truth, and is now defined in world space. This exposed an existing
        #        bug where if a field in Dims is modified by the root_validator, events won't
        #        be fired for it. This won't happen for properties because we have dependency
        #        checks. To fix this, we need dep checks for fields (psygnal!) and then we
        #        can remove the following line. Note that because of this we fire double events,
        #        but this should be ok because we have early returns when slices are unchanged.
        self.dims.events.current_step.connect(self._update_layers)

        # Track previous ndisplay for per-mode camera state caching.
        self._previous_ndisplay: int = self.dims.ndisplay

        self.dims.events.margin_left.connect(self._update_layers)
        self.dims.events.margin_right.connect(self._update_layers)
        self.layers.events.inserted.connect(self._on_add_layer)
        self.layers.events.removed.connect(self._on_remove_layer)
        self.layers.events.reordered.connect(self._on_layers_change)
        self.layers.selection.events.active.connect(self._on_active_layer)
        self.layers.events.units.connect(self._on_layers_change)

        # Add mouse callback
        self.mouse_wheel_callbacks.append(dims_scroll)
        self.mouse_wheel_callbacks.append(layers_scroll)
        self.mouse_double_click_callbacks.append(double_click_to_zoom)
        self.mouse_drag_callbacks.append(drag_to_zoom)

    def _update_camera_orientation(self):
        """Update camera orientation based on settings."""
        settings = get_settings()

        self.scene.camera.orientation = (
            settings.application.depth_axis_orientation,
            settings.application.vertical_axis_orientation,
            settings.application.horizontal_axis_orientation,
        )

    def _update_synced_camera(self):
        """Update camera synced mode based on settings."""
        settings = get_settings()
        self.scene.camera.synced = settings.application.synced_camera

    def json(self, **kwargs):
        """Serialize to json."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        exclude = exclude.union(EXCLUDE_JSON)
        return super().json(exclude=exclude, **kwargs)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Convert to a dictionary."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        return super().model_dump(exclude=exclude, **kwargs)

    def __hash__(self):
        return id(self)

    def __str__(self):
        """Simple string representation"""
        return f'napari.View: {self.name}'

    @property
    def _sliced_extent_world_augmented(self) -> np.ndarray:
        """Extent of layers in world coordinates after slicing.

        D is either 2 or 3 depending on if the displayed data is 2D or 3D.

        Returns
        -------
        sliced_extent_world : array, shape (2, D)
        """
        # if not layers are present, assume image-like with dimensions of size 512
        if len(self.layers) == 0:
            return np.vstack(
                [np.full(self.dims.ndim, -0.5), np.full(self.dims.ndim, 511.5)]
            )
        return self.layers._extent_world_augmented[:, self.dims.displayed]

    def reset_view(
        self, *, margin: float = 0.05, reset_camera_angle: bool = True
    ) -> None:
        """Reset the camera and fit the current layers to the canvas.

        Resets the angles of the camera, adjust the camera zoom,
        and centers the view so that all layers are visible,
        accounting for the current grid mode and margin.

        Parameters
        ----------
        margin : float in [0, 1)
            Margin as fraction of the canvas, showing blank space around the
            data. Default is 0.05 (5% of the canvas).
        reset_camera_angle : bool
            Whether to reset the camera angles to (0, 0, 0) before fitting
            to view. Default is True.
        """
        if self.dims.ndisplay == 3 and reset_camera_angle:
            self.scene.camera.angles = (0, 0, 0)
        self.fit_to_view(margin=margin)

    def fit_to_view(self, *, margin: float = 0.05) -> None:
        """Fit the current data view to the canvas.

        Adjusts the camera zoom and centers the view so that all visible layers
        are within the canvas.

        Parameters
        ----------
        margin : float in [0, 1)
            Margin as fraction of the canvas, showing blank space around the
            data. Default is 0.05 (5% of the canvas).
        """
        # Get the scene parameters
        extent, scene_size, corner = self._get_scene_parameters()

        self.scene.camera.center = self._calculate_view_center(
            corner, scene_size
        )

        scale_factor = self._get_scale_factor(margin)

        # Set camera zoom based on ndisplay
        # zoom is defined as the number of canvas pixels per world pixel
        # The default value used below will zoom such that the whole field
        # of view will occupy 95% of the canvas on the most filled axis
        if np.max(scene_size) == 0:
            # TODO: does this even ever happen?
            self.scene.camera.zoom = scale_factor * np.min(self.canvas.size)

        elif self.dims.ndisplay == 2:
            self.scene.camera.zoom = self._get_2d_camera_zoom(
                scene_size, scale_factor
            )

        elif self.dims.ndisplay == 3:
            self.scene.camera.zoom = self._get_3d_camera_zoom(
                extent, scale_factor
            )

        # Emit a reset view event, which is no longer used internally, but
        # which maybe useful for building on napari.
        self.events.reset_view(
            center=self.scene.camera.center,
            zoom=self.scene.camera.zoom,
            angles=self.scene.camera.angles,
        )

    def _save_camera_state(self) -> None:
        """Save camera state for the mode we're leaving (runs at 'first').

        Always caches the current camera state so that the "separate"
        (synced=False) mode can restore it when returning to this
        ndisplay mode. Caching is harmless in synced mode since
        ``_on_ndisplay_changed`` does not use the cached values there.
        """
        self.scene.camera._cache_state(self._previous_ndisplay)

    def _on_ndisplay_changed(self) -> None:
        """Handle ndisplay changes based on the current camera synced mode.

        * ``synced=True`` — center and zoom persist between modes.
          The depth (z) component is set from the dims slider on 2D→3D
          and the dims slider tracks the camera z on 3D→2D.
        * ``synced=False`` — each mode remembers its own center, zoom,
          and angles independently (per-mode caching).
        """
        if self.scene.camera.synced:
            center = list(self.scene.camera.center)
            if len(self.dims.order) >= 3:
                new_display_dim = self.dims.order[-3]
                if self.dims.ndisplay == 3:
                    center[0] = float(self.dims.point[new_display_dim])
                else:
                    self.dims.set_point(new_display_dim, center[0])
                    center[0] = 0.0
            elif self.dims.ndisplay == 2:
                center[0] = 0.0
            self.scene.camera.center = center[0], center[1], center[2]
            self._previous_ndisplay = self.dims.ndisplay
            return

        # Separate (synced=False) — per-mode caching
        new_mode = self.dims.ndisplay
        cached = self.scene.camera._pop_cached_state(new_mode)
        if cached is not None:
            self.scene.camera.center = cached.center
            self.scene.camera.zoom = cached.zoom
            self.scene.camera.angles = cached.angles
        else:
            # First time in this mode — use fit_to_view defaults
            self.fit_to_view()
            self.scene.camera._cache_state(new_mode)
        self._previous_ndisplay = new_mode

    def _get_scene_parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the scene parameters for the current grid mode.

        Returns
        -------
        extent : array, shape (2, D)
            An array with the min/max coordinate values of the layers
            First row is min values, second row is max values.
        scene_size : array, shape (D,)
            Size of the bounding box containing all layers.
        corner : array, shape (D,)
            Minimum coordinate values of the bounding box (i.e. extent[0]).
        """
        extent = self._sliced_extent_world_augmented
        scene_size = extent[1] - extent[0]
        corner = extent[0]

        return extent, scene_size, corner

    def _calculate_view_center(self, corner, scene_size):
        """Calculate the center of the view based on the scene size."""

        center_array = np.add(corner, np.divide(scene_size, 2))[
            -self.dims.ndisplay :
        ]
        center = cast(
            tuple[float, float, float] | tuple[float, float],
            tuple(
                [0.0] * (self.dims.ndisplay - len(center_array))
                + list(center_array)
            ),
        )
        assert len(center) in (2, 3)
        return center

    def _get_scale_factor(self, margin: float) -> float:
        """Get the scale factor for camera zoom with a valid margin."""
        if 0 <= margin < 1:
            return 1 - margin
        raise ValueError(
            f'margin must be between 0 and 1; got {margin} instead.'
        )

    def _get_2d_camera_zoom(
        self, scene_size: np.ndarray, scale_factor: float
    ) -> float:
        """Get the camera zoom for 2D view."""
        scale = np.array(scene_size[-2:])
        scale[np.isclose(scale, 0)] = 1
        return scale_factor * np.min(
            self.canvas.viewbox_size(self.layers) / scale
        )

    def _get_3d_camera_zoom(
        self, extent: np.ndarray, scale_factor: float
    ) -> float:
        """Calculate the zoom such that the minimum of the bounding box fits the canvas."""
        bounding_box = self._calculate_bounding_box(
            extent=extent,
            view_direction=self.scene.camera.view_direction,
            up_direction=self.scene.camera.up_direction,
        )
        return scale_factor * np.min(
            self.canvas.viewbox_size(self.layers) / bounding_box
        )

    @staticmethod
    def _calculate_bounding_box(
        extent: np.ndarray,
        view_direction: tuple[float, float, float],
        up_direction: tuple[float, float, float],
    ) -> np.ndarray:
        """Calculate the bounding box of the rotated extent.

        Parameters
        ----------
        extent : array, shape (2, D)
            An array with shape (2, D) where D is the number of dimensions.
            The min/max coordinate values of the layers in world coordinates.
            First row contains minimum values, second row contains maximum
            values.
        view_direction : 3-tuple of float
            3D view direction vector of the camera.
        up_direction : 3-tuple of float
            3D direction vector pointing up on the canvas.

        Returns
        -------
        bounding_box : array, shape (2,)
            The bounding box of the rotated extent.
        """
        # calculate the difference between the min and max values of the extent
        # to know the size, and then squeeze the (1,D) array to (D) as
        # required for dot product
        size = np.squeeze(np.diff(extent, axis=0))

        # if the size vector is (2,) and the camera vector is (3,)
        # add a very small thickness to the size vector in the Z position
        # to make sure the cross product is valid, and no division by zero
        if len(size) < len(view_direction):
            size = np.insert(size, 0, 1e-10)

        # get the "rightward" direction that is perpendicular to the view and up directions
        right_direction = np.cross(view_direction, up_direction)

        # project the size vector onto the up and right directions to get the
        # displayed height and width.
        # size = [Z Y X] ; direction = [a b c]
        # size · direction =  Za + Yb + Xc = distance of size vector in given direction
        displayed_height = np.dot(np.abs(up_direction), size)
        displayed_width = np.dot(np.abs(right_direction), size)

        return np.array([displayed_height, displayed_width])

    def _new_labels(self) -> None:
        """Create new labels layer filling full world coordinates space."""
        if isinstance(
            base_layer := self.layers.selection.active, ScalarFieldBase
        ):
            layer = Labels(
                data=np.zeros(
                    base_layer.data.shape[
                        : base_layer.ndim
                    ],  # use :base_layer.ndim to cut channels from rgb images
                    dtype=get_settings().application.new_labels_dtype,
                ),
                scale=base_layer.scale,
                translate=base_layer.translate,
                rotate=base_layer.rotate,
                shear=base_layer.shear,
                units=base_layer.units,
                axis_labels=base_layer.axis_labels,
                affine=base_layer.affine,
                name=base_layer.name + ' - Labels',
            )
        elif self.layers.selection:
            # non scalar field layer or more than one layer selected
            layers_extent = self.layers.get_extent(self.layers.selection)
            extent = layers_extent.world
            scale = layers_extent.step
            units = layers_extent.units
            scene_size = extent[1] - extent[0]
            corner = extent[0]
            shape = [
                np.round(s / sc).astype('int') + 1
                for s, sc in zip(scene_size, scale, strict=False)
            ]
            dtype_str = get_settings().application.new_labels_dtype
            empty_labels = np.zeros(shape, dtype=dtype_str)
            active = self.layers.selection.active
            axis_labels = active.axis_labels if active is not None else None
            layer = Labels(
                data=empty_labels,
                translate=np.array(corner),
                scale=scale,
                units=units,
                axis_labels=axis_labels,
            )
        else:
            layer = Labels(
                data=np.zeros(
                    (512, 512),
                    dtype=get_settings().application.new_labels_dtype,
                )
            )
        self.layers.append(layer)

    def _on_layer_reload(self, event: Event) -> None:
        self.dims.units = self.layers.extent.units
        self._layer_slicer.submit(
            layers=[event.layer], dims=self.dims, force=True
        )

    def _update_layers(self, *, layers=None):
        """Updates the contained layers.

        Parameters
        ----------
        layers : list of napari.layers.Layer, optional
            List of layers to update. If none provided updates all.
        """
        layers = layers or self.layers
        self.dims.units = self.layers.extent.units
        self._layer_slicer.submit(layers=layers, dims=self.dims)
        # If the currently selected layer is sliced asynchronously, then the value
        # shown with this position may be incorrect. See the discussion for more details:
        # https://github.com/napari/napari/pull/5377#discussion_r1036280855
        position = list(self.cursor.position)
        if len(position) < self.dims.ndim:
            # cursor dimensionality is outdated — reset to correct dimension
            position = [0.0] * self.dims.ndim
        for ind in self.dims.order[: -self.dims.ndisplay]:
            position[ind] = self.dims.point[ind]
        self.cursor.position = tuple(position)

    def _on_active_layer(self, event):
        """Update viewer state for a new active layer."""
        active_layer = event.value
        if active_layer is None:
            for layer in self.layers:
                layer.update_transform_box_visibility(False)
                layer.update_highlight_visibility(False)
            self.help = ''
            self.cursor.style = CursorStyle.STANDARD
            self.scene.camera.mouse_pan = True
            self.scene.camera.mouse_zoom = True
        else:
            active_layer.update_transform_box_visibility(True)
            active_layer.update_highlight_visibility(True)
            for layer in self.layers:
                if layer != active_layer:
                    layer.update_transform_box_visibility(False)
                    layer.update_highlight_visibility(False)
            self.help = active_layer.help
            self.cursor.style = active_layer.cursor
            self.cursor.size = active_layer.cursor_size
            self.scene.camera.mouse_pan = active_layer.mouse_pan
            self.scene.camera.mouse_zoom = active_layer.mouse_zoom

    def _merge_dims_and_layers_axis_labels(self) -> tuple[str, ...]:
        """Combine layerlist axis labels onto the current dims labels.

        Replaces dims axis label at indices where layers axis labels exist.
        """
        updated_axis_labels = list(self.dims.axis_labels)
        for pos, label in enumerate(self.layers.axis_labels):
            if label != str(pos - self.dims.ndim):
                updated_axis_labels[pos] = label
        return tuple(updated_axis_labels)

    def _on_layers_change(self):
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            ranges = self.layers._ranges
            # TODO: can be optimized with dims.update(), but events need fixing
            self.dims.ndim = len(ranges)
            self.dims.range = ranges
            self.dims.units = self.layers.units
            self.dims.axis_labels = self._merge_dims_and_layers_axis_labels()

        new_dim = self.dims.ndim
        dim_diff = new_dim - len(self.cursor.position)
        if dim_diff < 0:
            self.cursor.position = self.cursor.position[:new_dim]
        elif dim_diff > 0:
            self.cursor.position = tuple(
                list(self.cursor.position) + [0] * dim_diff
            )

    def _update_mouse_pan(self, event):
        """Set the viewer interactive mouse panning"""
        if event.source is self.layers.selection.active:
            self.scene.camera.mouse_pan = event.mouse_pan

    def _update_mouse_zoom(self, event):
        """Set the viewer interactive mouse zoom"""
        if event.source is self.layers.selection.active:
            self.scene.camera.mouse_zoom = event.mouse_zoom

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor.size = event.cursor_size

    def _update_async(self, event: Event) -> None:
        """Set layer slicer to force synchronous if async is disabled."""
        self._layer_slicer._force_sync = not event.value

    def _on_add_layer(self, event):
        """Connect new layer events.

        Parameters
        ----------
        event : :class:`napari.layers.Layer`
            Layer to add.
        """
        layer = event.value

        # Connect individual layer events to viewer events
        # TODO: in a future PR, we should now be able to connect viewer *only*
        # to viewer.layers.events... and avoid direct viewer->layer connections
        layer.events.mouse_pan.connect(self._update_mouse_pan)
        layer.events.mouse_zoom.connect(self._update_mouse_zoom)
        layer.events.cursor.connect(self._update_cursor)
        layer.events.cursor_size.connect(self._update_cursor_size)
        layer.events.data.connect(self._on_layers_change)
        layer.events.scale.connect(self._on_layers_change)
        layer.events.units.connect(self._on_layers_change)
        layer.events.translate.connect(self._on_layers_change)
        layer.events.rotate.connect(self._on_layers_change)
        layer.events.shear.connect(self._on_layers_change)
        layer.events.affine.connect(self._on_layers_change)
        layer.events.axis_labels.connect(self._on_layers_change)
        layer.events.name.connect(self.layers._update_name)
        layer.events.reload.connect(self._on_layer_reload)

        # Update dims
        self._on_layers_change()
        # Slice current layer based on dims
        self._update_layers(layers=[layer])

        if len(self.layers) == 1:
            # set dims slider to the middle of all dimensions
            self.reset_view()
            self.dims._go_to_center_step()

    def _on_remove_layer(self, event):
        """Disconnect old layer events.

        Parameters
        ----------
        event : napari.utils.event.Event
            Event which will remove a layer.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        layer = event.value

        # Disconnect all connections from layer
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self.layers)

        self._on_layers_change()
