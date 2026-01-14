"""napari.components provides the public-facing models for widgets
and other utilities that the user will be able to programmatically interact
with.

Classes
-------
Dims
    Current indices along each data dimension, together with which dimensions
    are being displayed, projected, sliced...
LayerList
    List of layers currently present in the viewer.
ViewerModel
    Data viewer displaying the currently rendered scene and
    layer-related controls.
"""

from lazy_loader import attach

# Note that importing _viewer_key_bindings is needed as the Viewer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
import napari.components._viewer_key_bindings as _viewer_key_bindings

del _viewer_key_bindings

_submod_attrs = {
    'camera': ['Camera'],
    'dims': ['Dims'],
    'layerlist': ['LayerList'],
    'viewer_model': ['ViewerModel'],
}

_proto_all_ = []

__getattr__, __dir__, __all__ = attach(
    __name__, submodules=_proto_all_, submod_attrs=_submod_attrs
)
