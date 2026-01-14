from lazy_loader import attach

_submod_attrs = {
    'colormaps': ['Colormap', 'CyclicLabelColormap', 'DirectLabelColormap'],
    'info': ['citation_text', 'sys_info'],
    'notebook_display': ['NotebookScreenshot', 'nbscreenshot'],
    'progress': ['cancelable_progress', 'progrange', 'progress'],
}

_proto_all_ = []

__getattr__, __dir__, __all__ = attach(
    __name__, submodules=_proto_all_, submod_attrs=_submod_attrs
)
