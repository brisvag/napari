import contextlib


class _ParentValidatorMixin:
    def __init__(self, *args, **kwargs):
        self._parent = None
        self._parent_key = None
        self._do_validation = True
        super().__init__(*args, **kwargs)

    def _validate(self, new_values):
        if self._parent is None or not self._do_validation:
            return new_values

        if self._parent_key is not None:
            return self._parent._validate({self._parent_key: new_values})[
                self._parent_key
            ]
        else:
            raise ValueError('parented evented objects must set _parent_key')

    @contextlib.contextmanager
    def _no_validation(self):
        self._do_validation = False
        yield
        self._do_validation = True
