from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from app_model.backends.qt import qkeysequence2modelkeybinding
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QKeySequence
from superqt.utils import qthrottled

from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.input_events import (
    InputEventEmitter,
    NapariKeyEvent,
    NapariMouseEvent,
)
from napari.utils.interactions import (
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)

if TYPE_CHECKING:
    from napari.components import ViewerModel


class CanvasEventMapper(Protocol):
    def get_viewbox_at(
        self, position: np.ndarray | tuple[float, float]
    ) -> tuple[Any | None, tuple[int, int] | None]: ...

    def map_canvas_to_world(
        self, position: np.ndarray | tuple[float, float], viewbox: Any
    ) -> tuple[float, ...]: ...

    def calculate_view_direction(
        self, event_pos: tuple[float, float]
    ) -> np.ndarray | None: ...

_MODIFIER_MAP = (
    (Qt.ShiftModifier, 'Shift'),
    (Qt.ControlModifier, 'Control'),
    (Qt.AltModifier, 'Alt'),
    (Qt.MetaModifier, 'Meta'),
)

_BUTTON_MAP = {
    Qt.LeftButton: 1,
    Qt.RightButton: 2,
    Qt.MiddleButton: 3,
}

_MOUSE_EVENT_TYPES = {
    QEvent.Type.MouseButtonPress: 'mouse_press',
    QEvent.Type.MouseButtonRelease: 'mouse_release',
    QEvent.Type.MouseMove: 'mouse_move',
    QEvent.Type.MouseButtonDblClick: 'mouse_double_click',
}


def _qt_pos(event) -> np.ndarray | None:
    if hasattr(event, 'position'):
        pos = event.position()
    else:
        pos = event.pos()
    if pos is None:
        return None
    return np.array([pos.x(), pos.y()], dtype=float)


def _qt_modifiers(modifiers: Qt.KeyboardModifiers) -> tuple[str, ...]:
    return tuple(
        name for mask, name in _MODIFIER_MAP if modifiers & mask
    )


def _qt_button(button: Qt.MouseButton) -> int | None:
    if button in _BUTTON_MAP:
        return _BUTTON_MAP[button]
    return None


def _qt_buttons(buttons: Qt.MouseButtons) -> tuple[int, ...]:
    return tuple(
        mapped
        for qt_button, mapped in _BUTTON_MAP.items()
        if buttons & qt_button
    )


def _qt_wheel_delta(event) -> np.ndarray:
    delta = event.angleDelta()
    if delta.isNull():
        delta = event.pixelDelta()
    return np.array([delta.x(), delta.y()], dtype=float)


def qt_key_event_to_napari(event, event_type: str) -> NapariKeyEvent | None:
    key = event.key()
    if key in (
        Qt.Key.Key_unknown,
        Qt.Key.Key_Control,
        Qt.Key.Key_Shift,
        Qt.Key.Key_Alt,
        Qt.Key.Key_Meta,
    ):
        return None
    key_seq = QKeySequence(int(event.modifiers()) | int(key))
    if key_seq.isEmpty():
        return None
    key_binding = qkeysequence2modelkeybinding(key_seq)
    return NapariKeyEvent(
        type=event_type,
        key=key_binding,
        modifiers=_qt_modifiers(event.modifiers()),
        text=event.text() or None,
        is_auto_repeat=event.isAutoRepeat(),
        native=event,
    )


class QtInputDispatcher(QObject):
    def __init__(
        self,
        widget,
        events: InputEventEmitter,
        *,
        on_enter: Callable[[], None] | None = None,
        on_leave: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(widget)
        self._events = events
        self._last_mouse_event: NapariMouseEvent | None = None
        self._press_event: NapariMouseEvent | None = None
        self._pressed_buttons: set[int] = set()
        self._on_enter = on_enter
        self._on_leave = on_leave
        widget.installEventFilter(self)

    def emit_mouse_event(
        self,
        event_type: str,
        *,
        pos,
        modifiers=(),
        button: int | None = None,
        buttons: tuple[int, ...] | None = None,
        delta: np.ndarray | None = None,
        native=None,
    ) -> NapariMouseEvent:
        if pos is not None:
            pos = np.asarray(pos, dtype=float)
        modifiers = tuple(modifiers)
        if buttons is None:
            if event_type == 'mouse_press':
                buttons = (button,) if button else ()
            elif event_type == 'mouse_release':
                buttons = ()
            else:
                buttons = tuple(self._pressed_buttons)
        if event_type == 'mouse_press' and button:
            self._pressed_buttons.add(button)
        elif event_type == 'mouse_release' and button:
            self._pressed_buttons.discard(button)
        press_event = None if event_type == 'mouse_press' else self._press_event
        event = NapariMouseEvent(
            type=event_type,
            pos=pos,
            button=button,
            buttons=buttons,
            modifiers=modifiers,
            delta=delta,
            last_event=self._last_mouse_event,
            press_event=press_event,
            native=native,
        )
        if event_type == 'mouse_press':
            self._press_event = event
        elif event_type == 'mouse_release' and not self._pressed_buttons:
            self._press_event = None
        self._last_mouse_event = event
        getattr(self._events, event_type).emit(event)
        return event

    def eventFilter(self, obj, event) -> bool:
        event_type = event.type()
        if event_type == QEvent.Type.Enter:
            if self._on_enter:
                self._on_enter()
            return False
        if event_type == QEvent.Type.Leave:
            if self._on_leave:
                self._on_leave()
            return False
        if event_type == QEvent.Type.Wheel:
            modifiers = _qt_modifiers(event.modifiers())
            if modifiers:
                return True
            napari_event = self.emit_mouse_event(
                'mouse_wheel',
                pos=_qt_pos(event),
                modifiers=modifiers,
                delta=_qt_wheel_delta(event),
                native=event,
            )
            return napari_event.handled
        if event_type not in _MOUSE_EVENT_TYPES:
            return False
        napari_event = self.emit_mouse_event(
            _MOUSE_EVENT_TYPES[event_type],
            pos=_qt_pos(event),
            modifiers=_qt_modifiers(event.modifiers()),
            button=_qt_button(event.button()),
            buttons=_qt_buttons(event.buttons()),
            native=event,
        )
        return napari_event.handled


class QtMouseEventHandler:
    def __init__(self, viewer: ViewerModel, canvas: CanvasEventMapper) -> None:
        self._viewer = viewer
        self._canvas = canvas
        self._mouse_move_handler = qthrottled(self._on_mouse_move, timeout=5)

    def connect(self, events: InputEventEmitter) -> None:
        events.mouse_double_click.connect(self._on_mouse_double_click)
        events.mouse_move.connect(self._mouse_move_handler)
        events.mouse_press.connect(self._on_mouse_press)
        events.mouse_release.connect(self._on_mouse_release)
        events.mouse_wheel.connect(self._on_mouse_wheel)

    def disconnect(self, events: InputEventEmitter) -> None:
        events.mouse_double_click.disconnect(self._on_mouse_double_click)
        events.mouse_move.disconnect(self._mouse_move_handler)
        events.mouse_press.disconnect(self._on_mouse_press)
        events.mouse_release.disconnect(self._on_mouse_release)
        events.mouse_wheel.disconnect(self._on_mouse_wheel)

    def _process_mouse_event(
        self, mouse_callbacks: Callable, event: NapariMouseEvent
    ) -> None:
        if event.pos is None:
            return

        if event.press_event is not None:
            viewbox, grid_coords = self._canvas.get_viewbox_at(
                event.press_event.pos
            )
        else:
            viewbox, grid_coords = self._canvas.get_viewbox_at(event.pos)

        self._viewer.cursor.viewbox = grid_coords

        if viewbox is None:
            event.handled = True
            return

        event.view_direction = self._canvas.calculate_view_direction(event.pos)
        event.up_direction = self._viewer.camera.calculate_nd_up_direction(
            self._viewer.dims.ndim, self._viewer.dims.displayed
        )
        event.camera_zoom = self._viewer.camera.zoom
        event.position = self._canvas.map_canvas_to_world(event.pos, viewbox)
        event.dims_displayed = list(self._viewer.dims.displayed)
        event.dims_point = list(self._viewer.dims.point)
        event.viewbox = grid_coords

        self._viewer.cursor._view_direction = event.view_direction
        self._viewer.cursor.position = event.position

        read_only_event = ReadOnlyWrapper(event, exceptions=('handled',))
        mouse_callbacks(self._viewer, read_only_event)

        layer = self._viewer.layers.selection.active
        if layer is not None:
            mouse_callbacks(layer, read_only_event)

        event.handled = read_only_event.handled

    def _on_mouse_double_click(self, event: NapariMouseEvent) -> None:
        self._process_mouse_event(mouse_double_click_callbacks, event)

    def _on_mouse_move(self, event: NapariMouseEvent) -> None:
        self._process_mouse_event(mouse_move_callbacks, event)

    def _on_mouse_press(self, event: NapariMouseEvent) -> None:
        self._process_mouse_event(mouse_press_callbacks, event)

    def _on_mouse_release(self, event: NapariMouseEvent) -> None:
        self._process_mouse_event(mouse_release_callbacks, event)

    def _on_mouse_wheel(self, event: NapariMouseEvent) -> None:
        self._process_mouse_event(mouse_wheel_callbacks, event)
