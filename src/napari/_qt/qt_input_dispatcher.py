from __future__ import annotations

from typing import Callable

import numpy as np
from app_model.backends.qt import qkeysequence2modelkeybinding
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QKeySequence

from napari.utils.input_events import (
    InputEventEmitter,
    NapariKeyEvent,
    NapariMouseEvent,
)

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

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
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

