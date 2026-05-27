from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from app_model.types import KeyBinding
from psygnal import Signal


@dataclass
class InputEvent:
    type: str
    native: Any | None = None
    handled: bool = False


@dataclass
class NapariKeyEvent(InputEvent):
    key: KeyBinding | None = None
    modifiers: tuple[str, ...] = ()
    text: str | None = None
    is_auto_repeat: bool = False


@dataclass
class NapariMouseEvent(InputEvent):
    pos: npt.NDArray[np.float64] | None = None
    button: int | None = None
    buttons: tuple[int, ...] = ()
    modifiers: tuple[str, ...] = ()
    delta: npt.NDArray[np.float64] | None = None
    last_event: NapariMouseEvent | None = None
    press_event: NapariMouseEvent | None = None
    view_direction: npt.NDArray[np.float64] | None = None
    up_direction: npt.NDArray[np.float64] | None = None
    camera_zoom: float | None = None
    position: tuple[float, ...] | None = None
    dims_displayed: list[int] = field(default_factory=list)
    dims_point: list[float] = field(default_factory=list)
    viewbox: tuple[int, int] | None = None

    @property
    def is_dragging(self) -> bool:
        return self.press_event is not None and bool(self.buttons)

    def __post_init__(self) -> None:
        if self.pos is not None and not isinstance(self.pos, np.ndarray):
            self.pos = np.asarray(self.pos, dtype=float)
        if self.delta is not None and not isinstance(self.delta, np.ndarray):
            self.delta = np.asarray(self.delta, dtype=float)


class InputEventEmitter:
    mouse_press = Signal(object)
    mouse_release = Signal(object)
    mouse_move = Signal(object)
    mouse_double_click = Signal(object)
    mouse_wheel = Signal(object)
    key_press = Signal(object)
    key_release = Signal(object)
