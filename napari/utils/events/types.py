from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    # note that if this gets *any* object with an `events` attribute, it will pass
    # always because type hints are not checked!
    # this happened with some dask objects, so this is a weak spot in some cases
    events: EmitterGroup


@runtime_checkable
class EventedMutable(SupportsEvents, Protocol):
    _parent: Optional[EventedMutable]
    _parent_key: Optional[str]

    def _update_inplace(self, other: Any) -> None:
        """
        Update inplace the contents of the EventedMutable to match `other`.
        """
