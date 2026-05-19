import pytest

from napari._tests.utils import layer_test_data
from napari.settings import get_settings
from napari.utils.action_manager import action_manager


def _get_provider_actions(type_):
    actions = set()
    for superclass in type_.mro():
        actions.update(
            action.command
            for action in action_manager._get_provider_actions(
                superclass
            ).values()
        )
    return actions


def _assert_shortcuts_exist_for_each_action(type_):
    actions = _get_provider_actions(type_)
    shortcuts = {
        name.partition(':')[-1] for name in get_settings().shortcuts.shortcuts
    }
    shortcuts.update(func.__name__ for func in type_.class_keymap.values())
    for action in actions:
        assert (
            action.__name__ in shortcuts
        ), f"missing shortcut for action '{action.__name__}' on '{type_.__name__}' is missing"


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_all_layer_actions_are_accessible_via_shortcut(
    layer_class, data, ndim
):
    """
    Make sure we do find all the actions attached to a layer via keybindings
    """
    # instantiate to make sure everything is initialized correctly
    _ = layer_class(data)
    _assert_shortcuts_exist_for_each_action(layer_class)
