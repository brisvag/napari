import magicgui
import pytest
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget

import napari
from napari import Viewer, plugins
from napari.plugins import hook_specifications


class Widg1(QWidget):
    pass


class Widg2(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()


class Widg3(QWidget):
    def __init__(self, v: Viewer):
        self.viewer = v
        super().__init__()


def magicfunc(viewer: 'napari.Viewer'):
    return viewer


dwidget_args = {
    'single_class': Widg1,
    'class_tuple': (Widg1, {'area': 'right'}),
    'tuple_list': [(Widg1, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(Widg1, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (Widg1, 1),
    'bad_double_tuple': ((Widg1, {}), (Widg2, {})),
}


# test_plugin_manager and add_implementation fixtures are
#     provided by napari_plugin_engine._testsupport
# monkeypatch, request, recwarn fixtures are from pytest
@pytest.mark.parametrize('arg', dwidget_args.values(), ids=dwidget_args.keys())
def test_dock_widget_registration(
    arg, test_plugin_manager, add_implementation, monkeypatch, request, recwarn
):
    """Test that dock widgets get validated and registerd correctly."""
    test_plugin_manager.project_name = 'napari'
    test_plugin_manager.add_hookspecs(hook_specifications)
    hook = test_plugin_manager.hook.napari_experimental_provide_dock_widget

    with monkeypatch.context() as m:
        registered = {}
        m.setattr(plugins, "dock_widgets", registered)

        @napari_hook_implementation
        def napari_experimental_provide_dock_widget():
            return arg

        add_implementation(napari_experimental_provide_dock_widget)
        hook.call_historic(
            result_callback=plugins.register_dock_widget, with_impl=True
        )
        if '[bad_' in request.node.name:
            assert len(recwarn) == 1
            assert not registered
        else:
            assert len(recwarn) == 0
            assert registered[(None, 'Widg1')][0] == Widg1
            if 'tuple_list' in request.node.name:
                assert registered[(None, 'Widg2')][0] == Widg2


@pytest.fixture
def test_plugin_widgets(monkeypatch):
    """A smattering of example registered dock widgets and function widgets."""
    with monkeypatch.context() as m:
        dock_widgets = {
            ("TestP1", "Widg1"): (Widg1, {}),
            ("TestP1", "Widg2"): (Widg2, {}),
            ("TestP2", "Widg3"): (Widg3, {}),
        }
        m.setattr(plugins, "dock_widgets", dock_widgets)

        function_widgets = {
            ("TestP3", "magic"): (
                magicfunc,
                {'call_button': True},
                {'area': 'right'},
            ),
        }
        m.setattr(plugins, "function_widgets", function_widgets)
        yield


def test_plugin_widgets_menus(test_plugin_widgets, make_test_viewer):
    """Test the plugin widgets get added to the window menu correctly."""
    viewer = make_test_viewer()
    actions = viewer.window._plugin_dock_widget_menu.actions()
    assert len(actions) == 3
    expected_text = ['TestP1', 'TestP2: Widg3', 'TestP3: magic']
    assert [a.text() for a in actions] == expected_text

    # the first item in the menu is a submenu (for "Test plugin1")
    assert actions[0].menu()
    subnames = ['Widg1', 'Widg2']
    assert [a.text() for a in actions[0].menu().actions()] == subnames

    # the other items in the menu are not submenus
    assert not actions[1].menu()
    assert not actions[2].menu()


def test_making_plugin_dock_widgets(test_plugin_widgets, make_test_viewer):
    """Test that we can create dock widgets, and they get the viewer."""
    viewer = make_test_viewer()
    actions = viewer.window._plugin_dock_widget_menu.actions()

    # trigger the 'TestP2: Widg3' action
    actions[1].trigger()
    # make sure that a dock widget was created
    assert 'TestP2: Widg3' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP2: Widg3']
    assert isinstance(dw.widget(), Widg3)
    # This widget uses the parameter annotation method to receive a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Cannot add twice
    with pytest.warns(UserWarning):
        actions[1].trigger()

    # trigger the 'TestP1 > Widg2' action (it's in a submenu)
    action = actions[0].menu().actions()[1]
    assert action.text() == 'Widg2'
    action.trigger()
    # make sure that a dock widget was created
    assert 'TestP1: Widg2' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP1: Widg2']
    assert isinstance(dw.widget(), Widg2)
    # This widget uses parameter *name* "napari_viewer" to get a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Cannot add twice
    with pytest.warns(UserWarning):
        action.trigger()


def test_making_function_dock_widgets(test_plugin_widgets, make_test_viewer):
    """Test that we can create magicgui widgets, and they get the viewer."""
    viewer = make_test_viewer()
    actions = viewer.window._plugin_dock_widget_menu.actions()

    # trigger the 'TestP3: magic' action
    actions[2].trigger()
    # make sure that a dock widget was created
    assert 'TestP3: magic' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP3: magic']
    # make sure that it contains a magicgui widget
    magic_widget = dw.widget()._magic_widget
    assert isinstance(magic_widget, magicgui.FunctionGui)
    # This magicgui widget uses the parameter annotation to receive a viewer
    assert isinstance(magic_widget.viewer.value, napari.Viewer)
    # The function just returns the viewer... make sure we can call it
    assert isinstance(magic_widget(), napari.Viewer)
    # Cannot add twice
    with pytest.warns(UserWarning):
        actions[2].trigger()


def test_clear_all_plugin_widgets(test_plugin_widgets, make_test_viewer):
    """Test the the 'Remove Dock Widgets' menu item clears added widgets."""
    viewer = make_test_viewer()
    actions = viewer.window._plugin_dock_widget_menu.actions()
    actions[1].trigger()
    actions[0].menu().actions()[1].trigger()
    assert len(viewer.window._dock_widgets) == 2
    clear_action = next(
        a
        for a in viewer.window.window_menu.actions()
        if a.text().startswith("Remove Dock Widgets")
    )
    clear_action.trigger()
    assert len(viewer.window._dock_widgets) == 0
