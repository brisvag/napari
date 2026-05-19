import gc
import sys
from unittest import mock

import pytest

import napari
from napari import __main__


@pytest.fixture
def mock_run():
    """mock to prevent starting the event loop."""
    with (
        mock.patch('napari.run'),
    ):
        yield napari.run


@pytest.fixture
def stub_viewer():
    qt_viewer = mock.Mock()
    qt_viewer._qt_open = mock.Mock()
    window = mock.Mock()
    window._qt_viewer = qt_viewer
    viewer = mock.Mock()
    viewer._window = window
    viewer.window = window
    return viewer


def test_cli_works(monkeypatch, capsys):
    """Test the cli runs and shows help"""
    monkeypatch.setattr(sys, 'argv', ['napari', '-h'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert 'napari command line viewer.' in str(capsys.readouterr())


def test_cli_shows_plugins(monkeypatch, capsys, tmp_plugin):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--info'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert tmp_plugin.name in str(capsys.readouterr())


def test_cli_parses_unknowns(mock_run, monkeypatch, stub_viewer):
    """test that we can parse layer keyword arg variants"""
    v = stub_viewer  # our mock view_path will return this object

    # testing all the variants of literal_evals
    with mock.patch('napari.__main__.Viewer', return_value=v):
        with monkeypatch.context() as m:
            m.setattr(
                sys, 'argv', ['n', 'file', '--contrast-limits', '(0, 1)']
            )
            __main__._run()
        v._window._qt_viewer._qt_open.assert_called_once_with(
            ['file'],
            stack=[],
            plugin=None,
            layer_type=None,
            contrast_limits=(0, 1),
        )
        v._window._qt_viewer._qt_open.reset_mock()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0,1)'])
            __main__._run()
        v._window._qt_viewer._qt_open.assert_called_once_with(
            ['file'],
            stack=[],
            plugin=None,
            layer_type=None,
            contrast_limits=(0, 1),
        )
        v._window._qt_viewer._qt_open.reset_mock()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0, 1)'])
            __main__._run()
        v._window._qt_viewer._qt_open.assert_called_once_with(
            ['file'],
            stack=[],
            plugin=None,
            layer_type=None,
            contrast_limits=(0, 1),
        )
        v._window._qt_viewer._qt_open.reset_mock()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0,1)'])
            __main__._run()
        v._window._qt_viewer._qt_open.assert_called_once_with(
            ['file'],
            stack=[],
            plugin=None,
            layer_type=None,
            contrast_limits=(0, 1),
        )


def test_cli_raises(monkeypatch):
    """test that unknown kwargs raise the correct errors."""
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--nonsense'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: unrecognized argument: --nonsense'

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--gamma'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: argument --gamma expected one argument'


@pytest.mark.usefixtures('builtins')
def test_cli_runscript(monkeypatch, tmp_path, make_napari_viewer):
    """Test that running napari script.py runs a script"""
    v = make_napari_viewer()
    script = tmp_path / 'test.py'
    script.write_text('import napari; v = napari.Viewer(); v.add_points([])')

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', str(script)])
        m.setattr(__main__, 'Viewer', mock.Mock(return_value=v))
        m.setattr(
            'qtpy.QtWidgets.QApplication.exec_', lambda *_: None
        )  # revent event loop if run this test standalone
        __main__._run()

    assert len(v.layers) == 1


def test_cli_passes_kwargs(mock_run, monkeypatch, stub_viewer):
    """test that we can parse layer keyword arg variants"""
    v = stub_viewer

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(sys, 'argv', ['n', 'file', '--name', 'some name'])
        __main__._run()

    v._window._qt_viewer._qt_open.assert_called_once_with(
        ['file'],
        stack=[],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_passes_kwargs_stack(mock_run, monkeypatch, stub_viewer):
    """test that we can parse layer keyword arg variants"""
    v = stub_viewer

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(
            sys,
            'argv',
            [
                'n',
                'file',
                '--stack',
                'file1',
                'file2',
                '--stack',
                'file3',
                'file4',
                '--name',
                'some name',
            ],
        )
        __main__._run()

    v._window._qt_viewer._qt_open.assert_called_once_with(
        ['file'],
        stack=[['file1', 'file2'], ['file3', 'file4']],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_retains_viewer_ref(mock_run, monkeypatch, stub_viewer):
    """Test that napari.__main__ is retaining a reference to the viewer."""
    v = stub_viewer  # our mock view_path will return this object
    ref_count = None  # counter that will be updated before __main__._run()

    def _check_refs(**kwargs):
        # when run() is called in napari.__main__, we will call this function
        # it forces garbage collection, and then makes sure that at least one
        # additional reference to our viewer exists.
        gc.collect()
        if sys.getrefcount(v) <= ref_count:  # pragma: no cover
            raise AssertionError(
                'Reference to napari.viewer has been lost by '
                'the time the event loop started in napari.__main__'
            )

    mock_run.side_effect = _check_refs
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file.tif'])
        # return our local v
        with mock.patch('napari.__main__.Viewer', return_value=v) as mock_viewer:
            ref_count = sys.getrefcount(v)  # count current references
            __main__._run()
            mock_viewer.assert_called_once()
            v._window._qt_viewer._qt_open.assert_called_once()


def test_cli_plugin_info(monkeypatch):
    """--plugin-info delegates to npe2.cli.list_ and exits."""
    monkeypatch.setattr(sys, 'argv', ['napari', '--plugin-info'])
    with (
        mock.patch('npe2.cli.list_') as mock_list,
        pytest.raises(SystemExit),
    ):
        __main__._run()
    mock_list.assert_called_once()
