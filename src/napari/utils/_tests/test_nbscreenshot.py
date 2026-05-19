import html
from unittest.mock import Mock

import pytest

from napari._tests.utils import skip_on_win_ci
from napari.utils import nbscreenshot


@skip_on_win_ci
@pytest.mark.parametrize(
    "alt_text_input, expected_alt_text",
    [
        (None, None),
        ("Good alt text", "Good alt text"),
        # Naughty strings https://github.com/minimaxir/big-list-of-naughty-strings
        # ASCII punctuation
        (r",./;'[]\-=", ',./;&#x27;[]\\-='),
        # ASCII punctuation 2, skipping < because that is interpreted as the start
        # of an HTML element.
        ('>?:"{}|_+', '&gt;?:&quot;{}|_+'),
        ("!@#$%^&*()`~", '!@#$%^&amp;*()`~'),  # ASCII punctuation 3
        # # Emojis
        ("😍", "😍"),  # emoji 1
        (
            "👨‍🦰 👨🏿‍🦰 👨‍🦱 👨🏿‍🦱 🦹🏿‍♂️",
            "👨‍🦰 👨🏿‍🦰 👨‍🦱 👨🏿‍🦱 🦹🏿‍♂️",
        ),  # emoji 2
        (r"¯\_(ツ)_/¯", '¯\\_(ツ)_/¯'),  # Japanese emoticon
        # # Special characters
        ("田中さんにあげて下さい", "田中さんにあげて下さい"),  # two-byte characters
        (
            "表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀",  # noqa: RUF001
            "表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀",  # noqa: RUF001
        ),  # special unicode chars
        ("گچپژ", "گچپژ"),  # Persian special characters
        # # Script injection
        ("<script>alert(0)</script>", None),  # script injection 1
        ("&lt;script&gt;alert(&#39;1&#39;);&lt;/script&gt;", None),
        ("<svg><script>123<1>alert(3)</script>", None),
    ],
)
def test_safe_alt_text(alt_text_input, expected_alt_text):
    display_obj = nbscreenshot(Mock(), alt_text=alt_text_input)
    if not expected_alt_text:
        assert not display_obj.alt_text
    else:
        assert html.escape(display_obj.alt_text) == expected_alt_text


@skip_on_win_ci
def test_invalid_alt_text():
    with pytest.warns(
        UserWarning,
        match='The provided alt text does not constitute valid html',
    ):
        # because string with only whitespace messes up with the parser
        display_obj = nbscreenshot(Mock(), alt_text=" ")
    assert display_obj.alt_text is None

    with pytest.warns(
        UserWarning,
        match='The provided alt text does not constitute valid html',
    ):
        # because string with only whitespace messes up with the parser
        display_obj = nbscreenshot(Mock(), alt_text="")
    assert display_obj.alt_text is None
