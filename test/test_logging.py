from __future__ import annotations

import pytest
from rich.logging import RichHandler

from greenonet.logging_mixin import LoggingMixin


class _LoggingOwner(LoggingMixin):
    pass


def _rich_handler(owner: _LoggingOwner) -> RichHandler:
    for handler in owner.logger.handlers:
        if isinstance(handler, RichHandler):
            return handler
    raise AssertionError("LoggingMixin did not install a RichHandler.")


def test_logging_mixin_uses_fixed_terminal_width(tmp_path):
    owner = _LoggingOwner(
        logger_name="test_logging_mixin_uses_fixed_terminal_width",
        work_dir=tmp_path,
        terminal_width=250,
    )

    handler = _rich_handler(owner)

    assert handler.console.width == 250
    assert getattr(handler.console, "_width") == 250
    assert (tmp_path / "training.log").exists()


def test_logging_mixin_preserves_auto_terminal_width(tmp_path):
    owner = _LoggingOwner(
        logger_name="test_logging_mixin_preserves_auto_terminal_width",
        work_dir=tmp_path,
        terminal_width=None,
    )

    handler = _rich_handler(owner)

    assert getattr(handler.console, "_width") is None
    assert (tmp_path / "training.log").exists()


def test_logging_mixin_rejects_non_positive_terminal_width(tmp_path):
    with pytest.raises(ValueError, match="terminal.width"):
        _LoggingOwner(
            logger_name="test_logging_mixin_rejects_non_positive_terminal_width",
            work_dir=tmp_path,
            terminal_width=0,
        )
