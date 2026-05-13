from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class LoggingMixin:
    """Provides a configured logger that mirrors console output to a file."""

    def __init__(
        self,
        logger_name: Optional[str] = None,
        work_dir: Optional[Path] = None,
        terminal_width: int | None = None,
    ) -> None:
        if terminal_width is not None and terminal_width <= 0:
            raise ValueError("terminal.width must be positive or null.")
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)
        if self.logger.handlers:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()

        if terminal_width is not None:
            handler = RichHandler(
                console=Console(width=terminal_width),
                rich_tracebacks=True,
                show_path=True,
                omit_repeated_times=False,
            )
        else:
            handler = RichHandler(
                rich_tracebacks=True,
                show_path=True,
                omit_repeated_times=False,
            )
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.work_dir / "training.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logging.root.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
