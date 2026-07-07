from __future__ import annotations

import sys
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream:
    def __init__(self, terminal: TextIO, log: TextIO, lock: threading.Lock):
        self.terminal = terminal
        self.log = log
        self.lock = lock

    def write(self, value: str) -> int:
        with self.lock:
            terminal_count = self.terminal.write(value)
            self.log.write(value)
            if "\n" in value:
                self.terminal.flush()
                self.log.flush()
        return terminal_count

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding

    @property
    def errors(self) -> str | None:
        return self.terminal.errors


def terminal_log_path(console_dir: str | Path, prefix: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(console_dir).expanduser().resolve() / f"{prefix}_{timestamp}.txt"


@contextmanager
def tee_terminal(path: str | Path) -> Iterator[Path]:
    """Mirror Python stdout and stderr to one line-buffered run log."""
    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_showwarning = warnings.showwarning
    lock = threading.Lock()

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = _TeeStream(original_stdout, log, lock)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log, lock)  # type: ignore[assignment]

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            target = sys.stderr if file is None else file
            target.write(warnings.formatwarning(message, category, filename, lineno, line))

        warnings.showwarning = showwarning
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            warnings.showwarning = original_showwarning
            sys.stdout = original_stdout
            sys.stderr = original_stderr
