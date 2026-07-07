from __future__ import annotations

import sys
import tempfile
import traceback
import warnings
from pathlib import Path

from common.terminal_logging import tee_terminal


def test_terminal_tee_captures_stdout_stderr_warning_and_traceback() -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with tempfile.TemporaryDirectory(prefix="slide_terminal_log_") as temp_dir:
        log_path = Path(temp_dir) / "train_terminal.txt"
        with tee_terminal(log_path):
            print("stdout marker")
            print("stderr marker", file=sys.stderr)
            warnings.warn("warning marker", RuntimeWarning)
            try:
                raise RuntimeError("traceback marker")
            except RuntimeError:
                traceback.print_exc(file=sys.stderr)

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        contents = log_path.read_text(encoding="utf-8")
        for marker in ("stdout marker", "stderr marker", "warning marker", "traceback marker"):
            assert marker in contents
