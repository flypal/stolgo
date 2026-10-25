"""Tests for stolgo.cli.main."""

import os
import subprocess
import sys
from pathlib import Path


def test_cli_help():
    repo = Path(__file__).resolve().parents[1]
    python = repo / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    proc = subprocess.run(
        [str(python), "-m", "stolgo.cli.main", "--help"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "lib")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "backtest" in proc.stdout.lower()
