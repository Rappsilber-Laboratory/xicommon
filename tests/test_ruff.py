import subprocess
import pytest


def test_ruff():
    """
    Run ruff via subprocess and assert it passes.
    This replaces the deprecated pytest-pydocstyle plugin.
    """
    # We only select pydocstyle rules (D) to replace pytest-pydocstyle.
    ignored_rules = [
        # Missing docstrings
        "D100", "D101", "D102", "D103", "D104", "D105", "D106", "D107",
        # Formatting and style
        "D200", "D202", "D203", "D204", "D205", "D209", "D210", "D212", "D213",
        "D400", "D401", "D403", "D404", "D406", "D407", "D411", "D413", "D415", "D417"
    ]

    cmd = [
        "ruff", "check", ".",
        "--select", "D",
        "--ignore", ",".join(ignored_rules)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Ruff checks failed:\n\n{result.stdout}\n{result.stderr}")
