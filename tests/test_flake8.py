import subprocess
import pytest


def test_flake8():
    """
    Run flake8 via subprocess and assert it passes.
    This replaces the deprecated pytest-flake8 plugin.
    """
    result = subprocess.run(['flake8'], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Flake8 formatting checks failed:\n\n{result.stdout}\n{result.stderr}")
