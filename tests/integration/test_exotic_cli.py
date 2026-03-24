import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_barrier_cli_runs():
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "barrier-price",
            "--S0", "100",
            "--K", "100",
            "--T", "1.0",
            "--r", "0.05",
            "--sigma", "0.2",
            "--barrier", "120",
            "--direction", "up",
            "--barrier-style", "out",
            "--M", "500",
            "--n-steps", "32",
            "--seed", "42",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Barrier option" in result.stderr or "Barrier option" in result.stdout



def test_lookback_cli_runs_under_heston():
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "lookback-price",
            "--model", "heston",
            "--S0", "100",
            "--K", "100",
            "--T", "1.0",
            "--r", "0.05",
            "--sigma", "0.2",
            "--lookback-style", "floating",
            "--M", "500",
            "--n-steps", "32",
            "--kappa", "2.0",
            "--theta", "0.04",
            "--sigma-v", "0.3",
            "--rho", "-0.7",
            "--v0", "0.04",
            "--seed", "42",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Lookback option" in result.stderr or "Lookback option" in result.stdout
