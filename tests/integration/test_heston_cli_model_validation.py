import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_heston_price_rejects_model_flag():
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "heston-price",
            "--model", "gbm",
            "--S0", "100",
            "--K", "100",
            "--T", "1.0",
            "--r", "0.03",
            "--M", "200",
            "--n-steps", "16",
            "--params-json", str(ROOT / "examples" / "heston" / "reference_params.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --model gbm" in result.stderr
