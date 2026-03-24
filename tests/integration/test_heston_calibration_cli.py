import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_calibrate_heston_cli_writes_json(tmp_path):
    out = tmp_path / "calibrated_heston.json"
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "calibrate-heston",
            "--input-csv", "examples/heston/synthetic_heston_quotes.csv",
            "--S0", "100",
            "--r", "0.03",
            "--M", "400",
            "--n-steps", "16",
            "--maxiter", "3",
            "--antithetic",
            "--output-json", str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "params" in payload
    assert set(payload["params"].keys()) == {"kappa", "theta", "sigma_v", "rho", "v0"}
