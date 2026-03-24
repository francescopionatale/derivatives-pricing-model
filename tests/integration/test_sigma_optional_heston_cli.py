import subprocess
import sys
from pathlib import Path


def test_heston_cli_accepts_params_json_without_sigma(tmp_path):
    params_path = tmp_path / 'params.json'
    params_path.write_text('{"params": {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7, "v0": 0.04}}')
    result = subprocess.run(
        [
            sys.executable,
            'main.py',
            'heston-price',
            '--S0', '100',
            '--K', '100',
            '--T', '1.0',
            '--r', '0.03',
            '--params-json', str(params_path),
            '--M', '200',
            '--n-steps', '16',
            '--seed', '7',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
