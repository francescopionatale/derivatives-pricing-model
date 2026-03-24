import json
from pathlib import Path

from quant_derivatives.engines.calibration.heston import calibrate_heston_to_quotes
from quant_derivatives.io.loaders import load_quotes_csv
from quant_derivatives.utils.heston_params import load_heston_params_json


ROOT = Path(__file__).resolve().parents[2]


def test_heston_calibration_returns_parameter_payload():
    quotes = load_quotes_csv(str(ROOT / "examples" / "heston" / "synthetic_heston_quotes.csv"))
    result = calibrate_heston_to_quotes(
        quotes=quotes,
        S0=100.0,
        r=0.03,
        n_steps=16,
        n_paths=400,
        seed=123,
        antithetic=True,
        maxiter=4,
    ).to_dict()

    assert set(result["params"].keys()) == {"kappa", "theta", "sigma_v", "rho", "v0"}
    assert result["n_quotes"] == len(quotes)
    assert result["rmse_price"] < 3.0
    assert "per_quote_pricing" in result["diagnostics"]



def test_load_heston_params_json_reads_reference_payload():
    params = load_heston_params_json(str(ROOT / "examples" / "heston" / "reference_params.json"))
    assert params["kappa"] > 0.0
    assert -1.0 < params["rho"] < 1.0
