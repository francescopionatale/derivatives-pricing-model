import numpy as np
from quant_derivatives.engines.calibration.surface import check_no_arbitrage

def test_arbitrage_detection():
    strikes = np.array([90, 100, 110])
    maturities = np.array([1.0, 1.0, 1.0])
    # Call prices should be decreasing with strike
    prices = np.array([15.0, 10.0, 12.0]) # 12.0 > 10.0 is an arbitrage
    
    issues = check_no_arbitrage(strikes, maturities, prices, is_call=True)
    assert len(issues) > 0
    assert "Call price increasing" in issues[0]
