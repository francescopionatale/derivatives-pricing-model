from pathlib import Path

from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks


def test_main_uses_installable_package_import():
    main_text = Path('main.py').read_text()
    assert 'from quant_derivatives.cli.main import main' in main_text
    assert 'src.quant_derivatives' not in main_text


def test_black_scholes_supports_zero_maturity():
    call = bs_price_and_greeks(105.0, 100.0, 0.0, 0.05, 0.2, True)
    put = bs_price_and_greeks(95.0, 100.0, 0.0, 0.05, 0.2, False)
    assert call['price'] == 5.0
    assert put['price'] == 5.0
    assert call['gamma'] == 0.0
    assert put['vega'] == 0.0
