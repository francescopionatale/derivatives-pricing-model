import numpy as np
from engines.pricing.pde import bs_pde_price

from engines.pricing.black_scholes import bs_price_and_greeks


class TestBSPDE:
    BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

    def test_call_matches_bs(self):
        # CN PDE price should match BS analytical within 0.05 (ATM 1Y call)
        pde = bs_pde_price(**self.BASE, is_call=True, N_x=300, N_t=200)
        bs_params = {**self.BASE, "S": self.BASE["S0"]}
        bs_params.pop("S0")
        bs = bs_price_and_greeks(**bs_params, is_call=True)
        assert abs(pde["price"] - bs["price"]) < 0.05

    def test_put_matches_bs(self):
        pde = bs_pde_price(**self.BASE, is_call=False, N_x=300, N_t=200)
        bs_params = {**self.BASE, "S": self.BASE["S0"]}
        bs_params.pop("S0")
        bs = bs_price_and_greeks(**bs_params, is_call=False)
        assert abs(pde["price"] - bs["price"]) < 0.05

    def test_put_call_parity(self):
        call = bs_pde_price(**self.BASE, is_call=True, N_x=200, N_t=100)
        put = bs_pde_price(**self.BASE, is_call=False, N_x=200, N_t=100)
        parity = self.BASE["S0"] - self.BASE["K"] * np.exp(-self.BASE["r"] * self.BASE["T"])
        assert abs((call["price"] - put["price"]) - parity) < 0.05

    def test_returns_dict_keys(self):
        result = bs_pde_price(**self.BASE)
        for key in ("price", "delta", "gamma", "grid_S", "grid_V", "method", "settings"):
            assert key in result

    def test_delta_in_range(self):
        # Call delta in (0,1), put delta in (-1,0)
        call = bs_pde_price(**self.BASE, is_call=True)
        put = bs_pde_price(**self.BASE, is_call=False)
        assert 0 < call["delta"] < 1
        assert -1 < put["delta"] < 0

    def test_convergence_with_grid_size(self):
        # Coarser grid should be less accurate than finer grid (monotone convergence)
        bs_params = {**self.BASE, "S": self.BASE["S0"]}
        bs_params.pop("S0")
        bs_ref = bs_price_and_greeks(**bs_params)["price"]
        err_coarse = abs(bs_pde_price(**self.BASE, N_x=50, N_t=50)["price"] - bs_ref)
        err_fine = abs(bs_pde_price(**self.BASE, N_x=300, N_t=200)["price"] - bs_ref)
        assert err_fine < err_coarse

    def test_otm_call_small(self):
        # Deep OTM call should be close to 0
        result = bs_pde_price(S0=100, K=200, T=0.5, r=0.05, sigma=0.2, is_call=True)
        assert result["price"] < 0.1

    def test_deep_itm_call(self):
        # Deep ITM call ≈ S - K*exp(-rT)
        result = bs_pde_price(S0=200, K=100, T=1.0, r=0.05, sigma=0.2, is_call=True, N_x=300, N_t=200)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert abs(result["price"] - intrinsic) < 1.0


class TestAmericanPDE:
    def test_american_put_exceeds_european(self):
        # American put >= European put
        params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        eur = bs_pde_price(**params, is_call=False, is_american=False, N_x=200, N_t=100)
        amer = bs_pde_price(**params, is_call=False, is_american=True, N_x=200, N_t=100)
        assert amer["price"] >= eur["price"] - 0.01  # small tolerance for numerical error

    def test_american_call_no_div_equals_european(self):
        # For a call without dividends, American == European (no early exercise benefit)
        params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        eur = bs_pde_price(**params, is_call=True, is_american=False, N_x=200, N_t=100)
        amer = bs_pde_price(**params, is_call=True, is_american=True, N_x=200, N_t=100)
        assert abs(amer["price"] - eur["price"]) < 0.05

    def test_american_put_positive(self):
        result = bs_pde_price(S0=100, K=110, T=1.0, r=0.05, sigma=0.2, is_call=False, is_american=True)
        assert result["price"] > 0
