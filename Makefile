.PHONY: test test-unit test-integration lint run-bs run-barrier run-lookback run-hedge run-hedge-heston run-heston-price run-calib run-calib-heston run-barrier-calibrated run-lookback-calibrated run-opt clean

test: test-unit test-integration

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

lint:
	python -m compileall -q src tests
	PYTHONPATH=src python -c "import quant_derivatives; print(quant_derivatives.__name__)"
	pytest --collect-only -q >/dev/null

run-bs:
	python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2

run-barrier:
	python main.py barrier-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --barrier 120 --direction up --barrier-style out --M 5000 --n-steps 126 --seed 42 --antithetic

run-lookback:
	python main.py lookback-price --model heston --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --lookback-style floating --M 5000 --n-steps 126 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --seed 42

run-heston-price:
	python main.py heston-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --M 5000 --n-steps 126 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --seed 42

run-hedge:
	python main.py hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --seed 42

run-hedge-heston:
	python main.py hedge-sim --model heston --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --seed 42

run-calib:
	python main.py calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05

run-calib-heston:
	python main.py calibrate-heston --input-csv examples/heston/synthetic_heston_quotes.csv --S0 100 --r 0.03 --M 2500 --n-steps 32 --maxiter 10 --antithetic --output-json examples/heston/calibrated_params.json

run-barrier-calibrated:
	python main.py barrier-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --barrier 120 --direction up --barrier-style out --M 5000 --n-steps 126 --seed 42

run-lookback-calibrated:
	python main.py lookback-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --lookback-style floating --M 5000 --n-steps 126 --seed 42

run-opt:
	python main.py optimize-risk --input-json examples/risk/portfolio_case.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
