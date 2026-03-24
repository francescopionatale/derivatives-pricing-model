#!/bin/bash
python main.py lookback-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --lookback-style floating --M 5000 --n-steps 126 --seed 42 --antithetic
