#!/bin/bash
python main.py barrier-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --barrier 120 --direction up --barrier-style out --M 5000 --n-steps 126 --seed 42 --antithetic
