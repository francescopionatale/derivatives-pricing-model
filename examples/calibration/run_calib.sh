#!/bin/bash
# Assuming quotes.csv exists in the current directory
# echo "strike,maturity,mid_price,cp_flag" > quotes.csv
# echo "100,1.0,10.45,C" >> quotes.csv
python main.py calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05
