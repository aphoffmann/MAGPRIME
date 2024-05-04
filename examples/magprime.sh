#!/bin/bash
pip install git+https://github.com/aphoffmann/MAGPRIME.git
echo "Running simulation_A.py..."
python simulation_A.py > output_A.txt

echo "Running simulation_B.py..."
python simulation_B.py > output_B.txt

echo "Running simulation_C.py..."
python simulation_C.py > output_C.txt

echo "All simulations completed."