rm A100/*.csv
rm our/*.csv
rm *.pdf

cd ../..

python -m ae.figure12.test_throughput

cd ae/figure12
python plot_throughput.py