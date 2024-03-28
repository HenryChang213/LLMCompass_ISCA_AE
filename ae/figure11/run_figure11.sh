rm *.csv
rm *.pdf

cd ../..

python -m ae.figure11.test_decoding

# cd ae/figure10_11
# python plot_latency.py