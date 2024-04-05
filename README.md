[![DOI](https://zenodo.org/badge/779008229.svg)](https://zenodo.org/doi/10.5281/zenodo.10892431)

# LLMCompass

## Set up the environment

```
$ conda create -n llmcompass_ae python=3.9
$ conda activate llmcompass_ae
$ pip3 install scalesim
$ conda install pytorch==2.0.0 -c pytorch
$ pip3 install matplotlib
$ pip3 install seaborn
$ pip3 install scipy
```

## Installation

```
git clone https://github.com/HenryChang213/LLMCompass_ISCA_AE.git
```

## Experiment workflow
```
# Figure 5 (around 100 min) 
$ cd ae/figure5
$ bash run_figure5.sh 

# Figure 6 (around 1 min)
$ cd ae/figure6
$ bash run_figure6.sh

# Figure 7 (around 20 min)
$ cd ae/figure7
$ bash run_figure7.sh

# Figure 8 (around 40 min)
$ cd ae/figure8
$ bash run_figure8.sh

# Figure 9 (around 30 min)
$ cd ae/figure9
$ bash run_figure9.sh

# Figure 10 (around 45 min)
$ cd ae/figure10
$ bash run_figure10.sh

# Figure 11 (around 5 min) 
$ cd ae/figure11
$ bash run_figure11.sh

# Figure 12 (around 4 hours) 
$ cd ae/figure12
$ bash run_figure12.sh
```
