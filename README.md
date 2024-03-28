# Test scripts

## Matmul

```
python -m test.test_matmul.test_matmul --simgpu
```

## Allreduce
```
python -m test.test_allreduce.plot_allreduce_gpu
```

Bandwidth can be modified at `hardware_model/interconnect.py`

# Scale-sim
For scale-sim to work, try to first install numpy 1.19.0 before installing scale-sim. Scale-sim will raise some `np.int` error with numpy>=1.20 as in this issue https://github.com/scalesim-project/scale-sim-v2/issues/66.