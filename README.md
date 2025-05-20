## # MC-CP Library

This library is based on the original implementation of [MC-CP](https://github.com/team-daniel/MC-CP).

There is an error in the original repo while running the regression model:

```shell
Traceback (most recent call last):
  File "/home/shenghao/mccp_lib/run_mccp_bostonhousing.py", line 39, in <module>
    model = train_mqnn_model(x_train, y_train, quantiles=[0.05, 0.95], internal_nodes=[128,128], 
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shenghao/mccp_lib/mccp_lib/models/mqnn.py", line 57, in train_mqnn_model
    model = build_mqnn(quantiles, x_train, internal_nodes=internal_nodes, montecarlo=montecarlo)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shenghao/mccp_lib/mccp_lib/models/mqnn.py", line 35, in build_mqnn
    inputs = keras.layers.Input(shape=input_dim)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shenghao/anaconda3/envs/mccp/lib/python3.12/site-packages/keras/src/layers/core/input_layer.py", line 209, in Input
    layer = InputLayer(
            ^^^^^^^^^^^
  File "/home/shenghao/anaconda3/envs/mccp/lib/python3.12/site-packages/keras/src/layers/core/input_layer.py", line 92, in __init__
    shape = backend.standardize_shape(shape)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shenghao/anaconda3/envs/mccp/lib/python3.12/site-packages/keras/src/backend/common/variables.py", line 562, in standardize_shape
    raise ValueError(f"Cannot convert '{shape}' to a shape.")
ValueError: Cannot convert '13' to a shape.
```

## Requirements

```bash
pip install setuptools
pip install wheel
pip install tensorflow==2.13.1
pip install scikit-learn
pip install torch=2.7.0
pip install torchvision
pip install pandas
pip install tqdm
pip install scikit-learn
pip install pillow
pip install matplotlib
```


## Benchmarking

Classification

- [ ] CIFAR10
- [ ] CIFAR100
- [ ] MNIST
- [ ] FASHIONMNIST

Regression

- [ ] Abalone
- [ ] BostonHousing
- [ ] Concrete


## Tips

```shell
# To control the devices
CUDA_VISIBLE_DEVICES=1 
```

##  Pipeline and optimizations

The whold work combines:

- MC Dropout: to get predictive distributions (uncertainty)
- Conformal Prediction (CP): to guarantee calibrated prediction sets (classification) or intervals (regression)

Two methods:
- Classification: use RAPS (Regularized Adaptive Prediction Sets)
- Regression: use CQR (Conformalized Quantile Regression)

## Mini benchmarking for each component

- [ ] `dynamic_mc_predict`
  - Return: `montecarlo_predictions` -> Tensor of shape (N, num_classes)
- [ ] `raps_calibration`
  - Return: `q_hat` -> float as threshold
- [ ] `raps_cp` (classification), `cqr_cp` (regression)
  - Return: `pred_set`, `label_set` and `error`

### Optimization

- All tensors maintained by `torch` to ensure CUDA support
- Running dropout-enabled forward passes in batches (to utilize GPU parallelism)
- Vectorization
- Incorporate the optional randomization after calibration (in the prediction step) rather than during calibration.
- Broadcasting to avoid looping over classes in `raps_calibration`