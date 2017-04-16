# Robust Kalman

![Alt Text](/images/x0_spike_outliers.png?raw=true)

Python implementation of a robust Kalman estimator using so called M-robust estimation with support for adaptive noise variance estimation. Robust estimation is used to give better estimates when the data is polluted by outliers (see figure above).

Implementation is based on the method presented in the paper [Robust Estimation with Unknown Noise Statistics](http://ieeexplore.ieee.org/abstract/document/769393/). Main difference is that iterative Nelder-Mead algorithm is used for nonlinear minimization problems instead of approximate linear method proposed by original authors (one may try out other methods if interested by editing [the code](robust_kalman/robust_kalman.py#L127)). Adaptive variance estimation is implemented only for measurement noise.

## Usage
Robust Kalman may be easily integrated in the user's code using few intuitive API calls as shown in the sample below.

```python
# Import RobustKalman provided by the package in this repo
from robust_kalman import RobustKalman

# Create the estimator by passing model parameters
kalman = RobustKalman(F, B, H, x0, P0, Q0, R0, use_robust_estimation=True)

# ...

# Do updates on every time step
kalman.time_update()
kalman.measurement_update(measurements)

# ...

# Get the estimations
print('Current state estimates', kalman.current_estimate)

```

Simple and fully functional example is available in the [examples/example_simple.py](examples/example_simple.py#L2?raw=true). This example contains model definition, update loop and result plotting.

The example in the [examples/example_advanced.py](examples/example_advanced.py#L2?raw=true) was intended for the authors coursework in the Stochastic System Theory at Masters program in Signal processing at University of Belgrade, School of Electrical Engineering. That script was used to generate results presented in the image above.

Author: Miloš Stojanović Stojke
