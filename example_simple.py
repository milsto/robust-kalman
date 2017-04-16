"""
Simple but fully functional example for usage of the RobustKalman implementation.

The system model is defined, system evaluation and estimation loop is implemented and results are plotted.

Author: Milos Stojanovic (github: milsto)
"""
import numpy as np
import matplotlib.pyplot as plt

from robust_kalman import RobustKalman
from utils import HuberScore, VariablesHistory, WindowStatisticsEstimator

# Define a linear state space model
dt = 0.01
end_time = 1.0
F = np.array([[1, dt], [0, 1]], np.float32)
G = np.array([[0.5 * dt**2, dt]], np.float32).T
H = np.array([[1, 0]], np.float32)
x0 = np.array([[0.01, 0.01]], np.float32).T
P0 = np.ones((2, 2), np.float32) * 0.001
sigma_process = 10.0
sigma_measure = 0.1
x0_kalman = np.array([[0, 0]], np.float32).T

Q0 = np.matmul(G, G.T) * sigma_process**2
R0 = np.eye(1, dtype=np.float32) * sigma_measure**2

# Create instance of the robust Kalman filter filter
kalman_linear = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=False)
kalman_robust = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=True)

# Initialize
x = x0
z = np.matmul(H, x0)
t_axis = np.arange(0, end_time, dt)

# Use this utility to track variables over time for plotting
history = VariablesHistory()

for t in t_axis:
    history.update('x', x)
    history.update('z', z)
    history.update('x_kalman', kalman_linear.current_estimate)
    history.update('x_kalman_robust', kalman_robust.current_estimate)

    q = np.random.normal(0.0, sigma_process, size=(1, 1))

    rare_event = 1 if np.random.uniform(0, 1.0) > 0.9 else 0
    r = np.random.normal(0.0, sigma_measure, size=(1, 1)) + np.random.choice([-1.0, 1.0]) * np.random.uniform(1.0, 1.5) * rare_event

    x = np.matmul(F, x) + np.matmul(G, q)
    z = np.matmul(H, x) + r

    kalman_linear.time_update()
    kalman_linear.measurement_update(z)
    kalman_robust.time_update()
    kalman_robust.measurement_update(z)

plt.plot(t_axis, [x[0, 0] for x in history['x']], 'g', label='$x_0\ (true\ state)$')
plt.plot(t_axis, [z[0, 0] for z in history['z']], 'b', linewidth=0.5, label='$z_0\ (measurement)$')
plt.plot(t_axis, [k[0, 0] for k in history['x_kalman']], 'm', label='$\hat{x}^{Kalman}_0$')
plt.plot(t_axis, [k[0, 0] for k in history['x_kalman_robust']], 'r', label='$\hat{x}^{robust\ Kalman}_0$')
plt.show()