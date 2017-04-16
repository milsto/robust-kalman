"""
Script to evaluate the robust and adaptive Kalman estimator.

Primarily used for to make conclusions and plots for the paper written
for Stochastic System Theory (MSc) course at University of Belgrade, School of Electrical Engineering.

Author: Milos Stojanovic (github: milsto)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.insert(0, '..')

from robust_kalman import RobustKalman
from robust_kalman.utils import HuberScore, VariablesHistory, WindowStatisticsEstimator

np.random.seed(256)

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'font.family': 'Times'
}
matplotlib.rcParams.update(params)

# Plot robust score function
# t = np.linspace(-50, 50, 1000)
# h = HuberScore()
# hv = np.vectorize(h.evaluate)
# plt.plot(t, hv(t))
# plt.show()

# Example 1
# dt = 0.01
# end_time = 1.0
# F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]], np.float32)
# G = np.array([[0, 0, 1]], np.float32).T
# H = np.array([[1, 0, 0]], np.float32)
# x0 = np.array([[0.1, 0.1, 0.1]], np.float32).T
# P0 = np.eye(3, dtype=np.float32) * 0.01
# sigma_process = 10.0
# sigma_measure = 1.0
# x0_kalman = np.array([[0, 0, 0]], np.float32).T


# Example 2
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

IS_SPIKE_EXPERIMENT = True
PLOT_ADAPTIVE_CEE = True


Q0 = np.matmul(G, G.T) * sigma_process**2
R0 = np.eye(1, dtype=np.float32) * sigma_measure**2

kalman_linear = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=False, use_adaptive_statistics=False)
kalman_robust = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=True, use_adaptive_statistics=False)
kalman_robust_stat = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=True, use_adaptive_statistics=True)

wstat_q = WindowStatisticsEstimator(win_size=25)
wstat_r = WindowStatisticsEstimator(win_size=25)


x = x0
z = np.matmul(H, x0)
cee_x = 0.0
cee_xres = 0.0
cee_xres_stat = 0.0
step = 2
t_axis = np.arange(0, end_time, dt)

history = VariablesHistory()

for t in t_axis:
    history.update('x', x)
    history.update('z', z)
    history.update('x_kalman', kalman_linear.current_estimate)
    history.update('x_kalman_robust', kalman_robust.current_estimate)
    history.update('x_kalman_robust_stat', kalman_robust_stat.current_estimate)
    cee_x += (np.linalg.norm(kalman_linear.current_estimate - x) / (np.linalg.norm(x) + 0.0001)) / step
    cee_xres += (np.linalg.norm(kalman_robust.current_estimate - x) / (np.linalg.norm(x) + 0.0001)) / step
    cee_xres_stat += (np.linalg.norm(kalman_robust_stat.current_estimate - x) / (np.linalg.norm(x) + 0.0001)) / step
    history.update('cee_x_history', cee_x)
    history.update('cee_xres_history', cee_xres)
    history.update('cee_xres_stat_history', cee_xres_stat)

    history.update('r_mean_est', kalman_robust_stat.r_mean_est)
    history.update('r_var_est', kalman_robust_stat.r_var_est)

    q = np.random.normal(0.0, sigma_process, size=(1, 1))
    if not IS_SPIKE_EXPERIMENT:
        r = 0.85 * np.random.normal(0.0, sigma_measure, size=(1, 1)) + 0.15 * np.random.normal(0.0, 5.0, size=(1, 1))
    else:
        rare_event = 1 if np.random.uniform(0, 1.0) > 0.9 else 0
        r = np.random.normal(0.0, sigma_measure, size=(1, 1)) + np.random.choice([-1.0, 1.0]) * np.random.uniform(1.0, 1.5) * rare_event #+ 0.15 * np.random.normal(0.0, 5.0, size=(1, 1))

    wstat_q.update(q)
    wstat_r.update(r)
    history.update('wstat_r_mean', wstat_r.mean())
    history.update('wstat_r_var', wstat_r.variance())

    x = np.matmul(F, x) + np.matmul(G, q)
    z = np.matmul(H, x) + r

    kalman_linear.time_update()
    kalman_linear.measurement_update(z)
    kalman_robust.time_update()
    kalman_robust.measurement_update(z)
    kalman_robust_stat.time_update()
    kalman_robust_stat.measurement_update(z)

    history.update('inov', kalman_robust.current_inovation)

    step += 1

plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, [x[0, 0] for x in history['x']], 'g', label='$x_0\ (true\ state)$')
plt.plot(t_axis, [z[0, 0] for z in history['z']], 'b', linewidth=0.5, label='$z_0\ (measurement)$')
plt.plot(t_axis, [k[0, 0] for k in history['x_kalman']], 'm', label='$\hat{x}^{Kalman}_0$')
plt.plot(t_axis, [k[0, 0] for k in history['x_kalman_robust']], 'r', label=r'$\hat{x}^\mathbf{robust\ Kalman}_0$')
plt.xlabel(r'$t [\mathrm{s}]$')
plt.ylabel(r'$x_0 [\mathrm{m}]$')
plt.axis('tight')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.legend()
if IS_SPIKE_EXPERIMENT:
    plt.savefig('x0_spike_outliers.pdf')
else:
    plt.savefig('x0_normal_outliers.pdf')


plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, [x[1, 0] for x in history['x']], 'g', label='$x_1$')
plt.plot(t_axis, [k[1, 0] for k in history['x_kalman']], 'm', label='$\hat{x}^{Kalman}_1$')
plt.plot(t_axis, [k[1, 0] for k in history['x_kalman_robust']], 'r', label='$\hat{x}^{robust\ Kalman}_1$')
plt.xlabel(r'$t [\mathrm{s}]$')
plt.ylabel(r'$x_1 [\mathrm{m/s}]$')
plt.axis('tight')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.legend()
if IS_SPIKE_EXPERIMENT:
    plt.savefig('x1_spike_outliers.pdf')
else:
    plt.savefig('x1_normal_outliers.pdf')


plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, history['cee_x_history'] / np.arange(1, len(history['cee_x_history']) + 1, 1), 'm', label='$\mathrm{CEE}_{Kalman}$')
plt.plot(t_axis, history['cee_xres_history'] / np.arange(1, len(history['cee_xres_history']) + 1, 1), 'r', label='$\mathrm{CEE}_{robust\ Kalman}$')
if PLOT_ADAPTIVE_CEE:
    plt.plot(t_axis, history['cee_xres_stat_history'] / np.arange(1, len(history['cee_xres_stat_history']) + 1, 1), 'b', label='$\mathrm{CEE}_{robust\ adaptive\ Kalman}$')
plt.xlabel(r'$t [\mathrm{s}]$')
plt.ylabel(r'$\mathrm{CEE}$')
plt.axis('tight')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.legend()
if IS_SPIKE_EXPERIMENT:
    plt.savefig('cee_spike_outliers.pdf')
else:
    plt.savefig('cee_normal_outliers.pdf')


plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, [k[0, 0] for k in history['inov']], 'k')
plt.title('inovation')

plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, history['wstat_r_mean'], 'k', label=r'$\mathrm{E}\left\{r_{windowed}\right\}$')
plt.plot(t_axis, history['r_mean_est'], 'b', label=r'$\mathrm{E}\left\{\hat{r}_{est}\right\}$')
plt.xlabel(r'$t [\mathrm{s}]$')
plt.ylabel(r'$\mathrm{E}\left\{r\right\} [\mathrm{m}]$')
plt.axis('tight')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.legend()
if IS_SPIKE_EXPERIMENT:
    plt.savefig('r_mean_spike_outliers.pdf')
else:
    plt.savefig('r_mean_normal_outliers.pdf')

plt.figure(figsize=[15/2.54, 10/2.54])
plt.plot(t_axis, history['wstat_r_var'], 'k', label=r'$\mathrm{Var}\left\{r_{windowed}\right\}$')
plt.plot(t_axis, history['r_var_est'], 'b', label=r'$\mathrm{Var}\left\{\hat{r}_{est}\right\}$')
plt.xlabel(r'$t [\mathrm{s}]$')
plt.ylabel(r'$\mathrm{Var}\left\{r\right\} [\mathrm{m^2}]$')
plt.axis('tight')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.legend()
if IS_SPIKE_EXPERIMENT:
    plt.savefig('r_variance_spike_outliers.pdf')
else:
    plt.savefig('r_variance_normal_outliers.pdf')

plt.show()
