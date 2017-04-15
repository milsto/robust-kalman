import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from robust_kalman import RobustKalman, HuberScore

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


class VariablesHistory:
    def __init__(self):
        self._variables_history = dict()

    def __getitem__(self, item):
        return self._variables_history[item]

    def update(self, variable_name, value):
        if variable_name not in self._variables_history:
            self._variables_history[variable_name] = list()

        self._variables_history[variable_name].append(value)

# Example 1
# dt = 4.0
# end_time = 400.0
# F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]], np.float32)
# G = np.array([[0, 0, 1]], np.float32).T
# H = np.array([[1, 0, 0]], np.float32)
# x0 = np.array([[0, 0, 0]], np.float32).T
# P0 = np.eye(3, dtype=np.float32) * 0.01
# sigma_process = 1.0
# sigma_measure = 1.0
#x0_kalman = np.array([[0, 0]], np.float32).T


# Example 2
dt = 0.01
end_time = 1.0
F = np.array([[1, dt], [0, 1]], np.float32)
G = np.array([[0.5 * dt**2, dt]], np.float32).T
H = np.array([[1, 0]], np.float32)
x0 = np.array([[0.1, 0.1]], np.float32).T
P0 = np.ones((2, 2), np.float32) * 0.001
sigma_process = 8.0
sigma_measure = 0.1
x0_kalman = np.array([[0, 0]], np.float32).T


Q0 = np.matmul(G, G.T) * sigma_process**2
R0 = np.eye(1, dtype=np.float32) * sigma_measure**2

kalman_linear = RobustKalman(F, [0], H, x0_kalman, P0, Q0, R0, use_robust_estimation=False)
kalman_robust = RobustKalman(F, [0], H, x0_kalman, P0, Q0, R0, use_robust_estimation=True)


x = x0
z = np.matmul(H, x0)
cee_x = 0.0
cee_xres = 0.0
step = 2
t_axis = np.arange(0, end_time, dt)

history = VariablesHistory()

for t in t_axis:
    history.update('x', x)
    history.update('z', z)
    history.update('x_kalman', kalman_linear.x)
    history.update('x_kalman_robust', kalman_robust.x)
    cee_x += (np.linalg.norm(kalman_linear.x - x) / (np.linalg.norm(x) + 0.0001)) / step
    cee_xres += (np.linalg.norm(kalman_robust.x - x) / (np.linalg.norm(x) + 0.0001)) / step
    history.update('cee_x_history', cee_x)
    history.update('cee_xres_history', cee_xres)

    q = np.random.normal(0.0, sigma_process, size=(1, 1))
    #r = 0.85 * np.random.normal(0.0, sigma_measure, size=(1, 1)) + 0.15 * np.random.normal(0.0, 5.0, size=(1, 1))
    rare_event = 1 if np.random.uniform(0, 1.0) > 0.9 else 0
    r = np.random.normal(0.0, sigma_measure, size=(1, 1)) + np.random.choice([-1.0, 1.0]) * np.random.uniform(1.0, 2.0) * rare_event #+ 0.15 * np.random.normal(0.0, 5.0, size=(1, 1))

    x = np.matmul(F, x) + np.matmul(G, q)
    z = np.matmul(H, x) + r

    kalman_linear.time_update()
    kalman_linear.measurement_update(z)
    kalman_robust.time_update()
    kalman_robust.measurement_update(z)

    history.update('inov', kalman_robust.inovation)

    step += 1

plt.figure()
plt.plot(t_axis, [x[0, 0] for x in history['x']], 'g', label='x0')

plt.hold(True)

plt.plot(t_axis, [z[0, 0] for z in history['z']], 'b', label='z0')

plt.plot(t_axis, [k[0, 0] for k in history['x_kalman']], 'r', label='x est')

plt.plot(t_axis, [k[0, 0] for k in history['x_kalman_robust']], 'k', label='x est robust')

plt.legend()

plt.figure()
plt.plot(t_axis, history['cee_x_history'] / np.arange(1, len(history['cee_x_history']) + 1, 1), 'r')
plt.hold(True)
plt.plot(t_axis, history['cee_xres_history'] / np.arange(1, len(history['cee_xres_history']) + 1), 'k')

plt.figure()
plt.plot(t_axis, [k[0, 0] for k in history['inov']], 'k')
plt.title('inovation')

plt.show()