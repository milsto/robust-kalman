"""
Robust Kalman filter implementation.

Author: Milos Stojanovic (github: milsto)
"""

import numpy as np
from scipy.optimize import minimize
from .utils import HuberScore


class RobustKalman():
    """Robust Kalman filter for estimation immune to outliers.

    The implementation is based on rewriting classical linear recursive Kalman approach as linear regression problem.
    Linear regression representation is equivalent to the original problem when it is solved as least square
    minimization problem. To implement robust Kalman estimation, instead of least square criterion, some other robust
    score function is used. The robust score function is responsible to suppress outliers during error
    calculations by having less steep derivative when the error is too large (it is assumed that in that case an
    outlier is observed).

    Usage of robust estimations is controlled by use_robust_estimation flag. When it is turned off estimatior behaves
    as classical recursive Kalman. Estimations of state covariance matrix P is always done by classical Kalman aproach
    and is (good) approximation in the cases when robust score function is used. The robust estimation approach is slower
    than the standard one and to solve nonlinear minimization problem the iterative Nedler-Mead algorithm is used.

    A prototype of adaptive measurement variance estimation is also available with use_adaptive_statistics. The method
    is based on estimation the variance based on history of the noise samples. Be aware that in this case the Kalman
    procedure is not purely recursive anymore but uses memory to store previous samples.

    """
    def __init__(self, F, B, H, x0, P0, Q0, R0, use_robust_estimation=False, use_adaptive_statistics=False, robust_score=HuberScore(delta=1.5)):
        """Initialize robust Kalman. All input matrices are coppied.

        Args:
            F: State transition matrix
            B: Input transition matrix (may be None if model has no inputs)
            H: Observation matrix
            x0: Initial state vector
            P0: Initial state covariance matrix
            Q0: (Initial) state noise covariance
            R0: (Initial) observation noise covariance
            use_robust_estimation: True if robust estimation procedure should be used
            use_adaptive_statistics: True if adaptive robust estimation of noise variance should be used
            robust_score: Score function for robust estimation. (1.5)-Huber is the default.
        """
        self.F = F.copy()
        self.B = B.copy() if B is not None else None
        self.H = H.copy()
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q0.copy()
        self.R = R0.copy()

        self.use_robust_estimation = use_robust_estimation
        self.use_adaptive_statistics = use_adaptive_statistics

        # Used for adaptive noise estimation
        self.history_inovation = list()
        self.r_mean_est = 0.0
        self.r_var_est = 0.0

        self.robust_score = robust_score

    def time_update(self, inputs=None):
        """
        Time propagation of the system model.

        Args:
            inputs: Model inputs if any.

        """
        if inputs is None:
            self.x = np.matmul(self.F, self.x)
        else:
            self.x = np.matmul(self.F, self.x) + np.matmul(self.B, inputs)

        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

    def measurement_update(self, measurements):
        """
        Measurement update. Not that time update must preceded the measurement update
        for valid estimation results.

        Args:
            measurements: Observations of measured quantities.

        """
        # Residual or inovation
        self.inovation = measurements - np.matmul(self.H, self.x)

        # Inovation covariance matrix
        Pinov = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R

        # Kalman gain K = Pxy * Pinov^-1
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(Pinov))

        if self.use_robust_estimation:
            # Represent Kalman filter as linear regression problem
            # Y = X * x_est + zeta
            # This is achieved by stacking system and measurement update equations in matrix quation
            # and than manipulating linear algebra to get the linear regression form (see reference papers for details).

            # If we use simple square function as robust_score this update is
            # equivalent to standard recursive linear Kalman (it is equivalent to least square minimization)
            # But his approach is a bit slower so the standard implementation is used in the other case.

            # Create block matrix representing covariance of error in linear regression representation of Kalman
            epsilon_covariance = np.bmat([[self.P, np.zeros((self.P.shape[0], self.R.shape[1]))],
                                          [np.zeros((self.R.shape[0], self.P.shape[1])), self.R]])

            # Factorize covariance to S * S^T form with Cholesky decomposition.
            S = np.linalg.cholesky(epsilon_covariance)
            Sinv = np.linalg.inv(S)

            # self.x <=> F(k-1) * x_est(k-1|k-1)
            Y = np.matmul(Sinv, np.vstack((self.x, measurements)))

            #       |I|
            # S^-1 *|H|
            X = np.matmul(Sinv, np.vstack((np.eye(self.x.shape[0]), self.H)))

            # Exact solution to non-robust Kalman linear regression problem (for debuge)
            # exact = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

            # TODO: Expose the possibility of using other optimization methods in interface.
            res = minimize(lambda xx: self._m_estimate_criterion(xx, Y, X), self.x, method='nelder-mead')

            self.x = res.x[np.newaxis].T
        else:
            # Linear state update
            self.x = self.x + np.matmul(K, self.inovation)

        # State prediction covariance update
        # This covariance update is used for robust estimator also, and in thath
        # case it is an approximation.
        self.P = self.P - np.matmul(np.matmul(K, self.H), self.P)

        if self.use_adaptive_statistics:
            assert self.R.shape == (1, 1), 'Current implementation for robust variance estimation tested only for ' \
                                           'models with one observable.'
            self.history_inovation.append(self.inovation)
            if len(self.history_inovation) < 6:
                self.r_mean_est = 0.0
                self.r_var_est = self.R[0, 0]
            else:
                # Adaptive estimate of R
                r_arr = np.array(self.history_inovation, dtype=np.float32)
                d = np.median(np.fabs(r_arr - np.median(r_arr)) / 0.6745)

                self.r_mean_est = minimize(lambda xx: self._m_estimate_r_criterion(xx, r_arr, d), self.history_inovation[-1], method='nelder-mead').x
                self.r_var_est = d**2 - np.matmul(np.matmul(self.H, self.P), self.H.T)

            self.R[0, 0] = self.r_var_est

    @property
    def current_estimate(self):
        return self.x

    @property
    def current_estimate_covariance(self):
        return self.P

    @property
    def current_inovation(self):
        return self.inovation

    def _m_estimate_criterion(self, x, Y, X):
        """Criterion for robust state estimation"""
        crit = 0.0
        for i in range(Y.shape[0]):
            crit += self.robust_score.evaluate(Y[i, :] - np.matmul(X[i, :], x))
            #crit += (Y[i, :] - np.matmul(X[i, :], x))**2

        return crit

    def _m_estimate_r_criterion(self, x, r_est_arr, d):
        """Criterion for robust variance estimation in adaptive procedure."""
        crit = 0.0
        for i in range(len(r_est_arr)):
            crit += self.robust_score.evaluate((r_est_arr[i] - x) / d)

        return crit
