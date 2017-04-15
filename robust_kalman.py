import numpy as np
from scipy.optimize import minimize



class HuberScore:
    def __init__(self, delta=1.5):
        self._delta = delta

    def evaluate(self, z):
        if abs(z) >= self._delta:
            return self._delta * abs(z) - pow(self._delta, 2) / 2.0
        else:
            return pow(z, 2) / 2.0

    def derivative(self, z):
        raise NotImplemented


class RobustKalman():
    def __init__(self, F, B, H, x0, P0, Q0, R0, use_robust_estimation=False):
        """

        Args:
            F: State transition matrix
            B: Input transition matrix
            H: Observation matrix
            x0: Initial state vector
            P0: Initial state covariance matrix
            Q0: (Initial) state noise covariance
            R0: (Initial) observation noise covariance
        """
        self.F = F
        self.B = B
        self.H = H
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0

        self.use_robust_estimation = use_robust_estimation

        self.robust_score = HuberScore(delta=1.5)

    def time_update(self, inputs=None):
        if inputs is None:
            self.x = np.matmul(self.F, self.x)
        else:
            self.x = np.matmul(self.F, self.x) + np.matmul(self.B, inputs)

        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

    def measurement_update(self, measurements):
        # Residual or inovation
        self.inovation = measurements - np.matmul(self.H, self.x)

        # Inovation covariance matrix
        Pinov = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R

        # Kalman gain K = Pxy * Pinov^-1
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(Pinov))

        if self.use_robust_estimation:
            # Represent Kalman filter as linear regression problem
            # Y = X * x_est + zeta

            # If we use simple square function as robust_score this update is
            # equivalent to standard recursive linear Kalman (it is equivalent to least square minimization)
            # But his approach is a bit slower so the standard implementation in the other case.

            # Create block matrix representing covariance of error in linear regression representation of Kalman
            epsilon_covariance = np.bmat([[self.P, np.zeros((self.P.shape[0], self.R.shape[1]))],
                                          [np.zeros((self.R.shape[0], self.P.shape[1])), self.R]])

            S = np.linalg.cholesky(epsilon_covariance)
            Sinv = np.linalg.inv(S)

            # self.x <=> F(k-1) * x_est(k-1|k-1)
            Y = np.matmul(Sinv, np.vstack((self.x, measurements)))

            #       |I|
            # S^-1 *|H|
            X = np.matmul(Sinv, np.vstack((np.eye(self.x.shape[0]), self.H)))

            exact = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

            res = minimize(lambda xx: self.m_estimate_criterion(xx, Y, X), self.x, method='nelder-mead')

            self.x = res.x[np.newaxis].T
        else:
            # Linear state update
            self.x = self.x + np.matmul(K, self.inovation)

        # State prediction covariance update
        # This covariacne update is used for robust estimator also, and in thath
        # case it is an aproximation.
        self.P = self.P - np.matmul(np.matmul(K, self.H), self.P)

    def m_estimate_criterion(self, x, Y, X):
        crit = 0.0
        for i in range(Y.shape[0]):
            crit += self.robust_score.evaluate(Y[i, :] - np.matmul(X[i, :], x))
            #crit += (Y[i, :] - np.matmul(X[i, :], x))**2

        return crit
