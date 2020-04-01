from scipy.optimize import minimize
import numpy as np
from typing import Callable, Union, List


class TaylorApproximator:
    def __init__(self,
                 x: Callable, p: int = 2,
                 t0: Union[np.ndarray, List] = np.array([0, 0]),
                 t_m: Union[np.ndarray, List] = np.array([1, 2])):
        t0 = np.array(t0)
        t_m = np.array(t_m)
        self.p = p
        self.y = x(t_m)
        self.t_m = t_m
        self.t0 = t0

        self.poly = np.array(
            [(self.t_m - self.t0)**j for j in range(p+1)])

    def err(self, alpha: Union[np.ndarray, List]):
        return np.sum((self.y - alpha @ self.poly)**2)

    def __call__(self, init_guess: Union[np.ndarray, List]):
        # minimize on call
        self.result = minimize(self.err, init_guess, method='CG')
        # print(self.result)
        return self.result.x


class RHSApproximator:
    def __init__(self,
                 alpha_hat: Union[np.ndarray, List],
                 p: int = 2,
                 t0: Union[np.ndarray, List] = np.array([0, 0]),
                 t_m: Union[np.ndarray, List] = np.array([1, 2])):
        t0 = np.array(t0).astype(np.float32)
        t_m = np.array(t_m).astype(np.float32)
        self.t_m = t_m
        self.t0 = t0
        self.p = p
        self.alpha_hat = alpha_hat

    def x_hat(self, t):
        return self.alpha_hat @ np.array(
            [(t - self.t0)**j for j in range(self.p+1)])

    def x_hat_prime(self, t):
        return self.alpha_hat @ np.array(
            [j*(t - self.t0)**(max(j-1, 0)) for j in range(self.p+1)])

    def rhs(self, t, beta):
        return self.x_hat(t) + beta[0]

    def err(self, beta: Union[np.ndarray, List]):
        return np.sum((self.x_hat_prime(self.t_m) - self.rhs(self.t_m, beta))**2)

    def __call__(self, init_guess: Union[np.ndarray, List]):
        # minimize on call
        self.result = minimize(self.err, init_guess, method='CG')
        # print(self.result)
        return self.result.x


def coef(t_1, t_0):
    """Find error coefficient (relative error) for derivative method."""
    numer = (np.exp(t_1) - (1+t_1-t_0)*np.exp(t_0))
    denom = ((np.exp(t_1) - np.exp(t_0))*(t_1-t_0))
    return numer/denom


if __name__ == "__main__":
    def x(t):
        return 4*np.exp(t) - 3

    def rhs(t, beta):
        return x(t) + beta[0]

    p = 3

    app = TaylorApproximator(
        x=x, p=p, t0=[0, 0], t_m=[1, 2])

    def x_prime(t):
        return 4*np.exp(t)

    alpha_hat = app([0.001 for _ in range(p+1)])
    print(alpha_hat)

    app_rhs = RHSApproximator(alpha_hat=alpha_hat, p=p, t0=app.t0,
                              t_m=app.t_m)

    beta_hat = app_rhs([0])
    print(beta_hat[0])
