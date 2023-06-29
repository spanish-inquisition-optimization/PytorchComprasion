import torch

from core.gradient_descent import *
from numpy.polynomial.polynomial import Polynomial
import numpy as np

from core.utils import ApproxDump


def polynom_approx(x, y, deg, optimizer):
    x0 = torch.randn(deg, requires_grad=True).detach().numpy()

    def f(a: np.ndarray):
        return np.sum((Polynomial(a)(x) - y) ** 2)

    def df(a: np.ndarray):
        same = Polynomial(a)(x) - y
        return np.array([2 * np.sum(same * (x ** j)) for j in range(a.shape[0])])

    points = optimizer(None, None, f, df, x0)

    return ApproxDump('handmade', x, points[-1], [f(x) for x in points])


def polynom_approx_residuals(x, y, deg, optimizer):
    x0 = torch.randn(deg, requires_grad=True).detach().numpy()

    def residual(a: np.ndarray, i: int):
        return (Polynomial(a)(x[i]) - y[i])**2
    residuals = [lambda a: residual(a, i) for i in range(x.size)]

    def residual_gradient(a, i):
        # grad(r_i) = (d(r_i)/d(a_j))_{j\in [0..n)}
        err = Polynomial(a)(x[i]) - y[i]
        return np.array([2 * err * x[i]**j for j in range(a.size)])
    residual_grads = [lambda a: residual_gradient(a, i) for i in range(x.size)]

    def f(a: np.ndarray):
        return np.sum((Polynomial(a)(x) - y) ** 4)

    points = optimizer(residuals, residual_grads, f, None, x0)

    return ApproxDump('handmade', x, points[-1], [f(x) for x in points])

