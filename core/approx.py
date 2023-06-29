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

    points = optimizer(f, df, x0)

    return ApproxDump('handmade', x, points[-1], [f(x) for x in points])
