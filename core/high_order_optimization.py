from typing import Callable, List
from abc import ABC, abstractmethod

import numpy as np
import scipy.constants
from math import exp, floor, sqrt

from numpy import newaxis
from numpy.linalg import LinAlgError

from collections import deque

from core.gradient_descent import gradient_descent, steepest_descent
from core.utils import *


class NewtonDirectionApproximator(ABC):
    def __init__(self):
        self.inv_hessian = None

    def absorb_initial_approximation(self, initial_inv_hessian):  # Is given an inverse hessian
        self.inv_hessian = initial_inv_hessian

    @abstractmethod
    def approximate_inverse_hessian(self, new_point, new_gradient):  # Supposed to return a positive definite matrix
        pass

    def compute_direction(self, point, gradient) -> np.ndarray:
        h = self.approximate_inverse_hessian(point, gradient)
        return -h @ gradient if h is not None else None

    def absorb_step_size(self, step_size): # Called at the end of iteration: when the step along direction is chosen
        pass


class BFGSNewtonDirectionApproximator(NewtonDirectionApproximator):
    def __init__(self):
        super().__init__()
        self.old_gradient = None
        self.old_point = None
        self.iteration = 0

    def approximate_inverse_hessian(self, point, gradient):
        # assert self.inv_hessian is not None, "Should provide initial approximation first"

        if self.old_point is not None:
            # Update hessian approximation
            s = point - self.old_point
            y = gradient - self.old_gradient
            rho = 1 / (s @ y)
            if self.iteration <= 1:  # Second iteration
                gamma = s @ y / y @ y  # H^0 = gamma * I
                self.inv_hessian = np.eye(point.size) * gamma


            # Expand equation and put parentheses properly
            # to avoid matrix-matrix operations which are O(n^3)
            def pre_multiply(m):
                return s[:, newaxis] @ (y @ m)[newaxis]

            def post_multiply(m):
                return (m @ y[:, newaxis]) @ s[newaxis]

            old_ih = np.copy(self.inv_hessian)
            self.inv_hessian += \
                - rho * post_multiply(self.inv_hessian) \
                - rho * pre_multiply(self.inv_hessian) \
                + rho ** 2 * post_multiply(pre_multiply(self.inv_hessian)) \
                + rho * s[:, newaxis] @ s[newaxis]
            # delta = old_ih - self.inv_hessian
            # print(np.linalg.matrix_rank(delta))  # Is always 2 as expected
            # print(np.linalg.norm(delta) / np.linalg.norm(old_ih))

        # print(self.inv_hessian)

        self.old_gradient = gradient
        self.old_point = point
        self.iteration += 1
        return self.inv_hessian


class LBFGSNewtonDirectionApproximator(NewtonDirectionApproximator):
    def __init__(self, m: int):
        """
            m is the number of last iterations used to approximate hessian
        """
        super().__init__()
        self.secant_storage = deque()
        self.old_gradient = self.old_point = None
        self.m = m

    def approximate_inverse_hessian(self, point, gradient):
        assert False, "It's too computationally expensive!"

    def compute_direction(self, point, gradient):
        if self.old_point is not None:  # At least second iteration
            s = point - self.old_point
            y = gradient - self.old_gradient
            gamma = s @ y / y @ y  # H^0_k = gamma * I
            self.secant_storage.append((s, y))

            q = np.copy(gradient)
            alphas = []
            for (si, yi) in reversed(self.secant_storage):
                rho = 1 / (si @ yi)
                alpha = rho * si @ q
                alphas.append(alpha)
                q -= alpha * yi

            r = gamma * np.copy(q)  # H^0_k * q

            for ((si, yi), alpha) in zip(self.secant_storage, reversed(alphas)):
                rho = 1 / (si @ yi)
                beta = rho * yi @ r
                r += si * (alpha - beta)

            res = -r
        else:
            res = None

        self.old_gradient = gradient
        self.old_point = point

        if len(self.secant_storage) > self.m:
            self.secant_storage.popleft()

        return res


class GivenNewtonDirectionApproximator(NewtonDirectionApproximator):
    def __init__(self, computer):
        super().__init__()
        self.inv_hessian_computer = computer

    def approximate_inverse_hessian(self, point, gradient):
        return self.inv_hessian_computer(point)

    @classmethod
    def numerically_computing(cls, f, g):
        return cls(lambda x: np.linalg.inv(symmetrically_compute_hessian_by_gradient(f, g, NUMERIC_GRADIENT_COMPUTING_PRECISION, x)))


def numeric_inverse_jacobian_approximator(f, g, x0):
    return np.linalg.inv(symmetrically_compute_hessian_by_gradient(f, g, NUMERIC_GRADIENT_COMPUTING_PRECISION, x0))


def none_approximation(_f, _g, _x0):
    return None


def eye_initial_approximation(_f, _g, x0):
    return np.eye(x0.size)


def known_initial_approximator(provider):
    return lambda _f, _g, x0: provider(x0)


def newton_optimize(
        target_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        direction_approximator: NewtonDirectionApproximator,
        x0: np.ndarray,
        linear_search,
        terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool],
        initial_approximator=none_approximation
):
    def direction_function(x: np.ndarray, *args, **kwargs):
        g = gradient_function(x)
        direction = direction_approximator.compute_direction(x, g)
        return direction if direction is not None else -g

    direction_approximator.absorb_initial_approximation(
        initial_approximator(target_function, gradient_function, x0)
    )
    # print("[newton_optimize] Computed initial approximation")

    return gradient_descent(
        target_function,
        gradient_function,
        direction_function,
        x0,
        linear_search,
        terminate_condition
    )


def gauss_newton(residuals: List[Callable[[np.ndarray], float]],
                 gradients: List[Callable[[np.ndarray], np.ndarray]],
                 x0: np.ndarray,
                 termination_condition: Callable[[Callable, List[np.ndarray]], bool]):
    f = lambda x: sum((r(x) ** 2 for r in residuals))

    points = [x0]
    current = x0
    while not termination_condition(f, points):
        jac = np.array([grad(current) for grad in gradients])
        try:
            current = current - np.linalg.inv(np.transpose(jac) @ jac) @ np.transpose(jac) @ np.array(
                [r(current) for r in residuals])
        except LinAlgError:
            # jacobian is zero, so there is a actual local minimum
            return points

        points.append(current)

    return points


def gauss_newton_with_approx_grad(residuals: List[Callable[[np.ndarray], float]],
                                  x0: np.ndarray,
                                  termination_condition: Callable[[Callable, List[np.ndarray]], bool]):
    n = x0.shape[0]

    def gradient(n, f: Callable[[np.ndarray], float]):
        eps = 0.001

        def standard_vector(n, i):
            vec = np.zeros(n)
            vec[i] = 1
            return vec

        return lambda x: np.array([(f(x + eps * standard_vector(n, i)) - f(x)) / eps for i in range(n)])

    gradients = [gradient(n, r) for r in residuals]
    return gauss_newton(residuals, gradients, x0, termination_condition)



# Takes predicate f with conditions: f(lbound) == False, f(rbound) = True
def binary_search(f: Callable[[float], bool], lbound: float, rbound: float):
    while rbound - lbound > 1e-8:
        m = (lbound + rbound) / 2
        if f(m):
            lbound = m
        else:
            rbound = m

    return rbound


def dogleg(
        residuals: List[Callable[[np.ndarray], float]],
        gradients: List[Callable[[np.ndarray], np.ndarray]],
        max_trusted,
        linear_search,
        x0: np.ndarray,
        termination_condition: Callable[[Callable, List[np.ndarray]], bool]
):
    f = lambda x: 0.5 * sum((r(x) ** 2 for r in residuals))
    df = lambda x: sum((r(x) * g(x) for r, g in zip(residuals, gradients)))
    points = [x0]
    trusted = 1
    while not termination_condition(f, points):
        # TODO: probably can rid of specialization of gauss-newton direction
        current = points[-1]
        jac = np.array([grad(current) for grad in gradients])
        b = np.transpose(jac) @ jac

        try:
            gn = -np.linalg.inv(np.transpose(jac) @ jac) @ np.transpose(jac) @ np.array([r(current) for r in residuals])
        except LinAlgError:
            # go away from bad point
            gn = np.random.normal(size=x0.shape[0])
            gn /= np.linalg.norm(gn)
            gn *= trusted / 2
        p = np.zeros(x0.shape)

        if np.linalg.norm(gn - current) < trusted:
            p = gn
        else:
            sdp = steepest_descent(
                f, df, current,
                linear_search,
                lambda _, ps: len(ps) > 1
            )

            sd = sdp[-1] - current

            if np.linalg.norm(sd) > trusted:
                p = (trusted / np.linalg.norm(sd)) * sd
                if f(current + p) > f(sdp[-1]):
                    p = sd
            else:
                t = binary_search(lambda t: np.linalg.norm((gn - sd) * t + sd - current) > trusted, 0, 1)
                p = (gn - sd) * t + sd

        # update trusted region
        b = np.transpose(jac) @ jac
        m = lambda p: f(current) + np.dot(p, df(current)) + (0.5 * p) @ b @ np.transpose(p)
        rho = (f(current) - f(current + p)) / (m(np.zeros(x0.shape)) - m(p))

        if rho < 0.25:
            trusted *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) - trusted < 1e-7:
            trusted = min(2 * trusted, max_trusted)

        points.append(current + p)

    return points
