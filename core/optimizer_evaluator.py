import asyncio
from random import random, uniform
from typing import Callable, NamedTuple, Tuple, Any, Optional

import numpy as np
from numpy import newaxis

from core.gradient_descent import fibonacci_search, precision_termination_condition


class QuadraticForm(NamedTuple):
    """
    Represents a quadratic form of the form `x^T A x` where `A` is a symmetric matrix.
    Switching to arbitrary quadratic functions doesn't bring any additional interest to the problem.
    """
    matrix: np.ndarray
    k: Optional[float] = None
    n = property(lambda self: self.matrix.shape[0])

    def __call__(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return (x @ self.matrix @ x[:, newaxis])[0]

    def gradient_function(self):
        return lambda x: (2 * self.matrix @ x[:, newaxis])[:, 0]

    def hessian_function(self):
        return lambda x: 2 * self.matrix

    def get_conditional_number(self):
        return self.k


def random_orthonormal_basis(n):
    res = None
    while res is None or np.linalg.matrix_rank(res) < n:
        res = np.random.rand(n, n)
        res = np.linalg.qr(res)[0]
    return res


def canonical_basis(n):
    return np.eye(n)


def random_normalized_vector(n):
    vec = np.random.rand(n)
    return vec / np.linalg.norm(vec)


def generate_positive_definite_quadratic_form(dimensions, condition_number, eigenbasis_producer=canonical_basis):
    """
    Generates a random positive definite quadratic form of given dimensions and condition number.
    The condition number is defined as the ratio of the greatest eigenvalue to the smallest one.

    The smallest eigenvalue is always 1.
    The greatest eigenvalue is always `condition_number`.
    The eigenvalues in between are random.

    Firstly the representation of the matrix in the eigenbasis
    (which must be orthonormal to basis make basis and form transformations equivalent)
    is generated and then transformed to the canonical basis.
    """

    assert condition_number >= 1
    assert dimensions >= 2
    smallest_eigenvalue = 1.
    greatest_eigenvalue = smallest_eigenvalue * condition_number

    # The order of coordinates obviously doesn't matter
    eigenvalues = [smallest_eigenvalue] + [uniform(smallest_eigenvalue, greatest_eigenvalue) for _ in
                                           range(dimensions - 2)] + [greatest_eigenvalue]
    matrix_in_basis = np.diag(eigenvalues)

    B = eigenbasis_producer(dimensions)

    # Now we transform the matrix TO the canonical basis
    # Transposition matrix from canonical basis to B is just matrix `B`
    T_C_B = B
    T_B_C = B.T  # Just like `np.linalg.inv(T_C_B)` but faster
    return QuadraticForm(T_C_B @ matrix_in_basis @ T_B_C, condition_number)


def average(computation: Callable[[], float], n_times) -> float:
    return sum(computation() for _ in range(n_times)) / n_times


def iteration_count_computer(form_generator: Callable[[], QuadraticForm], optimizer: Callable, line_searcher) -> \
        Callable[[], float]:
    def computation():
        f = form_generator()
        return len(optimizer(f, f.gradient_function(), random_normalized_vector(f.n), line_searcher,
                             precision_termination_condition))

    return computation


def average_iterations_until_convergence(form_generator: Callable[[], QuadraticForm], optimizer: Callable, n_times,
                                         line_searcher=fibonacci_search(30)) -> float:
    return average(iteration_count_computer(form_generator, optimizer, line_searcher), n_times)


def logspace_range(start, stop, n, **kwargs):
    return np.logspace(np.log10(start), np.log10(stop), n, base=10, **kwargs)

