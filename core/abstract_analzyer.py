from collections import namedtuple
from typing import List, Callable, Optional, NamedTuple, Tuple

import numpy as np
import scipy.optimize
from numpy.polynomial import Polynomial
from tabulate import tabulate

from core.gradient_descent import wolfe_conditions_search, precision_termination_condition
from core.high_order_optimization import newton_optimize, NewtonDirectionApproximator, none_approximation, dogleg, \
    gauss_newton
from core.optimizer_evaluator import QuadraticForm, random_normalized_vector, generate_positive_definite_quadratic_form, \
    random_orthonormal_basis
from core.utils import time_limit, TimeoutException, mesuare_time

from scipy.optimize import rosen, Bounds, OptimizeResult
from scipy.optimize import rosen_der

from scipy.optimize import least_squares


class Problem(NamedTuple):
    name: str
    function: Callable
    x0: np.array
    gradient: Callable
    hessian: Optional[Callable]

    @classmethod
    def from_quadratic_form(cls, q: QuadraticForm, name=None):
        return Problem(
            name if name is not None else f"Quadratic[n={q.n}, k={q.get_conditional_number()}]",
            q,
            random_normalized_vector(q.n),
            q.gradient_function(),
            q.hessian_function()
        )

    @classmethod
    def from_rosen(cls, n: int):
        return Problem(
            f"Rosen[n={n}]",
            rosen,
            np.zeros(n),
            rosen_der,
            None
        )


class LeastSquaresProblem(NamedTuple):
    name: str
    residuals: List[Callable]
    residual_gradients: List[Callable]
    p0: np.array
    bounds: Optional[Tuple] = (-np.inf, np.inf)

    @classmethod
    def sample_p0_from_bounds(cls, bounds, n):
        mb = min(abs(b) for b in bounds)
        box = mb if mb != np.inf else 10
        return np.random.uniform(-box, box, n)

    @classmethod
    def sample_p0_from_shaped_bounds(cls, bounds):
        return np.random.uniform(bounds[0], bounds[1])

    @classmethod
    def quadratic_residuals(cls, n: int, m: int, k: int, bounds=(-np.inf, np.inf)):
        residuals = [generate_positive_definite_quadratic_form(n, k, random_orthonormal_basis) for _ in range(m)]
        gradients = [residuals[i].gradient_function() for i in range(m)]

        mb = min(abs(b) for b in bounds)
        return LeastSquaresProblem(
            f"QuadraticResiduals[n={n}, m={m}, k={k}]",
            residuals,
            gradients,
            random_normalized_vector(n) * (mb if mb != np.inf else 100),
            bounds
        )

    @classmethod
    def polynom_approx(cls, x, y, deg, bounds=(-np.inf, np.inf)):
        def residual(i):
            def f(a: np.ndarray):
                return Polynomial(a)(x[i]) - y[i]

            return f

        residuals = []
        for i in range(x.size):
            residuals.append(residual(i))

        def residual_gradient(i):
            def df(a: np.ndarray):
                return np.array([x[i] ** j for j in range(a.size)])

            return df

        residual_grads = []
        for i in range(x.size):
            residual_grads.append(residual_gradient(i))

        p0 = LeastSquaresProblem.sample_p0_from_bounds(bounds, deg)

        return LeastSquaresProblem(
            "ApproxPoly",
            residuals,
            residual_grads,
            p0,
            bounds
        )

    @classmethod
    def pattern_approximation(cls, name, x, y, pattern, pattern_grad, p0, margins):
        def residual(i):
            def f(a: np.ndarray):
                return pattern(a)(x[i]) - y[i]

            return f

        residuals = []
        for i in range(x.size):
            residuals.append(residual(i))

        def residual_gradient(i):
            def df(a: np.ndarray):
                return pattern_grad(a, x[i])  # dp/d{a_j}

            return df

        residual_grads = []
        for i in range(x.size):
            residual_grads.append(residual_gradient(i))

        return LeastSquaresProblem(
            name,
            residuals,
            residual_grads,
            p0 + cls.sample_p0_from_shaped_bounds(margins),
            (p0 + margins[0], p0 + margins[1])
        )

    def as_optimization_problem(self):
        f = lambda x: 0.5 * sum((r(x) ** 2 for r in self.residuals))
        df = lambda x: sum((r(x) * g(x) for r, g in zip(self.residuals, self.residual_gradients)))

        return Problem(
            self.name,
            f,
            self.p0,
            df,
            None
        )


class Algorithm(NamedTuple):
    name: str

    """ Returns some estimate of algorithm's work to be displayed in the table """
    solve: Callable[[Problem], np.array] | Callable[[LeastSquaresProblem], np.array]

    @classmethod
    def scipy_optimize_with_solver(cls, solver_name: str, result_extractor, **kwargs):
        def solve(problem: Problem):
            opt_res, elapsed = mesuare_time(lambda: scipy.optimize.minimize(
                problem.function,
                problem.x0,
                method=solver_name,
                jac=problem.gradient,
                hess=problem.hessian,
                **kwargs
            ))
            return result_extractor(opt_res, elapsed)

        return Algorithm(f"sp.{solver_name}", solve)

    @classmethod
    def quasi_newton_optimize_with_strategy(cls, strategy_name: str,
                                            strategy_provider: Callable[[], NewtonDirectionApproximator],
                                            result_extractor, initial_approximator=None):
        def solve(problem: Problem):
            opt_res, elapsed = mesuare_time(lambda: newton_optimize(
                problem.function,
                problem.gradient,
                strategy_provider(),
                problem.x0,
                wolfe_conditions_search(0.1, 0.9),
                precision_termination_condition,
                initial_approximator if initial_approximator is not None else none_approximation
            ))

            return result_extractor(opt_res, elapsed)

        return Algorithm(f"hm.{strategy_name}", solve)

    @classmethod
    def gradient_descent(cls, name: str, algo, result_extractor):
        def solve(problem: Problem):
            opt_res, elapsed = mesuare_time(lambda: algo(
                problem.function,
                problem.gradient,
                problem.x0,
                wolfe_conditions_search(0.1, 0.9),
                precision_termination_condition,
            ))

            return result_extractor(opt_res, elapsed)

        return Algorithm(f"hm.{name}", solve)

    @classmethod
    def dogleg(cls, result_extractor):
        def solve(problem: LeastSquaresProblem):
            opt_res, elapsed = mesuare_time(lambda: dogleg(
                problem.residuals,
                problem.residual_gradients,
                1,
                wolfe_conditions_search(0.1, 0.9),
                problem.p0,
                precision_termination_condition,
            ))

            return result_extractor(opt_res, elapsed)

        return Algorithm(f"hm.Dogleg", solve)

    @classmethod
    def gauss_newton(cls, result_extractor):
        def solve(problem: LeastSquaresProblem):
            opt_res, elapsed = mesuare_time(lambda: gauss_newton(
                problem.residuals,
                problem.residual_gradients,
                problem.p0,
                precision_termination_condition
            ))

            return result_extractor(opt_res, elapsed)

        return Algorithm(f"hm.Gauss-Newton", solve)

    @classmethod
    def sp_least_squares(cls, result_extractor, solver: str):
        def solve(problem: LeastSquaresProblem):
            def f(p):
                return np.array([r(p) for r in problem.residuals])

            def jac(p):
                return np.array([grad(p) for grad in problem.residual_gradients])

            opt_res, elapsed = mesuare_time(lambda: least_squares(
                f,
                problem.p0,
                jac,
                method=solver,
                bounds=problem.bounds
            ))
            print(opt_res)

            return result_extractor(opt_res, elapsed)

        return Algorithm(f"sp.LS[{solver}]", solve)

    def optimizer_as_ls_solver(self):
        return Algorithm(
            self.name,
            lambda p: self.solve(p.as_optimization_problem())
        )


def scipy_point_number_extractor(opt_res: OptimizeResult, elapsed: float):
    return opt_res.nit


def scipy_ls_param_extractor(opt_res: OptimizeResult, elapsed: float):
    return opt_res.x


def scipy_point_number_and_time_extractor(opt_res: OptimizeResult, elapsed: float):
    return f"{opt_res.nit}/{round(elapsed, 2)}"


def scipy_ls_point_number_and_time_extractor(opt_res: OptimizeResult, elapsed: float):
    return f"{opt_res.njev}/{round(elapsed, 2)}"


def list_point_number_extractor(opt_res: list, elapsed: float):
    return len(opt_res)


def list_point_number_and_time_extractor(opt_res: list, elapsed: float):
    return f"{len(opt_res)}/{round(elapsed, 2)}"


def compare_optimization_algorithms_in_table(algorithms: List[Algorithm],
                                             problems: List[Problem] | List[LeastSquaresProblem],
                                             tl_seconds: Optional[float], problem_discriminator="problem"):
    """
    Algorithms correspond to columns, problems correspond to rows.
    Problems can differ in function
    patterns/dimensionality/hyperparameters. That should be reflected in their names.
    Algorithms can differ by core algorithm or by hyperparameters.
    Tables can be split by either of the characteristics of problems or algorithms

    Normally, in each cell (y, x), the number of iterations of algorithm x on problem y is displayed.
    Algorithms are invoked with TL and exception handler. In the former case, «TL» will appear in the corresponding cell.
    In the latter case, that would be the «×» sign.
    """
    table = {problem_discriminator: [p.name for p in problems]}

    for algo in algorithms:
        results = table[algo.name] = []
        print(f"=========== Testing {algo.name} ===========")
        for p in problems:
            try:
                with time_limit(tl_seconds, 'sleep'):
                    result = algo.solve(p)
            except TimeoutException:
                result = "TL"
            except Exception as e:
                print("Exception occurred:", e)
                result = "×"

            print(f"Got result {result} at {problem_discriminator}={p.name}")
            results.append(result)

    print(tabulate(table, headers="keys", tablefmt="grid"))


def poly_pattern(a):
    return Polynomial(a)


def poly_pattern_grad(a, x):
    return np.array([x ** j for j in range(a.size)])


def exp_pattern(a):
    assert a.size == 4
    return lambda x: a[0] * np.exp(a[1] * x) * np.cos(a[2] * x + a[3])


def exp_pattern_grad(a, x):
    return np.array([
        np.exp(a[1] * x) * np.cos(a[2] * x + a[3]),
        a[0] * x * np.exp(a[1] * x) * np.cos(a[2] * x + a[3]),
        -a[0] * x * np.exp(a[1] * x) * np.sin(a[2] * x + a[3]),
        -a[0] * np.exp(a[1] * x) * np.sin(a[2] * x + a[3]),
    ])


def noisify(y, sigma):
    return y + np.random.normal(0, sigma, (y.size,))

exp_p0 = np.array([1, -0.5, 3, 1])

def constrained_exp(xs, intervals, ys = None):
    return LeastSquaresProblem.pattern_approximation(f"Exp+N[roi={intervals}]", xs, noisify(exp_pattern(exp_p0)(xs), intervals[0] / 20) if ys is None else ys,
                                                     exp_pattern, exp_pattern_grad, exp_p0, (-intervals, intervals))

