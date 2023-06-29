from collections import namedtuple
from typing import List, Callable, Optional, NamedTuple

import numpy as np
import scipy.optimize
from tabulate import tabulate

from core.gradient_descent import wolfe_conditions_search, precision_termination_condition
from core.high_order_optimization import newton_optimize, NewtonDirectionApproximator, none_approximation
from core.optimizer_evaluator import QuadraticForm, random_normalized_vector
from core.utils import time_limit, TimeoutException, mesuare_time

from scipy.optimize import rosen
from scipy.optimize import rosen_der

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


class Algorithm(NamedTuple):
    name: str

    """ Returns some estimate of algorithm's work to be displayed in the table """
    solve: Callable[[Problem], np.array]

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
    def quasi_newton_optimize_with_strategy(cls, strategy_name: str, strategy_provider: Callable[[], NewtonDirectionApproximator], result_extractor, initial_approximator=None):
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





def compare_optimization_algorithms_in_table(algorithms: List[Algorithm], problems: List[Problem],
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

