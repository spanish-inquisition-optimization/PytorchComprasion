from collections import namedtuple
from typing import List, Callable, Optional, NamedTuple

import numpy as np
from tabulate import tabulate


class Problem(NamedTuple):
    name: str
    function: Callable
    gradient: Callable
    hessian: Optional[Callable]


class Algorithm(NamedTuple):
    name: str

    """ Returns some estimate of algorithm's work to be displayed in the table """
    solve: Callable[[Problem], np.array]

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
            result = algo.solve(p)
            print(f"Got result {result} at {problem_discriminator}={p.name}")
            results.append(result)

    print(tabulate(table, headers="keys", tablefmt="grid"))

