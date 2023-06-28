from collections import namedtuple
from typing import List, Callable, Optional, NamedTuple


class Problem(NamedTuple):
    name: str
    function: Callable
    gradient: Callable
    hessian: Optional[Callable]


class Algorithm(NamedTuple):
    name: str

    """ Returns number of iterations required to achieve some result """
    solve: Callable[[Problem], int]


def compare_optimization_algorithms_in_table(algorithms: List[Algorithm], problems: List[Problem],
                                             tl_seconds: Optional[float]):
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
    pass
