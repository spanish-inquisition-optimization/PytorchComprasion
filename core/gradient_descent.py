from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import scipy.constants
from math import exp, floor, sqrt

from core.utils import NUMERIC_GRADIENT_COMPUTING_PRECISION


class SearchRegion2d(NamedTuple):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


precision = 1e-9


def gradient_descent(target_function: Callable[[np.ndarray], float],
                     gradient_function: Callable[[np.ndarray], np.ndarray],
                     direction_function,
                     x0: np.ndarray,
                     linear_search,
                     terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    points = [np.array(x0)]
    last_step_length = 0
    last_direction = 0

    while not terminate_condition(target_function, points):
        last_point = points[-1]
        iteration = len(points) - 1
        last_direction = np.array(
            direction_function(last_point, last_step_length=last_step_length, last_direction=last_direction,
                               iteration=iteration))
        norm = np.linalg.norm(last_direction)

        if norm == 0:
            continue
        if norm > 1e20:
            return points

        # false positive warning: np.dot returns scalar in this case
        last_step_length = linear_search(lambda l: target_function(last_point + last_direction * l),
                                         lambda l: np.dot(last_direction, gradient_function(
                                             last_point + last_direction * l)), iteration=iteration)
        next_point = last_point + last_direction * last_step_length
        points.append(next_point)
    return points


def gradient_descent_with_momentum(gamma: float, nesterov=False):
    assert 0 <= gamma < 1

    def search_function(target_function: Callable[[np.ndarray], float],
                        gradient_function: Callable[[np.ndarray], np.ndarray],
                        direction_function,
                        x0: np.ndarray,
                        linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                        terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
        def get_direction(x: np.ndarray, last_step_length=0, last_direction=None, **kwargs):
            return -(gamma * -last_direction + (1 - gamma) * -direction_function(
                x + last_step_length * gamma * last_direction if nesterov else x, **kwargs))

        return gradient_descent(target_function, gradient_function, get_direction, x0, linear_search,
                                terminate_condition)

    return search_function


def adagrad_descent(target_function: Callable[[np.ndarray], float],
                    gradient_function: Callable[[np.ndarray], np.ndarray],
                    direction_function,
                    x0: np.ndarray,
                    linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                    terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    G = 0

    def get_direction(x: np.ndarray, **kwargs):
        nonlocal G
        current_direction = -direction_function(x, **kwargs)
        G = G + np.outer(current_direction, current_direction)
        return -np.divide(current_direction, np.array(np.sqrt(np.diagonal(G) + 1e-8)))

    return gradient_descent(target_function, gradient_function, get_direction, x0, linear_search, terminate_condition)


def rms_prop_descent(gamma: float):
    assert 0 <= gamma < 1

    def search_function(target_function: Callable[[np.ndarray], float],
                        gradient_function: Callable[[np.ndarray], np.ndarray],
                        direction_function,
                        x0: np.ndarray,
                        linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                        terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
        G = 0

        def get_direction(x: np.ndarray, **kwargs):
            nonlocal G
            current_direction = -direction_function(x, **kwargs)
            G = gamma * G + (1 - gamma) * np.square(current_direction)
            return -np.divide(current_direction, np.sqrt(G + 1e-8))

        return gradient_descent(target_function, gradient_function, get_direction, x0, linear_search,
                                terminate_condition)

    return search_function


def adam_descent(alpha: float, beta: float):
    assert (0 < alpha < 1 and 0 < beta < 1)

    def search_function(target_function: Callable[[np.ndarray], float],
                        gradient_function: Callable[[np.ndarray], np.ndarray],
                        direction_function,
                        x0: np.ndarray,
                        linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                        terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
        v = 0
        s = 0

        def get_direction(x: np.ndarray, iteration=0, **kwargs):
            nonlocal v, s
            current_direction = -direction_function(x, iteration=iteration, **kwargs)
            v = alpha * v + (1 - alpha) * current_direction
            s = beta * s + (1 - beta) * np.square(current_direction)
            v_normalized = v / (1 - alpha ** (iteration + 1))
            s_normalized = s / (1 - beta ** (iteration + 1))
            return -np.divide(v_normalized, np.sqrt(s_normalized + 1e-8))

        return gradient_descent(target_function, gradient_function, get_direction, x0, linear_search,
                                terminate_condition)

    return search_function


def steepest_descent_base(base_search):
    def result(target_function: Callable[[np.ndarray], float],
               gradient_function: Callable[[np.ndarray], np.ndarray],
               x0: np.ndarray,
               linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
               terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
        return base_search(target_function, gradient_function, lambda x, **kwargs: -gradient_function(x), x0,
                           linear_search,
                           lambda f, steps: terminate_condition(f, steps) or (
                                   len(steps) > 2 and np.linalg.norm(steps[-1] - steps[-2]) < precision))

    return result


def steepest_descent(target_function: Callable[[np.ndarray], float],
                     gradient_function: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                     terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    return steepest_descent_base(gradient_descent)(target_function, gradient_function, x0, linear_search,
                                                   terminate_condition)


def steepest_descent_with_momentum(gamma: float, nesterov=False):
    return steepest_descent_base(gradient_descent_with_momentum(gamma, nesterov))


def steepest_descent_adagrad(target_function: Callable[[np.ndarray], float],
                             gradient_function: Callable[[np.ndarray], np.ndarray],
                             x0: np.ndarray,
                             linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                             terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    return steepest_descent_base(adagrad_descent)(target_function, gradient_function, x0, linear_search,
                                                  terminate_condition)


def steepest_descent_rms_prop(gamma: float):
    return steepest_descent_base(rms_prop_descent(gamma))


def steepest_descent_adam(alpha: float, beta: float):
    return steepest_descent_base(adam_descent(alpha, beta))


""" Could be:
def find_upper_bound(f: Callable[[float], float], derivative: Callable[[float], float]):
    minimal_f = f(0)
    r = 0.05

    this_f = f(r)
    while derivative(r) < 0 and this_f < minimal_f:
        r *= 1.3
        minimal_f = min(minimal_f, this_f)
        this_f = f(r)

    return r
"""


def gradient_descent_minibatch_base(base_search):
    def result(target_functions: List[Callable[[np.ndarray], float]],
               gradient_functions: List[Callable[[np.ndarray], np.ndarray]],
               batch_size: int,
               x0: np.ndarray,
               learning_rate_scheduler: Callable[[int], float],
               terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
        assert len(target_functions) == len(gradient_functions)
        ordered_gradient_functions = np.random.permutation(gradient_functions)

        def sum_functions(funcs):
            return lambda x: sum(f(x) for f in funcs)

        def get_direction(x: np.ndarray, iteration=0, **kwargs):
            return -sum(ordered_gradient_functions[(iteration * batch_size + i) % len(gradient_functions)](x) for i in
                        range(batch_size))

        return base_search(sum_functions(target_functions), sum_functions(gradient_functions),
                           get_direction, x0, learning_rate_scheduler,
                           terminate_condition)

    return result


def gradient_descent_minibatch(target_functions: List[Callable[[np.ndarray], float]],
                               gradient_functions: List[Callable[[np.ndarray], np.ndarray]],
                               batch_size: int,
                               x0: np.ndarray,
                               learning_rate_scheduler: Callable[[int], float],
                               terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    return gradient_descent_minibatch_base(gradient_descent)(target_functions, gradient_functions, batch_size, x0,
                                                             learning_rate_scheduler, terminate_condition)


def gradient_descent_minibatch_with_momentum(gamma: float, nesterov=False):
    return gradient_descent_minibatch_base(gradient_descent_with_momentum(gamma, nesterov))


def gradient_descent_minibatch_adagrad(target_functions: List[Callable[[np.ndarray], float]],
                                       gradient_functions: List[Callable[[np.ndarray], np.ndarray]],
                                       batch_size: int,
                                       x0: np.ndarray,
                                       learning_rate_scheduler: Callable[[int], float],
                                       terminate_condition: Callable[
                                           [Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    return gradient_descent_minibatch_base(adagrad_descent)(target_functions, gradient_functions, batch_size, x0,
                                                            learning_rate_scheduler, terminate_condition)


def gradient_descent_minibatch_rms_prop(gamma: float):
    return gradient_descent_minibatch_base(rms_prop_descent(gamma))


def gradient_descent_minibatch_adam(alpha: float, beta: float):
    return gradient_descent_minibatch_base(adam_descent(alpha, beta))


def step_learning_scheduler(initial_rate: float, step_rate: float, step_length: int, batch_size: int, total_funcs: int):
    return lambda *args, iteration=0, **kwargs: initial_rate * step_rate ** floor(
        floor(iteration * batch_size / total_funcs) / step_length)


def exponential_learning_scheduler(initial_rate: float, step_rate: float, batch_size: int, total_funcs: int):
    return lambda *args, iteration=0, **kwargs: initial_rate * exp(
        -step_rate * floor(iteration * batch_size / total_funcs))


def find_upper_bound(f: Callable[[float], float]):
    prev_value = f(0)
    cur_value = f(1)
    r = 1
    while cur_value <= prev_value:
        prev_value = cur_value
        r *= 1.2
        cur_value = f(r)
    return r


def fixed_step_search(step_length):
    return lambda f, derivative, **kwargs: step_length


def bin_search(f: Callable[[float], float], derivative: Callable[[float], float], **kwargs):
    # assume derivative(0) < 0 and derivative is rising
    l = 0
    r = find_upper_bound(f)

    while r - l > precision:
        m = (l + r) / 2
        if derivative(m) < 0:
            l = m
        else:
            r = m
    return r


def bin_search_with_iters(iters):
    def search(f: Callable[[float], float], derivative: Callable[[float], float], **kwargs):
        # assume derivative(0) < 0 and derivative is rising
        i = 0
        l = 0
        r = find_upper_bound(f)

        while i < iters and r - l > precision and abs(derivative(r)) > precision:
            m = (l + r) / 2
            i += 1
            if derivative(m) < 0:
                l = m
            else:
                r = m
        return r

    return search


def golden_ratio_search(f: Callable[[float], float], _derivative: Callable[[float], float], **kwargs):
    l = 0
    r = find_upper_bound(f)
    delta = (r - l) / scipy.constants.golden
    x1, x2 = r - delta, l + delta
    f1, f2 = f(x1), f(x2)

    while r - l > precision:
        if f1 < f2:
            r = x2
            f2 = f1
            x1, x2 = r - ((r - l) / scipy.constants.golden), x1
            f1 = f(x1)
        else:
            l = x1
            f1 = f2
            x1, x2 = x2, l + ((r - l) / scipy.constants.golden)
            f2 = f(x2)

    return r


def fibonacci_search(n_iters):
    def search(f, _derivative, **kwargs):
        l = 0
        r = find_upper_bound(f)
        length = r - l
        fibs = [1, 1]
        while len(fibs) <= n_iters:
            fibs.append(fibs[-1] + fibs[-2])
        x1 = l + length * fibs[-3] / fibs[-1]
        x2 = l + length * fibs[-2] / fibs[-1]
        y1, y2 = f(x1), f(x2)
        for k in range(n_iters - 2):
            if y1 > y2:
                l = x1
                x1 = x2
                x2 = l + (r - l) * fibs[-k - 3] / fibs[-k - 2]
                y1, y2 = y2, f(x2)
            else:
                r = x2
                x2 = x1
                x1 = l + (r - l) * fibs[-k - 4] / fibs[-k - 2]
                y1, y2 = f(x1), y1

        return r

    return search


def wolfe_conditions_search(c1, c2):
    assert 0 < c1 < c2 < 1

    def search(f: Callable[[float], float], derivative: Callable[[float], float], **kwargs):
        # Need to find x such that:
        # 1) f(x) <= f(0) + c1 * x * derivative(0)
        # 2) derivative(x) >= c2 * derivative(0)
        initial_value = f(0)
        initial_slope = derivative(0)
        desired_slope = initial_slope * c2

        def desired_descent(x):
            return initial_value + initial_slope * c1 * x

        def zoom(left, right):
            assert (left < right)
            while True:
                mid = (left + right) / 2
                value = f(mid)
                if value > desired_descent(mid) or value >= f(left):
                    right = mid
                else:
                    slope = derivative(mid)
                    if abs(slope) <= -desired_slope:
                        return mid
                    elif slope * (right - left) >= 0:
                        right = left
                    left = mid

        max_step = 1
        while f(max_step) <= desired_descent(max_step):
            max_step *= 1.3

        prev_step = 0
        cur_step = max_step / 2
        prev_value = initial_value

        while True:
            cur_value = f(cur_step)
            if cur_value > desired_descent(cur_step) or (prev_step != 0 and cur_value >= prev_value):
                return zoom(prev_step, cur_step)
            cur_slope = derivative(cur_step)
            if abs(cur_slope) <= -desired_slope:
                return cur_step
            if cur_slope >= 0:
                return zoom(cur_step, prev_step)
            prev_step = cur_step
            prev_value = cur_value
            cur_step = (prev_step + max_step) / 2

    return search


# TODO: n-ary search through log space?


def precision_termination_condition(_target_function: Callable[[np.ndarray], float], points: List[np.ndarray]):
    return len(points) > 2 and np.linalg.norm(points[-1] - points[-2]) < NUMERIC_GRADIENT_COMPUTING_PRECISION


def point_number_terminate_condition(m):
    return lambda f, points: len(points) >= m
