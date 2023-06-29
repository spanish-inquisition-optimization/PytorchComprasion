from typing import List, Callable, Optional, NamedTuple

import numpy as np
import time
from numpy import newaxis
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt


class CallsCount:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def fn_sum(*args):
    return lambda x: sum((f(x) for f in args))


def partially_vectorize(f, f_input_dims):
    def vectorized_f(t):
        if len(t.shape) == f_input_dims:
            return f(t)
        else:
            splitted = np.split(t, t.shape[-1], axis=-1)
            slices_along_last_axis = [splitted[i][..., 0] for i in range(t.shape[-1])]
            return np.concatenate([vectorized_f(s)[..., newaxis] for s in slices_along_last_axis], axis=-1)

    return vectorized_f


def supports_argument(f, smple_arg):
    try:
        f(smple_arg)
        return True
    except:
        return False


class AutoVectorizedFunction:
    def __init__(self, f, f_input_dims=None):
        self.f = f
        self.f_input_dims = f_input_dims

    def __call__(self, t):
        try:
            return self.f(t)
        except:
            assert self.f_input_dims is not None
            return partially_vectorize(self.f, self.f_input_dims)(t)


def coordinate_vector_like(coordinate_index: int, reference: np.ndarray):
    res = np.zeros_like(reference)
    res[coordinate_index] = 1
    return res


NUMERIC_GRADIENT_COMPUTING_PRECISION = 1e-5


def symmetrically_compute_partial_derivative(f: Callable[[np.ndarray], float], h: float, x: np.ndarray, i: int):
    # This trick only works on functions defined
    # in terms of scalar (or dimension-independent) np operations (aka ufuncs) which can thus be vectorized…
    # return (f(x[:, newaxis] + h * np.eye(n)) - f(x[:, newaxis] - h * np.eye(n))) / (2 * h)

    # This one is a more straightforward way:
    return (f(x + h * coordinate_vector_like(i, x)) - f(x - h * coordinate_vector_like(i, x))) / (2 * h)


def symmetrically_compute_gradient(f: Callable[[np.ndarray], float], h: float, x: np.ndarray):
    return np.array([
        symmetrically_compute_partial_derivative(f, h, x, i)
        for i in range(x.size)
    ])


def symmetrically_compute_jacobian(phi: Callable[[np.ndarray], np.ndarray], h: float, x: np.ndarray):
    n = x.size
    m = phi(x).size
    columns = []
    for i in range(m):
        columns.append(symmetrically_compute_partial_derivative(phi, h, x, i))
    return np.array(columns).T
    # return np.array(symmetrically_compute_gradient(lambda xx: ) for i in range(m))


def symmetrically_compute_second_order_partial_derivative(f: Callable[[np.ndarray], float], h: float, x: np.ndarray,
                                                          i: int, j: int):
    return symmetrically_compute_partial_derivative(lambda xx: symmetrically_compute_partial_derivative(f, h, xx, i), h,
                                                    x, j)


def symmetrically_compute_hessian(f: Callable[[np.ndarray], float], h: float, x: np.ndarray):
    return np.array([
        [symmetrically_compute_second_order_partial_derivative(f, h, x, i, j) for j in range(x.size)]
        for i in range(x.size)
    ])


def symmetrically_compute_hessian_by_gradient(f: Callable[[np.ndarray], float],
                                              gradient: Callable[[np.ndarray], np.ndarray], h: float, x: np.ndarray):
    return symmetrically_compute_jacobian(gradient, h, x)


def symmetric_gradient_computer(f: Callable[[np.ndarray], float], h: float = NUMERIC_GRADIENT_COMPUTING_PRECISION):
    return lambda x: symmetrically_compute_gradient(f, h, x)


def symmetric_hessian_computer(f: Callable[[np.ndarray], float], h: float = NUMERIC_GRADIENT_COMPUTING_PRECISION):
    return lambda x: symmetrically_compute_hessian(f, h, x)


class ApproxDump(NamedTuple):
    label: str
    x: np.ndarray
    w: np.ndarray
    loss_history: List[float]


def plot_approx(x: np.ndarray, y: np.ndarray, approxs: List[ApproxDump]):
    fig, (loss, graph) = plt.subplots(1, 2)

    for approx in approxs:
        loss.plot(approx.loss_history, label=approx.label)
        graph.plot(x, Polynomial(approx.w)(x), label=approx.label)

    loss.grid()
    loss.set_yscale('log')
    loss.legend()
    graph.plot(x, y, label='actual')
    graph.legend()

    return fig


def mesuare_time(f: Callable):
    start = time.time()
    res = f()
    end = time.time()
    return res, end - start


def n_calls_mocker(f):
    def mocked(*args, **kwargs):
        mocked.n_calls += 1
        return f(*args, **kwargs)

    mocked.n_calls = 0
    return mocked


def smoothly_criminally_call(f, *args, **kwargs):
    res = f(*args, **kwargs)
    if hasattr(f, "n_calls"):
        f.n_calls -= 1
    return res


from contextlib import contextmanager
import threading
import _thread


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


if __name__ == '__main__':

    import time

    # ends after 1 second
    with time_limit(1, 'sleep'):
        while True:
            pass


    @n_calls_mocker
    def f(x):
        print(f"Hello, {x}!")


    for i in range(5):
        f(i)

    smoothly_criminally_call(f, 12412)

    assert f.n_calls == 5
