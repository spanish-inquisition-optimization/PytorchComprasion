from typing import Tuple, NamedTuple, Callable, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import newaxis

from core.gradient_descent import gradient_descent, precision
from core.optimizer_evaluator import generate_positive_definite_quadratic_form, random_normalized_vector
from core.utils import supports_argument, symmetric_gradient_computer, AutoVectorizedFunction, n_calls_mocker


class SearchRegion2d(NamedTuple):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


def debug_mesh(roi: SearchRegion2d, n=1000):
    return np.meshgrid(
        np.linspace(roi.x_range[0], roi.x_range[1], n),
        np.linspace(roi.y_range[0], roi.y_range[1], n)
    )


def auto_meshgrid(f, roi: SearchRegion2d):
    X, Y = debug_mesh(roi, 1000) if supports_argument(f, np.stack(debug_mesh(roi, 1))) else debug_mesh(roi, 300)
    return X, Y


def visualize_function_3d(f, roi: SearchRegion2d):
    X, Y = auto_meshgrid(f, roi)
    ax = plt.figure().add_subplot(projection='3d')
    return ax.plot_surface(X, Y, AutoVectorizedFunction(f, 1)(np.stack((X, Y))))


def visualize_optimizing_process(f, roi: SearchRegion2d, points, true_minimum=None):
    X, Y = auto_meshgrid(f, roi)
    vectorized_f = AutoVectorizedFunction(f, 1)

    if true_minimum is None:
        fig, (ax1, ax3) = plt.subplots(1, 2)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(vectorized_f(points.T))
    ax1.set_title(f"Function value")
    ax1.grid()

    if true_minimum is not None:
        ax2.plot(vectorized_f(points.T) - true_minimum)
        ax2.set_yscale("log", nonpositive='clip')
        ax2.set_title(f"Logarithmic error")
        ax2.grid()

    ax3.plot(points[:, 0], points[:, 1], 'o-')
    # print(f"Optimizer trajectory:")
    # print(points)
    print(f"Best value found: x* = {points[-1]} with f(x*) = {vectorized_f(points[-1])}")

    levels = vectorized_f(points.T)
    ax3.contour(X, Y, vectorized_f(np.stack((X, Y))), levels=sorted(set(levels)))
    ax3.set_title(f"Visited contours")

    return fig  # For further customization


def plot_section_graphs(f, discrete_param_values, continuous_param_values):
    fig, plots = plt.subplots(1, len(discrete_param_values))
    for i, discrete_param_value in enumerate(discrete_param_values):
        plots[i].plot(continuous_param_values, [f(discrete_param_value, x) for x in continuous_param_values])
        plots[i].set_title(f"Section at {discrete_param_value}")
        plots[i].set_yscale("log")
        plots[i].set_xscale("log")

        plots[i].grid()


def test_linear_search(linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                       measure_performance=False):
    alpha = 5

    def quadratic(x):
        return alpha * (x[0] - 5) ** 2 + (x[1] - 7) ** 2

    def quadratic_derivative(x):
        return np.array([2 * alpha * (x[0] - 5), 2 * (x[1] - 7)])

    quadratic_roi = SearchRegion2d((-50, 50), (-50, 50))

    def trig(x):
        return np.sin(0.5 * x[0] ** 2 - 0.25 * x[1] ** 2 + 3) * np.cos(2 * x[0] + 1 - np.exp(x[1]))

    trig_derivative = symmetric_gradient_computer(trig)
    trig_roi = SearchRegion2d((-1.5, 1.5), (-1.5, 2))

    from scipy.optimize import rosen, rosen_der as rosen_derivative
    rosen_roi = SearchRegion2d((-2, 1), (-1, 1))

    class SearchSetup(NamedTuple):
        name: str
        f: Callable[[np.ndarray], float]
        gradient: Callable[[np.ndarray], np.ndarray]
        roi: SearchRegion2d
        num_steps: int
        start_point: np.ndarray
        true_minimum: float = None

    setups = [SearchSetup("Quadratic", quadratic, quadratic_derivative, quadratic_roi, 50, np.array([-20, -20]), 0),
              SearchSetup("Trigonometric", trig, trig_derivative, trig_roi, 100, np.array([-0.1, -0.4])),
              SearchSetup("Rosenbrock", rosen, rosen_derivative, rosen_roi, 20, np.array([-1.5, 0.25]), 0)
              ]
    if measure_performance:
        dims = 100
        k = 100
        mvf = generate_positive_definite_quadratic_form(dims, k)
        setups.append(SearchSetup(f"{dims}-dimensional quadratic with k = {k}", mvf, mvf.gradient_function(), None, 1000, random_normalized_vector(dims), 0))

    for setup in setups:
        resultant_true_minimum = setup.true_minimum

        if not measure_performance:
            visualize_function_3d(setup.f, setup.roi)

        if measure_performance and resultant_true_minimum is None:
            resultant_true_minimum = setup.f(gradient_descent(
                setup.f, setup.gradient, setup.start_point, linear_search,
                lambda f, points: len(points) > setup.num_steps
            )[-1])
            assert np.linalg.norm(resultant_true_minimum) < 1e5

        def terminate_condition(f: Callable[[np.ndarray], float], points: List[np.ndarray]) -> bool:
            have_true_minimum = resultant_true_minimum is not None
            return (have_true_minimum and f(points[-1]) - resultant_true_minimum < precision) \
                or len(points) > setup.num_steps

        mocked_f = n_calls_mocker(setup.f)
        mocked_grad = n_calls_mocker(setup.gradient)

        trajectory = gradient_descent(
            mocked_f, mocked_grad, setup.start_point, linear_search,
            terminate_condition
        )

        if measure_performance:
            if np.linalg.norm(trajectory[-1]) > 1e7:
                print(f"GD diverged at {setup.name} (got vector norm {np.linalg.norm(trajectory[-1])} after {len(trajectory)} steps), such a pity…")
            else:
                print(
                f"{setup.name} : f called {mocked_f.n_calls} times, f' called {mocked_grad.n_calls} times, total score: {mocked_f.n_calls + mocked_grad.n_calls}")
        else:
            visualize_optimizing_process(setup.f, setup.roi, np.array(trajectory), resultant_true_minimum)
