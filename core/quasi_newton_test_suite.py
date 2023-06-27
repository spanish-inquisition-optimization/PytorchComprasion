from core.gradient_descent import wolfe_conditions_search, point_number_terminate_condition, fixed_step_search
from core.high_order_optimization import *
from core.optimizer_evaluator import *
from core.utils import *
from core.visualizer import *
import numpy as np
from tabulate import tabulate


def evaluate_quasi_newton_methods_on_2dim(methods):
    print("—————— 2 Dimensions ——————")

    q = QuadraticForm(np.array([
        [10, 2],
        [2, 2]
    ]))
    roi = SearchRegion2d((-2, 2), (-2, 2))
    visualize_function_3d(q, roi)

    qg = q.gradient_function()

    for (name, get_approximator, get_initial_approximator) in methods:
        visualize_optimizing_process(q, roi, np.array(newton_optimize(
            q, qg, get_approximator(q, qg), np.array([1.5, -1.]), wolfe_conditions_search(0.1, 0.9), point_number_terminate_condition(10), get_initial_approximator(q, qg)
        )), 0).suptitle(name)

def evaluate_quasi_newton_methods_on_rosen(methods):
    print("—————— Many Dimensions Rosenbrock ——————")
    from scipy.optimize import rosen, rosen_der

    dims = [2, 10, 50, 100]
    data = { 'dim': dims }
    for (name, get_approximator, get_initial_approximator) in methods:
        results = data[name] = []
        for n in dims:
            f = rosen
            rg = rosen_der
            try:
                points = newton_optimize(f, rg, get_approximator(f, rg),
                                         # random_normalized_vector(n),  # Or just zero?
                                         np.zeros((n,)),
                                         wolfe_conditions_search(0.1, 0.9),
                                         # fibonacci_search(30),
                                         # fixed_step_search(1),
                                         precision_termination_condition,
                                         get_initial_approximator(f, rg)
                                         )
                results.append(str(len(points)))
                print(f"Iterations until convergence for n={n}: {len(points)}")
            except Exception as e:
                results.append("×")
                print(f"At n={n}: Failed with {e}")

    print(tabulate(data, headers="keys", tablefmt="grid"))

def evaluate_quasi_newton_methods_on_form(methods, tl_sec):
    print("—————— Many Dimensions ——————")
    dims = [2, 10, 50, 100, 1000]
    data = { 'dim': dims }
    for (name, get_approximator, get_initial_approximator) in methods:
        results = data[name] = []
        for n in dims:
            form = generate_positive_definite_quadratic_form(n, 1000, random_orthonormal_basis)

            try:
                with time_limit(tl_sec, 'sleep'):
                    fg = form.gradient_function()
                    points = newton_optimize(form, fg, get_approximator(form, fg), random_normalized_vector(form.n),
                                             wolfe_conditions_search(0.1, 0.9),
                                             # fibonacci_search(30),
                                             precision_termination_condition, get_initial_approximator(form, fg))
                    print(f"Iterations until convergence for n={n}: {len(points)}")
                    results.append(str(len(points)))
            except Exception:
                results.append("TL")
                print("TL")

    print(tabulate(data, headers="keys", tablefmt="grid"))






def evaluate_methods_on_sines(methods):
    print("—————— Many Dimensions Sines ——————")
    from scipy.optimize import rosen, rosen_der

    dims = [(2, 2), (7, 7), (8, 8)]
    data = { 'dim': dims }
    for (name, optimizer) in methods:
        results = data[name] = []
        for (n, m) in dims:
            coeffs = np.random.random(m)
            residuals = [(lambda x: np.sin(coeffs[i] * np.average(x))) for i in range(m)]
            gradients = [symmetric_gradient_computer(residuals[i]) for i in range(m)]

            f = lambda x: 0.5 * sum((r(x) ** 2 for r in residuals))
            df = lambda x: sum((r(x) * g(x) for r, g in zip(residuals, gradients)))

            x0 = random_normalized_vector(n) / n

            # try:
            points = np.array(optimizer(residuals, gradients, f, df, x0, lambda f, ps: f(ps[-1]) < 1e-9))
            results.append(str(len(points)))
            print(f"Iterations until convergence for {n}×{m}: {len(points)}")
            # except Exception as e:
            #     results.append("×")
            #     print(f"At n={n}: Failed with {e}")

    print(tabulate(data, headers="keys", tablefmt="grid"))


def evaluate_methods_on_square_sums(methods):
    print("—————— Many Dimensions Sq ——————")
    from scipy.optimize import rosen, rosen_der

    dims = [(2, 2), (7, 7), (8, 8), (10, 10)]
    data = { 'dim': dims }
    for (name, optimizer) in methods:
        results = data[name] = []
        for (n, m) in dims:
            residuals = [generate_positive_definite_quadratic_form(n, 200, random_orthonormal_basis) for _ in range(m)]
            gradients = [residuals[i].gradient_function() for i in range(m)]

            f = lambda x: 0.5 * sum((r(x) ** 2 for r in residuals))
            df = lambda x: sum((r(x) * g(x) for r, g in zip(residuals, gradients)))

            x0 = random_normalized_vector(n)

            # try:
            points = np.array(optimizer(residuals, gradients, f, df, x0, lambda f, ps: f(ps[-1]) < 1e-9))
            results.append(str(len(points)))
            print(f"Iterations until convergence for {n}×{m}: {len(points)}")
            # except Exception as e:
            #     results.append("×")
            #     print(f"At n={n}: Failed with {e}")

    print(tabulate(data, headers="keys", tablefmt="grid"))


