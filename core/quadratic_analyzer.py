from core.gradient_descent import *
from core.visualizer import *

# ax^2 + bxy + cy^2 + dx + ey
def create_quadratic(a, b, c, d, e):
    return lambda x: a * x[0] ** 2 + b * x[0] * x[1] + c * x[1] ** 2 + d * x[0] + e * x[1]

# ax^2 + bxy + cy^2 + dx + ey
def create_quadratic_derivative(a, b, c, d, e):
    return lambda x: np.array([2 * a * x[0] + b * x[1] + d, 2 * c * x[1] + b * x[0] + e])

def analyze_quadratic(roi, x0, fixed_steps, bin_iters, fib_iters, a, b, c, d, e):
    f = create_quadratic(a, b, c, d, e)
    df = create_quadratic_derivative(a, b, c, d, e)

    def visualize_optimizer_with(linear_search):
        if d == 0 and e == 0:
            true_min = 0
        else:
            true_min = None

        return visualize_optimizing_process(f, roi, np.array(gradient_descent(f, df, x0, linear_search, lambda f, points: len(points) > 20)), true_min)

    # print("Function plot:")
    visualize_function_3d(f, roi)

    cases = [(f"Optimizing with fixed step = {step}", fixed_step_search(step)) for step in fixed_steps] + [
        ("Optimizing with binary search", bin_search),
        (f"Optimizing with binary search limited by {bin_iters} iterations", bin_search_with_iters(bin_iters)),
        ("Optimizing with golden ration", golden_ratio_search),
        (f"Optimizing with fibonacci search limited by {fib_iters} iterations:", fibonacci_search(fib_iters)),
        ("Optimizing with backtracking method", wolfe_conditions_search(0.1, 0.9))
    ]

    for title, linear in cases:
        print(title)
        visualize_optimizer_with(linear).suptitle(title)
