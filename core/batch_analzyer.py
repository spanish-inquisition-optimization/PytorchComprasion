import numpy.linalg.linalg

from core.utils import CallsCount
from core.gradient_descent import *
from core.utils import *


def test_batch(fs, dfs, x0, scheduler):
    f = fn_sum(*fs)
    df = fn_sum(*dfs)

    target_point = steepest_descent(
        f, df, x0,
        bin_search,
        lambda f, steps: len(steps) > 100
    )[-1]

    target_val = f(target_point)

    result = []
    for batch in range(1, len(fs) + 1):
        points = gradient_descent_minibatch(
            fs, dfs, batch, x0,
            scheduler(batch),
            lambda f, steps: abs(f(steps[-1]) - target_val) < 0.01 or len(steps) > 200
        )

        result.append((f"batch = {batch}", len(points)))

    return result



def test_perfomance(funs, dfuns, batch_size, x0):
    wrap_input = lambda: ([CallsCount(f) for f in funs], [CallsCount(f) for f in dfuns])
    f1, df1 = wrap_input()
    f2, df2 = wrap_input()
    f3, df3 = wrap_input()
    f4, df4 = wrap_input()
    f5, df5 = wrap_input()
    f6, df6 = wrap_input()

    def terminate(f, steps):
        return f(steps[-1]) < 0.01 or len(steps) > 1000

    exp_scheduler1 = exponential_learning_scheduler(1, 0.2, batch_size, len(funs))
    exp_scheduler2 = exponential_learning_scheduler(0.3, 0.2, batch_size, len(funs))

    p1 = gradient_descent_minibatch(
        f1, df1, batch_size, x0,
        exp_scheduler2,
        terminate
    )

    p2 = gradient_descent_minibatch_with_momentum(0.2)(
        f2, df2, batch_size, x0,
        exp_scheduler2,
        terminate
    )

    p3 = gradient_descent_minibatch_with_momentum(0.7, True)(
        f3, df3, batch_size, x0,
        exp_scheduler2,
        terminate
    )

    p4 = gradient_descent_minibatch_adagrad(
        f4, df4, batch_size, x0,
        fixed_step_search(5),
        terminate
    )

    p5 = gradient_descent_minibatch_rms_prop(0.2)(
        f5, df5, batch_size, x0,
        exp_scheduler1,
        terminate
    )

    p6 = steepest_descent_adam(0.9, 0.999)(fn_sum(*f6), fn_sum(*df6), x0, bin_search_with_iters(5), terminate)

    return [
        (f"Minibatch", f"{sum(f.calls for f in f1) + sum(f.calls for f in df1)}/{len(p1)}"),
        (f"Momentum", f"{sum(f.calls for f in f2) + sum(f.calls for f in df2)}/{len(p2)}"),
        (f"Nesterov", f"{sum(f.calls for f in f3) + sum(f.calls for f in df3)}/{len(p3)}"),
        (f"AdaGrad", f"{sum(f.calls for f in f4) + sum(f.calls for f in df4)}/{len(p4)}"),
        (f"RMSProp", f"{sum(f.calls for f in f5) + sum(f.calls for f in df5)}/{len(p5)}"),
        (f"Adam", f"{sum(f.calls for f in f6) + sum(f.calls for f in df6)}/{len(p6)}")
    ]
