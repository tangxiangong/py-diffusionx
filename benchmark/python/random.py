import sys
import time

import numpy as np
from diffusionx.random import randn, stable_rand
from diffusionx.random import uniform as rand
from scipy.stats import levy_stable


def timeit(func, bench_size):
    result = np.zeros(bench_size)
    for i in range(bench_size):
        result[i] = time.time()
        func()
        result[i] = time.time() - result[i]
    return result


def show_timeit(result):
    mean_v = np.mean(result)
    stddev_v = np.std(result)
    min_v = np.min(result)
    max_v = np.max(result)
    print(
        f"mean: {mean_v:.4f}, stddev: {stddev_v:.4f}, min: {min_v:.4f}, max: {max_v:.4f}",
    )
    print()


def numpy_bench(N, bench_size):
    print("===============Python (NumPy / SciPy)==================")
    print()

    def uniform():
        return np.random.rand(N)

    print("-------------uniform random number sampling------------")
    show_timeit(timeit(uniform, bench_size))

    def normal():
        return np.random.randn(N)

    print("-------------normal random number sampling-------------")
    show_timeit(timeit(normal, bench_size))

    print("-------------stable random number sampling-------------")

    def stable():
        dist = levy_stable(alpha=0.7, beta=0.0, loc=0.0, scale=1.0)
        return dist.rvs(size=N)

    show_timeit(timeit(stable, bench_size))

    print("=======================================================")
    print()


def pyo3_bench(N, bench_size):
    print("===============Python (Rust by PyO3)===================")
    print()

    def uniform():
        return rand(N)

    print("-------------uniform random number sampling------------")
    show_timeit(timeit(uniform, bench_size))

    def normal():
        return randn(N)

    print("-------------normal random number sampling-------------")
    show_timeit(timeit(normal, bench_size))

    print("-------------stable random number sampling-------------")

    def stable():
        return stable_rand(0.7, 0.0, size=N)

    show_timeit(timeit(stable, bench_size))

    print("=======================================================")
    print()


def main():
    bench_size = 20 if len(sys.argv) <= 1 else int(sys.argv[1])
    N = 10_000_000

    print(f"bench size: {bench_size}, length of random vectors: {N}")
    print("unit: second")
    print()

    pyo3_bench(N, bench_size)
    numpy_bench(N, bench_size)


if __name__ == "__main__":
    main()
