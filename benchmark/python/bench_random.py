import numpy as np
from diffusionx.random import randn, stable_rand, randexp
from diffusionx.random import uniform as rand
from scipy.stats import levy_stable
from utils import bench


def numpy_bench(N, bench_size):
    print("===============Python (NumPy / SciPy)==================")
    print()

    def uniform():
        return np.random.rand(N)

    bench("uniform random number sampling", uniform, bench_size)

    def normal():
        return np.random.standard_normal(N)

    bench("normal random number sampling", normal, bench_size)

    def exponential():
        return np.random.standard_exponential(N)

    bench("exponential random number sampling", exponential, bench_size)

    def stable():
        dist = levy_stable(alpha=0.7, beta=0.0, loc=0.0, scale=1.0)
        return dist.rvs(size=N)

    bench("stable random number sampling", stable, bench_size)


def pyo3_bench(N, bench_size):
    print("===============Python (Rust by PyO3)===================")
    print()

    def uniform():
        return rand(N)

    bench("uniform random number sampling", uniform, bench_size)

    def normal():
        return randn(N)

    bench("normal random number sampling", normal, bench_size)

    def exponential():
        return randexp(N)

    bench("exponential random number sampling", exponential, bench_size)

    def stable():
        return stable_rand(0.7, 0.0, size=N)

    bench("stable random number sampling", stable, bench_size)


def main():
    bench_size = 20
    N = 10_000_000

    print(f"bench size: {bench_size}, length of random vectors: {N}")

    pyo3_bench(N, bench_size)
    numpy_bench(N, bench_size)


if __name__ == "__main__":
    main()
