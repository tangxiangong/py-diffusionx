from math import sqrt

import numpy as np
from numba import (  # pyright: ignore [reportMissingTypeStubs]
    njit,  # pyright: ignore [reportUnknownVariableType]
    prange,
)


class Bm:
    def __init__(self, starting_point: float = 0.0, diffusivity: float = 0.5):
        self.starting_point: float = starting_point
        self.diffusivity: float = diffusivity

    def simulate(self, duration: float, tau: float = 0.01):
        t = np.arange(0, duration, tau)
        n = len(t) - 1
        sigma = sqrt(2 * self.diffusivity * tau)
        noise = np.random.normal(0, 1, n) * sigma
        noise = np.insert(noise, 0, self.starting_point)
        x = np.cumsum(noise)
        return t, x

    def msd(
        self,
        duration: float,
        N: int = 10_000,
        tau: float = 0.01,
    ):
        return _msd(
            duration,
            self.diffusivity,
            N,
            tau,
        )


@njit(parallel=True)  # pyright: ignore [reportUntypedFunctionDecorator]
def _msd(
    duration: float,
    diffusivity: float = 0.5,
    N: int = 10_000,
    tau: float = 0.01,
):
    total = 0.0
    for _ in prange(N):
        t = np.arange(0, duration, tau)
        n = len(t) - 1
        sigma = sqrt(2 * diffusivity * tau)
        noise = np.random.normal(0, 1, n) * sigma
        x = np.cumsum(noise)
        total += x[-1] ** 2  # pyright: ignore [reportAny]
    return total / N


if __name__ == "__main__":
    import time

    duration = [i for i in range(100, 1100, 100)]
    bm = Bm()
    _ = bm.msd(1.0)
    msds = [0.0 for _ in duration]
    start = time.time()
    for i, d in enumerate(duration):
        msds[i] = bm.msd(d)
    elapsed_parallel = time.time() - start
    print(f" Time elapsed: {elapsed_parallel:.3f} seconds")
