from math import sqrt
from typing import Callable

import numpy as np
from numba import (  # pyright: ignore [reportMissingTypeStubs]
    njit,  # pyright: ignore [reportUnknownVariableType]
    prange,
)


class Langevin:
    def __init__(
        self,
        f: Callable[[float, float], float],
        g: Callable[[float, float], float],
        starting_point: float = 0.0,
    ):
        self.starting_point: float = starting_point
        self.drift: Callable[[float, float], float] = f
        self.diffusivity: Callable[[float, float], float] = g

    def simulate(self, duration: float, tau: float = 0.01):
        return _simulate(
            self.drift,
            self.diffusivity,
            duration,
            self.starting_point,
            tau,
        )

    def msd(
        self,
        duration: float,
        N: int = 10_000,
        tau: float = 0.01,
    ):
        return _msd(
            self.drift,
            self.diffusivity,
            duration,
            self.starting_point,
            N,
            tau,
        )


@njit
def _simulate(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    duration: float,
    starting_point: float = 0.0,
    tau: float = 0.01,
):
    t = np.arange(0, duration, tau)
    n = len(t) - 1
    noise = np.random.normal(0, 1, n)
    x = np.zeros((n + 1,))
    x[0] = starting_point
    for i in range(1, n + 1):
        drift_term = f(t[i - 1], x[i - 1]) * tau  # pyright: ignore [reportAny]
        diffusion_term = g(t[i - 1], x[i - 1]) * noise[i - 1] * sqrt(tau)  # pyright: ignore [reportAny]
        x[i] = x[i - 1] + drift_term + diffusion_term
    x = np.cumsum(noise)
    return t, x


@njit(parallel=True)  # pyright: ignore [reportUntypedFunctionDecorator]
def _msd(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    duration: float,
    starting_point: float = 0.0,
    N: int = 10_000,
    tau: float = 0.01,
):
    total = 0.0
    for _k in prange(N):
        _, x = _simulate(f, g, duration, starting_point, tau)
        total += (x[-1] - x[0]) ** 2  # pyright: ignore [reportAny]
    return total / N


if __name__ == "__main__":
    import time

    @njit
    def f(_t: float, x: float) -> float:
        return -x

    @njit
    def g(_t: float, _x: float) -> float:
        return 1.0

    duration = [i for i in range(100, 1100, 100)]
    eq = Langevin(f, g)
    _ = eq.msd(1.0)
    msds = [0.0 for _ in duration]
    start = time.time()
    for i, d in enumerate(duration):
        msds[i] = eq.msd(d)
    elapsed_parallel = time.time() - start
    print(f" Time elapsed: {elapsed_parallel:.3f} seconds")
