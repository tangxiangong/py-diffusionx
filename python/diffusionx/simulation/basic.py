from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from .utils import ensure_float


real = Union[int, float]


class StochasticProcess(ABC):
    @abstractmethod
    def simulate(
        self, duration: real, step_size: real
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the stochastic process.

        Args:
            duration (real): The total duration of the simulation.
            step_size (real): The time step for the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing arrays for time points and process values.
        """
        pass


class Trajectory:
    def __init__(self, sp: StochasticProcess, duration: real):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )

        try:
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(
                f"duration must be a number convertible to float. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")

        self.sp = sp
        self.duration = _duration

    def simulate(self, step_size: real = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a trajectory of the stochastic process.

        Args:
            step_size (real, optional): The time step for the simulation. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing arrays for time points and process values.
        """
        try:
            _step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(
                f"step_size must be a number convertible to float. Error: {e}"
            ) from e
        if _step_size <= 0:
            raise ValueError("step_size must be positive")

        return self.sp.simulate(self.duration, _step_size)
