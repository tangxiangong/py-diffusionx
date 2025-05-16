from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


real = Union[int, float]


class StochasticProcess(ABC):
    pass


class ContinuousProcess(StochasticProcess):
    @abstractmethod
    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the continuous stochastic process.

        Args:
            duration (real): The total duration of the simulation.
            step_size (float): The time step for the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing arrays for time points and process values.
        """
        pass

    @abstractmethod
    def moment(
        self,
        duration: real,
        order: int,
        step_size: float = 0.01,
        central: bool = True,
        particles: int = 10_000,
    ) -> float:
        pass


class PointProcess(StochasticProcess):
    @abstractmethod
    def simulate(
        self,
        duration: Optional[real] = None,
        num_step: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the point process.

        Args:
            duration (real): The total duration of the simulation.
            step_size (float): The time step for the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing arrays for time points and process values.
        """
        pass

    @abstractmethod
    def moment(
        self, duration: real, order: int, central: bool = True, particles: int = 10_000
    ) -> float:
        pass
