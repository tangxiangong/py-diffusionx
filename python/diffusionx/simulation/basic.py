from abc import ABC, abstractmethod
from typing import Union, Optional
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

    @abstractmethod
    def fpt(
        self,
        domain: tuple[real, real],
        # step_size/time_step and max_duration vary per process, so not strictly enforced here
        # but generally expected. For CTRW/Poisson, step_size is not used.
        # For SubordinatedLangevin, base fpt does not use time_step.
        # We'll rely on wrappers in functional.py to handle this, and specific implementations.
        *args,  # To accommodate varying parameters like step_size, max_duration
        **kwargs,  # To accommodate varying parameters like step_size, max_duration
    ) -> Optional[float]:
        """Calculate the first passage time."""
        pass

    @abstractmethod
    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        # step_size/time_step varies per process
        *args,  # To accommodate step_size if present
        **kwargs,  # To accommodate step_size if present
    ) -> float:
        """Calculate the occupation time."""
        pass

    @abstractmethod
    def fpt_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        # step_size/time_step, max_duration vary
        *args,
        **kwargs,
    ) -> Optional[float]:
        """Calculate the raw moment of the first passage time."""
        pass

    @abstractmethod
    def fpt_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        # step_size/time_step, max_duration vary
        *args,
        **kwargs,
    ) -> Optional[float]:
        """Calculate the central moment of the first passage time."""
        pass

    @abstractmethod
    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        # step_size/time_step varies
        *args,
        **kwargs,
    ) -> float:
        """Calculate the raw moment of the occupation time."""
        pass

    @abstractmethod
    def occupation_time_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        # step_size/time_step varies
        *args,
        **kwargs,
    ) -> float:
        """Calculate the central moment of the occupation time."""
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
