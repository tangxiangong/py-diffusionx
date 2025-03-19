from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class Fbm(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        hurst_exponent: real = 0.5,
    ):
        """
        Initialize a fractional Brownian motion object.

        Args:
            start_position (real, optional): The starting position of the fractional Brownian motion. Defaults to 0.0.
            hurst_exponent (real, optional): The Hurst exponent of the fractional Brownian motion. Defaults to 0.5.

        Raises:
            ValueError: If the Hurst exponent is not in the range (0, 1).
            ValueError: If the value is not a number.

        Returns:
            Fbm: The fractional Brownian motion object.
        """
        start_position = check_transform(start_position)
        hurst_exponent = check_transform(hurst_exponent)
        if hurst_exponent <= 0 or hurst_exponent >= 1:
            raise ValueError("hurst_exponent must be in the range (0, 1)")

        self.start_position = start_position
        self.hurst_exponent = hurst_exponent

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the fractional Brownian motion.

        Args:
            duration (real): The duration of the simulation.
            step_size (real, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the time and position of the fractional Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        return _core.fbm_simulate(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ):
        """
        Calculate the first passage time of the fractional Brownian motion.

        Args:
            domain (tuple[real, real]): The domain of the fractional Brownian motion.
            step_size (real, optional): The step size of the fractional Brownian motion. Defaults to 0.01.
            max_duration (real, optional): The maximum duration. Defaults to 1000.

        Returns:
            real: The first passage time of the fractional Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")
        return _core.fbm_fpt(
            self.start_position,
            self.hurst_exponent,
            step_size,
            (a, b),
            max_duration,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        """
        Calculate the raw moment of the fractional Brownian motion.

        Args:
            duration (real): The duration of the simulation.
            order (int): The order of the moment.
            particles (int): The number of particles.
            step_size (real, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Returns:
            real: The raw moment of the fractional Brownian motion.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.fbm_raw_moment(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
            order,
            particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ):
        """
        Calculate the central moment of the fractional Brownian motion.

        Args:
            duration (real): The duration of the simulation.
            order (int): The order of the moment.
            particles (int): The number of particles.
            step_size (float, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Raises:
            ValueError: If the order is not an integer.
            ValueError: If the order is negative.
            ValueError: If the order is zero.
            ValueError: If the particles number is not an integer.
            ValueError: If the particles number is not positive.
            ValueError: If the step size is not positive.

        Returns:
            real: The central moment of the fractional Brownian motion.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.fbm_central_moment(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        """
        Calculate the occupation time of the fractional Brownian motion.

        Args:
            domain (tuple[real, real]): The domain of the fractional Brownian motion.
            duration (real): The duration of the simulation.
            step_size (real, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Returns:
            real: The occupation time of the fractional Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        return _core.fbm_occupation_time(
            self.start_position,
            self.hurst_exponent,
            step_size,
            (a, b),
            duration,
        )

    def mean(self, duration: real, particles: int, step_size: float = 0.01) -> float:
        """
        Calculate the mean of the fractional Brownian motion.

        Args:
            duration (real): The duration of the simulation.
            particles (int): The number of particles.
            step_size (float, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Returns:
            float: The mean of the fractional Brownian motion.
        """
        return self.raw_moment(duration, 1, particles, step_size)

    def msd(self, duration: real, particles: int, step_size: float = 0.01) -> float:
        """
        Calculate the mean square displacement of the fractional Brownian motion.

        Args:
            duration (real): The duration of the simulation.
            particles (int): The number of particles.
            step_size (float, optional): The step size of the fractional Brownian motion. Defaults to 0.01.

        Returns:
            float: The mean square displacement of the fractional Brownian motion.
        """
        return self.central_moment(duration, 2, particles, step_size)
