from diffusionx import _core
from typing import Union, Optional
from .basic import StochasticProcess, Trajectory
from .utils import ensure_float
import numpy as np


real = Union[float, int]


class Bm(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        diffusion_coefficient: real = 1.0,
    ):
        """
        Initialize a Brownian motion object.

        Args:
            start_position (real, optional): Starting position of the Brownian motion. Defaults to 0.0.
            diffusion_coefficient (real, optional): Diffusion coefficient of the Brownian motion. Defaults to 1.0.

        Raises:
            TypeError: If start_position or diffusion_coefficient are not numbers.
            ValueError: If diffusion_coefficient is not positive.
        """
        try:
            _start_position = ensure_float(start_position)
            _diffusion_coefficient = ensure_float(diffusion_coefficient)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        self.start_position = _start_position
        self.diffusion_coefficient = _diffusion_coefficient

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Brownian motion.

        Args:
            duration (real): Total duration of the simulation.
            step_size (real, optional): Step size of the Brownian motion. Defaults to 0.01.

        Raises:
            TypeError: If duration or step_size are not numbers.
            ValueError: If duration or step_size are not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Brownian motion.
        """
        try:
            _duration = ensure_float(duration)
            _step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(
                f"duration and step_size must be numbers. Error: {e}"
            ) from e

        if _step_size <= 0:
            raise ValueError("step_size must be positive")
        if _duration <= 0:
            raise ValueError("duration must be positive")

        return _core.bm_simulate(
            self.start_position,
            self.diffusion_coefficient,
            _duration,
            _step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time of the Brownian motion.

        Args:
            domain (tuple[real, real]): The domain (a, b) for FPT. a must be less than b.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.
            max_duration (real, optional): Maximum duration to simulate for FPT. Defaults to 1000.

        Raises:
            TypeError: If domain elements, step_size, or max_duration are not numbers.
            ValueError: If domain is not a valid interval (a >= b), or if step_size or max_duration are not positive.

        Returns:
            Optional[float]: The first passage time, or None if max_duration is reached before FPT.
        """
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            _step_size = ensure_float(step_size)
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
            _max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(
                f"Domain elements, step_size, and max_duration must be numbers. Error: {e}"
            ) from e

        if _step_size <= 0:
            raise ValueError("step_size must be positive")
        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )
        if _max_duration <= 0:
            raise ValueError("max_duration must be positive")

        return _core.bm_fpt(
            self.start_position,
            self.diffusion_coefficient,
            _step_size,
            (a, b),
            _max_duration,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        """
        Calculate the raw moment of the Brownian motion.

        Args:
            duration (real): Duration of the simulation for moment calculation.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles (positive integer) for ensemble averaging.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If duration, order, particles, or step_size have incorrect types.
            ValueError: If order is negative, particles is not positive, or if duration or step_size are not positive.

        Returns:
            float: The raw moment of the Brownian motion.
        """
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if order == 0:
            return 1.0

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(
                f"duration and step_size must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _step_size <= 0:
            raise ValueError("step_size must be positive")

        return _core.bm_raw_moment(
            self.start_position,
            self.diffusion_coefficient,
            _duration,
            _step_size,
            order,
            particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        """
        Calculate the central moment of the Brownian motion.

        Args:
            duration (real): Duration of the simulation for moment calculation.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles (positive integer) for ensemble averaging.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If duration, order, particles, or step_size have incorrect types.
            ValueError: If order is negative, particles is not positive, or if duration or step_size are not positive.

        Returns:
            float: The central moment of the Brownian motion.
        """
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if order == 0:
            return 1.0
        if order == 1:
            return 0.0

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(
                f"duration and step_size must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _step_size <= 0:
            raise ValueError("step_size must be positive")

        return _core.bm_central_moment(
            self.start_position,
            self.diffusion_coefficient,
            _duration,
            _step_size,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        """
        Calculate the occupation time of the Brownian motion in a given domain.

        Args:
            domain (tuple[real, real]): The domain (a, b) for occupation time. a must be less than b.
            duration (real): The total duration of the simulation.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If domain elements, duration, or step_size are not numbers.
            ValueError: If domain is not a valid interval (a >= b), or if duration or step_size are not positive.

        Returns:
            float: The occupation time of the Brownian motion in the domain.
        """
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            _duration = ensure_float(duration)
            _step_size = ensure_float(step_size)
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements, duration, and step_size must be numbers. Error: {e}"
            ) from e

        if _step_size <= 0:
            raise ValueError("step_size must be positive")
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )

        return _core.bm_occupation_time(
            self.start_position,
            self.diffusion_coefficient,
            _step_size,
            (a, b),
            _duration,
        )
