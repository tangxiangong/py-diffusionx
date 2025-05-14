from diffusionx import _core
from typing import Union, Optional
from .basic import StochasticProcess, Trajectory
from .utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
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
        _a, _b = validate_domain(domain, process_name="Bm FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.bm_fpt(
            self.start_position,
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def fpt_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the raw moment of the first passage time for Brownian motion.

        Args:
            domain (tuple[real, real]): The domain (a, b). a must be less than b.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles for ensemble average (positive integer).
            step_size (real, optional): Step size. Defaults to 0.01.
            max_duration (real, optional): Maximum duration. Defaults to 1000.

        Returns:
            Optional[float]: The raw moment of FPT, or None if no passage for some particles.
        """
        _a, _b = validate_domain(domain, process_name="Bm FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        # _core.bm_fpt_raw_moment is expected to handle order 0 appropriately.
        # For FPT, 0th moment can relate to probability of passage if not certain.

        return _core.bm_fpt_raw_moment(
            self.start_position,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        )

    def fpt_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the central moment of the first passage time for Brownian motion.

        Args:
            domain (tuple[real, real]): The domain (a, b). a must be less than b.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles for ensemble average (positive integer).
            step_size (real, optional): Step size. Defaults to 0.01.
            max_duration (real, optional): Maximum duration. Defaults to 1000.

        Returns:
            Optional[float]: The central moment of FPT, or None if no passage for some particles.
        """
        _a, _b = validate_domain(domain, process_name="Bm FPT central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0
        # For order 1, the _core function is expected to return a value consistent with 0.0
        # or None if the mean FPT is not well-defined or calculable.

        return _core.bm_fpt_central_moment(
            self.start_position,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
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
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.bm_raw_moment(
            self.start_position,
            self.diffusion_coefficient,
            _duration,
            _step_size,
            _order,
            _particles,
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
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.bm_central_moment(
            self.start_position,
            self.diffusion_coefficient,
            _duration,
            _step_size,
            _order,
            _particles,
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
        _a, _b = validate_domain(domain, process_name="Bm Occupation time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.bm_occupation_time(
            self.start_position,
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        """
        Calculate the raw moment of the occupation time for Brownian motion.

        Args:
            domain (tuple[real, real]): The domain (a, b). a must be less than b.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles for ensemble average (positive integer).
            duration (real): Total duration of the simulation.
            step_size (real, optional): Step size. Defaults to 0.01.

        Returns:
            float: The raw moment of occupation time.
        """
        _a, _b = validate_domain(domain, process_name="Bm Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.bm_occupation_time_raw_moment(
            self.start_position,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        )

    def occupation_time_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        """
        Calculate the central moment of the occupation time for Brownian motion.

        Args:
            domain (tuple[real, real]): The domain (a, b). a must be less than b.
            order (int): Order of the moment (non-negative integer).
            particles (int): Number of particles for ensemble average (positive integer).
            duration (real): Total duration of the simulation.
            step_size (real, optional): Step size. Defaults to 0.01.

        Returns:
            float: The central moment of occupation time.
        """
        _a, _b = validate_domain(domain, process_name="Bm Occupation central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        if _order == 1:
            # The first central moment is E[X - E[X]], which is 0 by definition.
            return 0.0

        return _core.bm_occupation_time_central_moment(
            self.start_position,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        )
