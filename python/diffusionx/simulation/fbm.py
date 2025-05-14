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


class Fbm(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        hurst_exponent: real = 0.5,
    ):
        """
        Initialize a fractional Brownian motion object.

        Args:
            start_position (real, optional): The starting position. Defaults to 0.0.
            hurst_exponent (real, optional): The Hurst exponent. Must be in (0, 1). Defaults to 0.5.

        Raises:
            TypeError: If start_position or hurst_exponent are not numbers.
            ValueError: If hurst_exponent is not in the range (0, 1).
        """
        try:
            _start_position = ensure_float(start_position)
            _hurst_exponent = ensure_float(hurst_exponent)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (0 < _hurst_exponent < 1):
            raise ValueError(
                f"hurst_exponent must be in the range (0, 1), got {_hurst_exponent}"
            )

        self.start_position = _start_position
        self.hurst_exponent = _hurst_exponent

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the fractional Brownian motion.

        Args:
            duration (real): Total duration of the simulation.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If duration or step_size are not numbers.
            ValueError: If duration or step_size are not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and positions of the FBM.
        """
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

        return _core.fbm_simulate(
            self.start_position,
            self.hurst_exponent,
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
        Calculate the first passage time of the FBM.

        Args:
            domain (tuple[real, real]): Domain (a, b) for FPT. a must be less than b.
            step_size (real, optional): Step size. Defaults to 0.01.
            max_duration (real, optional): Maximum simulation duration for FPT. Defaults to 1000.

        Raises:
            TypeError: If domain elements, step_size, or max_duration are not numbers.
            ValueError: If domain is invalid (a >= b), or step_size/max_duration not positive.

        Returns:
            Optional[float]: The FPT, or None if max_duration reached first.
        """
        _a, _b = validate_domain(domain, process_name="Fbm FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.fbm_fpt(
            self.start_position,
            self.hurst_exponent,
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
        _a, _b = validate_domain(domain, process_name="Fbm FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.fbm_fpt_raw_moment(
            self.start_position,
            self.hurst_exponent,
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
        _a, _b = validate_domain(domain, process_name="Fbm FPT central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0

        return _core.fbm_fpt_central_moment(
            self.start_position,
            self.hurst_exponent,
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
        Calculate the raw moment of the FBM.

        Args:
            duration (real): Simulation duration.
            order (int): Moment order (non-negative integer).
            particles (int): Number of particles (positive integer).
            step_size (real, optional): Step size. Defaults to 0.01.

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values (e.g., negative order, non-positive particles/duration/step_size).

        Returns:
            float: The raw moment.
        """
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.fbm_raw_moment(
            self.start_position,
            self.hurst_exponent,
            _duration,
            _step_size,
            _order,
            _particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        """
        Calculate the central moment of the FBM.

        Args:
            duration (real): Simulation duration.
            order (int): Moment order (non-negative integer).
            particles (int): Number of particles (positive integer).
            step_size (real, optional): Step size. Defaults to 0.01.

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values.

        Returns:
            float: The central moment.
        """
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.fbm_central_moment(
            self.start_position,
            self.hurst_exponent,
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
        Calculate the occupation time of the FBM in a domain.

        Args:
            domain (tuple[real, real]): Domain (a, b). a must be less than b.
            duration (real): Total simulation duration.
            step_size (real, optional): Step size. Defaults to 0.01.

        Raises:
            TypeError: If domain elements, duration, or step_size are not numbers.
            ValueError: For invalid domain or non-positive duration/step_size.

        Returns:
            float: Occupation time.
        """
        _a, _b = validate_domain(domain, process_name="Fbm Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.fbm_occupation_time(
            self.start_position,
            self.hurst_exponent,
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
        _a, _b = validate_domain(domain, process_name="Fbm Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.fbm_occupation_time_raw_moment(
            self.start_position,
            self.hurst_exponent,
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
        _a, _b = validate_domain(domain, process_name="Fbm Occupation central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.fbm_occupation_time_central_moment(
            self.start_position,
            self.hurst_exponent,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        )

    def mean(self, duration: real, particles: int, step_size: real = 0.01) -> float:
        """
        Calculate the mean of the FBM.

        Args:
            duration (real): Simulation duration.
            particles (int): Number of particles.
            step_size (real, optional): Step size. Defaults to 0.01.

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values.

        Returns:
            float: The mean.
        """
        # Validation for particles, duration, step_size will be done in raw_moment
        # or can be duplicated here for immediate feedback if preferred.
        # For now, relying on raw_moment's validation.
        # Ensure parameters passed to raw_moment are validated if not already float/int
        _duration = ensure_float(duration)
        _step_size = ensure_float(step_size)
        # particles is already int, order is literal 1
        return self.raw_moment(_duration, 1, particles, _step_size)

    def msd(self, duration: real, particles: int, step_size: real = 0.01) -> float:
        """
        Calculate the mean square displacement (MSD) of the FBM.

        Args:
            duration (real): Simulation duration.
            particles (int): Number of particles.
            step_size (real, optional): Step size. Defaults to 0.01.

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values.

        Returns:
            float: The MSD.
        """
        # Similar to mean, relying on central_moment's validation for now.
        _duration = ensure_float(duration)
        _step_size = ensure_float(step_size)
        # particles is int, order is literal 2
        return self.central_moment(_duration, 2, particles, _step_size)
