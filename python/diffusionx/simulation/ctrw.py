from diffusionx import _core
from typing import Union, Optional
from .basic import StochasticProcess, Trajectory
from .utils import ensure_float
import numpy as np


real = Union[float, int]


class CTRW(StochasticProcess):
    def __init__(
        self,
        alpha: real = 1.0,
        beta: real = 2.0,
        start_position: real = 0.0,
    ):
        """
        Continuous Time Random Walk.

        Args:
            alpha (real, optional): Waiting time exponent (0, 1]. Defaults to 1.0.
            beta (real, optional): Jump length exponent (0, 2]. Defaults to 2.0.
            start_position (real, optional): Starting position. Defaults to 0.0.

        Raises:
            TypeError: If alpha, beta, or start_position are not numbers.
            ValueError: If alpha is not in (0, 1] or beta is not in (0, 2].
        """
        try:
            _alpha = ensure_float(alpha)
            _beta = ensure_float(beta)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (0 < _alpha <= 1):
            raise ValueError(f"alpha must be in the range (0, 1], got {_alpha}")
        if not (0 < _beta <= 2):
            raise ValueError(f"beta must be in the range (0, 2], got {_beta}")

        self.alpha = _alpha
        self.beta = _beta
        self.start_position = _start_position

    def __call__(self, duration: real) -> Trajectory:
        # Duration validation is handled by Trajectory.__init__
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the CTRW based on total duration.

        Note: `step_size` is not used by this simulation method but is kept for
              consistency with the StochasticProcess interface.

        Args:
            duration (real): Total duration of the simulation.
            step_size (real, optional): Ignored by this method. Defaults to 0.01.

        Raises:
            TypeError: If duration is not a number.
            ValueError: If duration is not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and positions of the CTRW.
        """
        try:
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e

        if _duration <= 0:
            raise ValueError(f"duration must be positive, got {_duration}")
        # step_size is intentionally not validated here as it's documented as unused.
        return _core.ctrw_simulate_duration(
            self.alpha,
            self.beta,
            self.start_position,
            _duration,
        )

    def simulate_with_step(self, num_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the CTRW for a specified number of steps.

        Args:
            num_step (int): The number of steps in the simulation.

        Raises:
            TypeError: If num_step is not an integer.
            ValueError: If num_step is not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and positions of the CTRW.
        """
        if not isinstance(num_step, int):
            raise TypeError(
                f"num_step must be an integer, got {type(num_step).__name__}"
            )
        if num_step <= 0:
            raise ValueError(f"num_step must be positive, got {num_step}")
        return _core.ctrw_simulate_step(
            self.alpha,
            self.beta,
            self.start_position,
            num_step,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time of the CTRW.

        Args:
            domain (tuple[real, real]): Domain (a, b) for FPT. a must be less than b.
            max_duration (real, optional): Maximum duration to simulate for FPT. Defaults to 1000.

        Raises:
            TypeError: If domain elements or max_duration are not numbers.
            ValueError: If domain is invalid (a >= b) or max_duration not positive.

        Returns:
            Optional[float]: The FPT, or None if max_duration reached first.
        """
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
            _max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(
                f"Domain elements and max_duration must be numbers. Error: {e}"
            ) from e

        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )
        if _max_duration <= 0:
            raise ValueError("max_duration must be positive")

        return _core.ctrw_fpt(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            _max_duration,
        )

    def raw_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the raw moment of the CTRW.

        Args:
            duration (real): Simulation duration.
            order (int): Moment order (non-negative integer).
            particles (int): Number of particles (positive integer).

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values.

        Returns:
            float: The raw moment.
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
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if _duration <= 0:
            raise ValueError("duration must be positive")

        return _core.ctrw_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            _duration,
            order,
            particles,
        )

    def central_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the central moment of the CTRW.

        Args:
            duration (real): Simulation duration.
            order (int): Moment order (non-negative integer).
            particles (int): Number of particles (positive integer).

        Raises:
            TypeError: If any parameter has an incorrect type.
            ValueError: For invalid parameter values.

        Returns:
            float: The central moment.
        """
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if order == 0:
            return 1.0
        if (
            order == 1
        ):  # First central moment is 0 (assuming mean is start_position or handled by _core)
            # For a general CTRW, the mean might not be start_position.
            # If E[X_t] != start_position, then first central moment is E[X_t - E[X_t]] = 0.
            # If _core.ctrw_central_moment calculates E[(X_t - start_position)^1], it might not be 0 if mean drifts.
            # However, typically, central moments are about the mean of the process at time t.
            # For now, assume it calculates E[(X_t - E[X_t])^order], so for order 1, it's 0.
            return 0.0

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if _duration <= 0:
            raise ValueError("duration must be positive")

        return _core.ctrw_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            _duration,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ) -> float:
        """
        Calculate the occupation time of the CTRW in a domain.

        Args:
            domain (tuple[real, real]): Domain (a, b). a must be less than b.
            duration (real): Total simulation duration.

        Raises:
            TypeError: If domain elements or duration are not numbers.
            ValueError: For invalid domain or non-positive duration.

        Returns:
            float: Occupation time.
        """
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            _duration = ensure_float(duration)
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements and duration must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )

        return _core.ctrw_occupation_time(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            _duration,
        )

    def mean(self, duration: real, particles: int) -> float:
        """
        Calculate the mean of the CTRW.

        Args:
            duration (real): Simulation duration.
            particles (int): Number of particles.

        Raises:
            TypeError: If duration or particles have incorrect types.
            ValueError: For invalid parameter values.

        Returns:
            float: The mean.
        """
        _duration = ensure_float(duration)  # Validate here before passing
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        # Further validation for _duration and particles (e.g. >0) is handled by raw_moment
        return self.raw_moment(_duration, 1, particles)

    def msd(self, duration: real, particles: int) -> float:
        """
        Calculate the mean square displacement (MSD) of the CTRW.

        Args:
            duration (real): Simulation duration.
            particles (int): Number of particles.

        Raises:
            TypeError: If duration or particles have incorrect types.
            ValueError: For invalid parameter values.

        Returns:
            float: The MSD.
        """
        _duration = ensure_float(duration)  # Validate here
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        # Further validation for _duration and particles is handled by central_moment
        return self.central_moment(_duration, 2, particles)
