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


class Poisson(StochasticProcess):
    def __init__(
        self,
        lambda_: real = 1.0,
    ):
        """
        Initialize a Poisson process object.

        Args:
            lambda_ (real, optional): Rate parameter (lambda > 0) of the Poisson process. Defaults to 1.0.

        Raises:
            TypeError: If lambda_ is not a number.
            ValueError: If lambda_ is not positive.
        """
        try:
            _lambda_ = ensure_float(lambda_)
        except TypeError as e:
            raise TypeError(f"lambda_ must be a number. Error: {e}") from e

        if _lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {_lambda_}")
        self.lambda_ = _lambda_

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Poisson process based on total duration.
        Note: `step_size` is not used by this simulation method (_core.poisson_simulate_duration
              likely uses event-based simulation up to `duration`) but is kept for
              consistency with the StochasticProcess interface.

        Args:
            duration (real): Total duration of the simulation (must be positive).
            step_size (real, optional): Ignored. Defaults to 0.01.

        Raises:
            TypeError: If duration is not a number.
            ValueError: If duration is not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and event counts of the Poisson process.
        """
        try:
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e

        if _duration <= 0:
            raise ValueError(f"duration must be positive, got {_duration}")
        return _core.poisson_simulate_duration(
            self.lambda_,
            _duration,
        )

    def simulate_step(self, num_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Poisson process for a specified number of events (steps).

        Args:
            num_step (int): Number of events (steps) in the simulation (must be positive).

        Raises:
            TypeError: If num_step is not an integer.
            ValueError: If num_step is not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and event counts of the Poisson process.
        """
        if not isinstance(num_step, int):
            raise TypeError(
                f"num_step must be an integer, got {type(num_step).__name__}"
            )
        if num_step <= 0:
            raise ValueError(f"num_step must be positive, got {num_step}")
        return _core.poisson_simulate_step(
            self.lambda_,
            num_step,
        )

    def raw_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the raw moment of the Poisson process at a given duration.

        Args:
            duration (real): Duration of the process (must be positive).
            order (int): Order of the moment (must be non-negative).
            particles (int): Number of particles for ensemble average (must be positive).

        Raises:
            TypeError: If parameters have incorrect types.
            ValueError: If parameters have invalid values.

        Returns:
            float: The raw moment of the Poisson process.
        """
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")

        if _order == 0:
            return 1.0

        return _core.poisson_raw_moment(
            self.lambda_,
            _duration,
            _order,
            _particles,
        )

    def central_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the central moment of the Poisson process at a given duration.

        Args:
            duration (real): Duration of the process (must be positive).
            order (int): Order of the moment (must be non-negative).
            particles (int): Number of particles for ensemble average (must be positive).

        Raises:
            TypeError: If parameters have incorrect types.
            ValueError: If parameters have invalid values.

        Returns:
            float: The central moment of the Poisson process.
        """
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.poisson_central_moment(
            self.lambda_,
            _duration,
            _order,
            _particles,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time for the Poisson process to reach a certain count/level.
        The 'domain' here usually refers to a target count. Assuming domain[0] is start, domain[1] is target count.

        Args:
            domain (tuple[real, real]): Domain (start_count, target_count). target_count must be > start_count.
                                         Typically start_count is 0 for FPT to N events.
            max_duration (real, optional): Maximum physical time to wait. Defaults to 1000.

        Raises:
            TypeError: If domain elements or max_duration are not numbers.
            ValueError: If domain is invalid or max_duration not positive.

        Returns:
            Optional[float]: The first passage time (physical time), or None if max_duration is reached.
        """
        _a, _b = validate_domain(
            domain, domain_type="poisson_fpt", process_name="Poisson FPT"
        )
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.poisson_fpt(
            self.lambda_,
            (_a, _b),
            _max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ) -> float:
        """
        Calculate the occupation time of the Poisson process (i.e., time spent where N(t) is within a given count range).

        Args:
            domain (tuple[real, real]): The count range [a,b]. a must be less than or equal to b.
            duration (real): The total physical time duration of the observation (must be positive).

        Raises:
            TypeError: If domain elements or duration are not numbers.
            ValueError: For invalid domain or non-positive duration.

        Returns:
            float: The total time N(t) spent in the count range [a,b].
        """
        _a, _b = validate_domain(
            domain,
            domain_type="poisson_occupation",
            process_name="Poisson Occupation Time",
        )
        _duration = validate_positive_float_param(duration, "duration")

        return _core.poisson_occupation_time(
            self.lambda_,
            (_a, _b),
            _duration,
        )

    def fpt_raw_moment(
        self,
        domain: tuple[real, real],  # (start_count, target_count)
        order: int,
        particles: int,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(
            domain, domain_type="poisson_fpt", process_name="Poisson FPT raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.poisson_fpt_raw_moment(
            self.lambda_,
            (_a, _b),  # _core expects float tuple
            _order,
            _particles,
            _max_duration,
        )

    def fpt_central_moment(
        self,
        domain: tuple[real, real],  # (start_count, target_count)
        order: int,
        particles: int,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(
            domain, domain_type="poisson_fpt", process_name="Poisson FPT central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0

        return _core.poisson_fpt_central_moment(
            self.lambda_,
            (_a, _b),
            _order,
            _particles,
            _max_duration,
        )

    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],  # (min_count, max_count)
        order: int,
        particles: int,
        duration: real,
    ) -> float:
        _a, _b = validate_domain(
            domain,
            domain_type="poisson_occupation",
            process_name="Poisson Occupation raw moment",
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")

        if _order == 0:
            return 1.0

        return _core.poisson_occupation_time_raw_moment(
            self.lambda_,
            (_a, _b),
            _order,
            _particles,
            _duration,
        )

    def occupation_time_central_moment(
        self,
        domain: tuple[real, real],  # (min_count, max_count)
        order: int,
        particles: int,
        duration: real,
    ) -> float:
        _a, _b = validate_domain(
            domain,
            domain_type="poisson_occupation",
            process_name="Poisson Occupation central moment",
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.poisson_occupation_time_central_moment(
            self.lambda_,
            (_a, _b),
            _order,
            _particles,
            _duration,
        )
