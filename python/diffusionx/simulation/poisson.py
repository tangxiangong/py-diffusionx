from diffusionx import _core
from typing import Union, Optional
from .basic import StochasticProcess, Trajectory
from .utils import ensure_float
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
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

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
            raise ValueError(f"duration must be positive, got {_duration}")

        if order == 0:
            return 1.0

        return _core.poisson_raw_moment(
            self.lambda_,
            _duration,
            order,
            particles,
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
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

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
            raise ValueError(f"duration must be positive, got {_duration}")

        if order == 0:
            return 1.0
        if order == 1:
            return 0.0

        return _core.poisson_central_moment(
            self.lambda_,
            _duration,
            order,
            particles,
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

        if not (
            isinstance(a, (int, float))
            and float(a).is_integer()
            and isinstance(b, (int, float))
            and float(b).is_integer()
        ):
            pass
        if a < 0 or b < 0:
            raise ValueError("Domain counts for FPT must be non-negative.")
        if b <= a:
            raise ValueError(
                f"Target count domain[1] ({b}) must be greater than start count domain[0] ({a})."
            )

        if _max_duration <= 0:
            raise ValueError("max_duration must be positive")

        return _core.poisson_fpt(
            self.lambda_,
            (a, b),
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
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(
                f"Domain elements and duration must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError(f"duration must be positive, got {_duration}")
        if a < 0 or b < 0:
            raise ValueError("Domain counts for occupation_time must be non-negative.")
        if a > b:
            raise ValueError(
                f"Invalid domain count range [{a}, {b}]; domain[0] must be less than or equal to domain[1]."
            )

        return _core.poisson_occupation_time(
            self.lambda_,
            (a, b),
            _duration,
        )

    def fpt_raw_moment(
        self,
        domain: tuple[real, real],  # (start_count, target_count)
        order: int,
        particles: int,
        max_duration: real = 1000,
    ) -> Optional[float]:
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            a = ensure_float(domain[0])  # start_count
            b = ensure_float(domain[1])  # target_count
            _max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(
                f"Domain counts and max_duration must be numbers. Error: {e}"
            ) from e

        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(f"Domain counts for FPT must be integers, got {domain}")
        a = int(a)
        b = int(b)
        if a < 0 or b < 0:
            raise ValueError(
                f"Domain counts for FPT must be non-negative, got {(a, b)}"
            )
        if b <= a:
            raise ValueError(
                f"Target count domain[1] ({b}) must be greater than start count domain[0] ({a})."
            )
        if _max_duration <= 0:
            raise ValueError("max_duration must be positive")

        return _core.poisson_fpt_raw_moment(
            self.lambda_,
            (float(a), float(b)),  # _core expects float tuple
            order,
            particles,
            _max_duration,
        )

    def fpt_central_moment(
        self,
        domain: tuple[real, real],  # (start_count, target_count)
        order: int,
        particles: int,
        max_duration: real = 1000,
    ) -> Optional[float]:
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            a = ensure_float(domain[0])  # start_count
            b = ensure_float(domain[1])  # target_count
            _max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(
                f"Domain counts and max_duration must be numbers. Error: {e}"
            ) from e

        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(f"Domain counts for FPT must be integers, got {domain}")
        a = int(a)
        b = int(b)
        if a < 0 or b < 0:
            raise ValueError(
                f"Domain counts for FPT must be non-negative, got {(a, b)}"
            )
        if b <= a:
            raise ValueError(
                f"Target count domain[1] ({b}) must be greater than start count domain[0] ({a})."
            )
        if _max_duration <= 0:
            raise ValueError("max_duration must be positive")

        if order == 0:
            return 1.0

        return _core.poisson_fpt_central_moment(
            self.lambda_,
            (float(a), float(b)),
            order,
            particles,
            _max_duration,
        )

    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],  # (min_count, max_count)
        order: int,
        particles: int,
        duration: real,
    ) -> float:
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            a = ensure_float(domain[0])  # min_count
            b = ensure_float(domain[1])  # max_count
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(
                f"Domain counts and duration must be numbers. Error: {e}"
            ) from e

        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(
                f"Domain counts for occupation time must be integers, got {domain}"
            )
        a = int(a)
        b = int(b)
        if a < 0 or b < 0:
            raise ValueError(
                f"Domain counts for occupation time must be non-negative, got {(a, b)}"
            )
        if b <= a:
            raise ValueError(
                f"max_count domain[1] ({b}) must be greater than min_count domain[0] ({a}) for occupation time."
            )  # Or b < a if point occupation is allowed for a single count N(t)=k
        if _duration <= 0:
            raise ValueError("duration must be positive")

        if order == 0:
            return 1.0

        return _core.poisson_occupation_time_raw_moment(
            self.lambda_,
            (float(a), float(b)),
            order,
            particles,
            _duration,
        )

    def occupation_time_central_moment(
        self,
        domain: tuple[real, real],  # (min_count, max_count)
        order: int,
        particles: int,
        duration: real,
    ) -> float:
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            a = ensure_float(domain[0])  # min_count
            b = ensure_float(domain[1])  # max_count
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(
                f"Domain counts and duration must be numbers. Error: {e}"
            ) from e

        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(
                f"Domain counts for occupation time must be integers, got {domain}"
            )
        a = int(a)
        b = int(b)
        if a < 0 or b < 0:
            raise ValueError(
                f"Domain counts for occupation time must be non-negative, got {(a, b)}"
            )
        if b <= a:
            raise ValueError(
                f"max_count domain[1] ({b}) must be greater than min_count domain[0] ({a}) for occupation time."
            )
        if _duration <= 0:
            raise ValueError("duration must be positive")

        if order == 0:
            return 1.0
        if order == 1:
            return 0.0  # First central moment is 0

        return _core.poisson_occupation_time_central_moment(
            self.lambda_,
            (float(a), float(b)),
            order,
            particles,
            _duration,
        )
