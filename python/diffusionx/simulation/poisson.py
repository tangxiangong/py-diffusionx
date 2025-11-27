from diffusionx import _core
from .basic import real, Vector
from .utils import (
    ensure_float,
    validate_bool,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float,
    validate_positive_integer,
)


class Poisson:
    def __init__(
        self,
        lambda_: real = 1.0,
    ):
        """
        Initialize a Poisson process object.

        Args:
            lambda_ (real, optional): Rate parameter (lambda > 0) of the Poisson process. Defaults to 1.0.

        """
        try:
            lambda_ = ensure_float(lambda_)
        except TypeError as e:
            raise TypeError(f"lambda_ must be a number. Error: {e}") from e

        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")
        self.lambda_ = lambda_

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        """
        Simulate the Poisson process based on total duration.
        Note: `time_step` is not used by this simulation method (_core.poisson_simulate_duration
              likely uses event-based simulation up to `duration`) but is kept for
              consistency with the StochasticProcess interface.

        Args:
            duration (real): Total duration of the simulation (must be positive).
            time_step (real, optional): Ignored. Defaults to 0.01.

        Returns:
            tuple[Vector, Vector]: Times and event counts of the Poisson process.
        """
        try:
            duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e

        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        return _core.poisson_simulate_duration(
            self.lambda_,
            duration,
        )

    def simulate_step(self, num_step: int) -> tuple[Vector, Vector]:
        """
        Simulate the Poisson process for a specified number of events (steps).

        Args:
            num_step (int): Number of events (steps) in the simulation (must be positive).

        Returns:
            tuple[Vector, Vector]: Times and event counts of the Poisson process.
        """
        num_step = validate_positive_integer(num_step, "num_step")
        return _core.poisson_simulate_step(
            self.lambda_,
            num_step,
        )

    def moment(
        self, duration: real, order: int, center: bool = False, particles: int = 10_000
    ) -> float:
        """
        Calculate the raw moment of the Poisson process at a given duration.

        Args:
            duration (real): Duration of the process (must be positive).
            order (int): Order of the moment (must be non-negative).
            particles (int): Number of particles for ensemble average (must be positive).

        Returns:
            float: The raw moment of the Poisson process.
        """
        validate_bool(center, "center")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")

        if order == 0:
            return 1.0

        result = (
            _core.poisson_raw_moment(
                self.lambda_,
                duration,
                order,
                particles,
            )
            if not center
            else _core.poisson_central_moment(
                self.lambda_,
                duration,
                order,
                particles,
            )
        )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ) -> float | None:
        """
        Calculate the first passage time for the Poisson process to reach a certain count/level.
        The 'domain' here usually refers to a target count. Assuming domain[0] is start, domain[1] is target count.

        Args:
            domain (tuple[real, real]): Domain (start_count, target_count). target_count must be > start_count.
                                         Typically start_count is 0 for FPT to N events.
            max_duration (real, optional): Maximum physical time to wait. Defaults to 1000.

        Returns:
            float | None: The first passage time (physical time), or None if max_duration is reached.
        """
        a, b = validate_domain(
            domain, domain_type="poisson_fpt", process_name="Poisson FPT"
        )
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.poisson_fpt(
            self.lambda_,
            (a, b),
            max_duration,
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

        Returns:
            float: The total time N(t) spent in the count range [a,b].
        """
        a, b = validate_domain(
            domain,
            domain_type="poisson_occupation",
            process_name="Poisson Occupation Time",
        )
        duration = validate_positive_float(duration, "duration")

        return _core.poisson_occupation_time(
            self.lambda_,
            (a, b),
            duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],  # (start_count, target_count)
        order: int,
        center: bool = False,
        particles: int = 10_000,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(center, "center")
        a, b = validate_domain(
            domain, domain_type="poisson_fpt", process_name="Poisson FPT raw moment"
        )
        order = validate_order(order)
        particles = validate_particles(particles)
        max_duration = validate_positive_float(max_duration, "max_duration")

        return (
            _core.poisson_fpt_raw_moment(
                self.lambda_,
                (a, b),  # _core expects float tuple
                order,
                particles,
                max_duration,
            )
            if not center
            else _core.poisson_fpt_central_moment(
                self.lambda_,
                (a, b),
                order,
                particles,
                max_duration,
            )
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],  # (min_count, max_count)
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
    ) -> float:
        _a, _b = validate_domain(
            domain,
            domain_type="poisson_occupation",
            process_name="Poisson Occupation raw moment",
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float(duration, "duration")

        if _order == 0:
            return 1.0

        return (
            _core.poisson_occupation_time_raw_moment(
                self.lambda_,
                (_a, _b),
                _order,
                _particles,
                _duration,
            )
            if not center
            else _core.poisson_occupation_time_central_moment(
                self.lambda_,
                (_a, _b),
                _order,
                _particles,
                _duration,
            )
        )
