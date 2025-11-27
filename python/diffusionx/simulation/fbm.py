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


class FBM:
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

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        """
        Simulate the fractional Brownian motion.

        Args:
            duration (real): Total duration of the simulation.
            time_step (real, optional): Step size for the simulation. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and positions of the FBM.
        """
        try:
            duration = ensure_float(duration)
            time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if duration <= 0:
            raise ValueError("duration must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.fbm_simulate(
            self.start_position,
            self.hurst_exponent,
            duration,
            time_step,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        """
        Calculate the first passage time of the FBM.

        Args:
            domain (tuple[real, real]): Domain (a, b) for FPT. a must be less than b.
            time_step (real, optional): Step size. Defaults to 0.01.
            max_duration (real, optional): Maximum simulation duration for FPT. Defaults to 1000.

        Returns:
            Optional[float]: The FPT, or None if max_duration reached first.
        """
        a, b = validate_domain(domain, process_name="Fbm FPT")
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.fbm_fpt(
            self.start_position,
            self.hurst_exponent,
            time_step,
            (a, b),
            max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(central, "central")
        a, b = validate_domain(domain, process_name="Fbm FPT raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.fbm_fpt_raw_moment(
                self.start_position,
                self.hurst_exponent,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not central
            else _core.fbm_fpt_central_moment(
                self.start_position,
                self.hurst_exponent,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
        )

        return result

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        """
        Calculate the raw moment of the FBM.

        Args:
            duration (real): Simulation duration.
            order (int): Moment order (non-negative integer).
            particles (int): Number of particles (positive integer).
            time_step (real, optional): Step size. Defaults to 0.01.

        Returns:
            float: The raw moment.
        """
        validate_bool(central, "central")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        result = (
            _core.fbm_raw_moment(
                self.start_position,
                self.hurst_exponent,
                duration,
                time_step,
                order,
                particles,
            )
            if not central
            else _core.fbm_central_moment(
                self.start_position,
                self.hurst_exponent,
                duration,
                time_step,
                order,
                particles,
            )
        )

        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: float = 0.01,
    ) -> float:
        """
        Calculate the occupation time of the FBM in a domain.

        Args:
            domain (tuple[real, real]): Domain (a, b). a must be less than b.
            duration (real): Total simulation duration.
            time_step (real, optional): Step size. Defaults to 0.01.

        Returns:
            float: Occupation time.
        """
        a, b = validate_domain(domain, process_name="Fbm Occupation Time")
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.fbm_occupation_time(
            self.start_position,
            self.hurst_exponent,
            time_step,
            (a, b),
            duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        validate_bool(central, "central")
        a, b = validate_domain(domain, process_name="Fbm Occupation raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        result = (
            _core.fbm_occupation_time_raw_moment(
                self.start_position,
                self.hurst_exponent,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not central
            else _core.fbm_occupation_time_central_moment(
                self.start_position,
                self.hurst_exponent,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
        )

        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.fbm_tamsd(
            self.start_position,
            self.hurst_exponent,
            duration,
            delta,
            time_step,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.fbm_eatamsd(
            self.start_position,
            self.hurst_exponent,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )
