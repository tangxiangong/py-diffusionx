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


class Gamma:
    def __init__(
        self,
        shape: real,  # Also known as alpha or k
        rate: real,  # Also known as beta or 1/theta (if theta is scale)
        start_position: real = 0.0,
    ):
        """
        Initialize a Gamma process object.
        A Gamma process is a Lévy process with non-decreasing sample paths.
        It is characterized by a shape (or concentration) and a rate parameter.

        Args:
            shape (real): Shape parameter (k > 0).
            rate (real): Rate parameter (beta > 0). The scale parameter is 1/rate.
            start_position (real, optional): Starting position. Defaults to 0.0.
        """
        try:
            shape = ensure_float(shape)
            rate = ensure_float(rate)
            start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if shape <= 0:
            raise ValueError("shape parameter (k) must be positive")
        if rate <= 0:
            raise ValueError("rate parameter (beta) must be positive")

        self.shape = shape
        self.rate = rate
        self.start_position = start_position

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.gamma_simulate(
            self.start_position,  # Transmitted to _core, which should handle it.
            self.shape,
            self.rate,
            duration,
            time_step,
        )

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        validate_bool(central, "central")

        if order == 0:
            return 1.0

        result = (
            _core.gamma_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                duration,
                time_step,
                order,
                particles,
            )
            if not central
            else _core.gamma_central_moment(
                self.start_position,
                self.shape,
                self.rate,
                duration,
                time_step,
                order,
                particles,
            )
        )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        # For Gamma process, domain[0] is typically current value, domain[1] is target threshold.
        # Since it's non-decreasing, domain[0] < domain[1] is essential.
        a, b = validate_domain(domain, process_name="Gamma FPT")
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        # Pass start_position from self, _core.gamma_fpt expects it.
        # The domain (a,b) is relative to the process value space.
        return _core.gamma_fpt(
            self.start_position,  # This is the initial value for the FPT problem
            self.shape,
            self.rate,
            time_step,
            (a, b),  # Target domain for the process value X(t)
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
        a, b = validate_domain(domain, process_name="Gamma FPT raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.gamma_fpt_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not central
            else _core.gamma_fpt_central_moment(
                self.start_position,
                self.shape,
                self.rate,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
        )

        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: float = 0.01,
    ) -> float:
        a, b = validate_domain(domain, process_name="Gamma Occupation Time")
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.gamma_occupation_time(
            self.start_position,
            self.shape,
            self.rate,
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
        a, b = validate_domain(domain, process_name="Gamma Occupation raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.gamma_occupation_time_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not central
            else _core.gamma_occupation_time_central_moment(
                self.start_position,
                self.shape,
                self.rate,
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

        return _core.gamma_tamsd(
            self.start_position,
            self.shape,
            self.rate,
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

        return _core.gamma_eatamsd(
            self.start_position,
            self.shape,
            self.rate,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )
