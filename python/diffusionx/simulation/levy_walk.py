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


class LevyWalk:
    def __init__(
        self,
        alpha: real,  # Jump length distribution exponent
        beta: real,  # Waiting time exponent or velocity parameter
        start_position: real = 0.0,
    ):
        """
        Initialize a Lévy Walk object.

        Args:
            alpha (real): Exponent for jump length distribution (typically (0, 2]).
            beta (real): Exponent for waiting time distribution (e.g., (0, 1] if power-law waiting times)
                         or a velocity parameter if jumps have constant speed between turns.
                         The exact interpretation depends on the underlying Rust implementation.
                         Assuming alpha in (0, 2] and beta > 0 for now.
            start_position (real, optional): Starting position. Defaults to 0.0.
        """
        alpha = validate_positive_float(alpha, "alpha")
        beta = validate_positive_float(beta, "beta")
        start_position = ensure_float(start_position)

        if not (0 < alpha <= 2):
            raise ValueError(f"alpha (jump exponent) must be in (0, 2], got {alpha}")
        self.alpha = alpha
        self.beta = beta
        self.start_position = start_position

    def simulate(
        self,
        duration: real,
        time_step: float = 0.01,  # time_step interpretation can vary for LW
    ) -> tuple[Vector, Vector]:
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.levy_walk_simulate(
            self.start_position,
            self.alpha,
            self.beta,
            duration,
            time_step,  # time_step might be a time resolution or number of steps for _core
        )

    def moment(
        self,
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        validate_bool(center, "center")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.levy_walk_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                duration,
                time_step,
                order,
                particles,
            )
            if not center
            else _core.levy_walk_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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
        a, b = validate_domain(domain, process_name="LevyWalk FPT")
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.levy_walk_fpt(
            self.start_position,
            self.alpha,
            self.beta,
            time_step,
            (a, b),
            max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(center, "center")
        a, b = validate_domain(domain, process_name="LevyWalk FPT raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.levy_walk_fpt_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not center
            else _core.levy_walk_fpt_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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
        a, b = validate_domain(domain, process_name="LevyWalk Occupation Time")
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.levy_walk_occupation_time(
            self.start_position,
            self.alpha,
            self.beta,
            time_step,
            (a, b),
            duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        validate_bool(center, "center")
        a, b = validate_domain(domain, process_name="LevyWalk Occupation raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.levy_walk_occupation_time_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not center
            else _core.levy_walk_occupation_time_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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
        quad_order: int = 32,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.levy_walk_tamsd(
            self.start_position,
            self.alpha,
            self.beta,
            duration,
            delta,
            time_step,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.levy_walk_eatamsd(
            self.start_position,
            self.alpha,
            self.beta,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )
