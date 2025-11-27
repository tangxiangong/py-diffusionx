from diffusionx import _core
from .basic import real, Vector
from .utils import (
    validate_bool,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float,
    validate_positive_integer,
)


class BrownianMeander:
    def __init__(
        self,
        diffusion_coefficient: real = 1.0,
    ):
        """
        Initialize a Brownian Meander object.
        A Brownian meander is a Brownian motion conditioned to be positive up to a specified duration.
        The start_position is implicitly 0.

        Args:
            diffusion_coefficient (real, optional): Diffusion coefficient. Defaults to 1.0.
        """
        diffusion_coefficient = validate_positive_float(
            diffusion_coefficient, "diffusion_coefficient"
        )

        self.diffusion_coefficient = diffusion_coefficient

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.meander_simulate(
            self.diffusion_coefficient,
            duration,
            time_step,
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
            _core.meander_raw_moment(
                self.diffusion_coefficient,
                duration,
                time_step,
                order,
                particles,
            )
            if not center
            else _core.meander_central_moment(
                self.diffusion_coefficient,
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
    ) -> float | None:
        a, b = validate_domain(domain, process_name="Meander FPT")
        time_step = validate_positive_float(time_step, "time_step")
        return _core.meander_fpt(
            self.diffusion_coefficient,
            time_step,
            (a, b),
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float | None:
        validate_bool(center, "center")
        a, b = validate_domain(domain, process_name="Meander FPT raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")

        result = (
            _core.meander_fpt_raw_moment(
                self.diffusion_coefficient,
                (a, b),
                order,
                particles,
                time_step,
            )
            if not center
            else _core.meander_fpt_central_moment(
                self.diffusion_coefficient,
                (a, b),
                order,
                particles,
                time_step,
            )
        )

        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,  # Meander duration
        time_step: float = 0.01,
    ) -> float:
        a, b = validate_domain(domain, process_name="Meander Occupation Time")
        duration = validate_positive_float(duration, "duration (meander duration)")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.meander_occupation_time(
            self.diffusion_coefficient,
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
        a, b = validate_domain(domain, process_name="Meander Occupation raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration (meander duration)")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.meander_occupation_time_raw_moment(
                self.diffusion_coefficient,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not center
            else _core.meander_occupation_time_central_moment(
                self.diffusion_coefficient,
                (a, b),
                duration,
                order,
                particles,
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
        duration = validate_positive_float(duration, "duration (meander duration)")
        delta = validate_positive_float(delta, "delta")
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.meander_tamsd(
            self.diffusion_coefficient,
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
        duration = validate_positive_float(duration, "duration (meander duration)")
        delta = validate_positive_float(delta, "delta")
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.meander_eatamsd(
            self.diffusion_coefficient,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )
