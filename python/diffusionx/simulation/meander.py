from diffusionx import _core
from typing import Union, Optional
from .basic import ContinuousProcess
from .utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
import numpy as np

real = Union[float, int]


class BrownianMeander(ContinuousProcess):
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

        Raises:
            TypeError: If diffusion_coefficient is not a number.
            ValueError: If diffusion_coefficient is not positive.
        """
        try:
            _diffusion_coefficient = ensure_float(diffusion_coefficient)
        except TypeError as e:
            raise TypeError(
                f"diffusion_coefficient must be a number. Error: {e}"
            ) from e

        if _diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        self.diffusion_coefficient = _diffusion_coefficient
        # start_position is implicitly 0

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.meander_simulate(
            self.diffusion_coefficient,
            _duration,
            _step_size,
        )

    def moment(
        self,
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.meander_raw_moment(
                self.diffusion_coefficient,
                _duration,
                _step_size,
                _order,
                _particles,
            )
            if not center
            else _core.meander_central_moment(
                self.diffusion_coefficient,
                _duration,
                _step_size,
                _order,
                _particles,
            )
        )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: float = 0.01,
        max_duration: real = 1000,  # This is the meander's fixed duration
    ) -> Optional[float]:
        # FPT for a meander of fixed duration T to a level within (0,T).
        # Domain reflects levels, max_duration is the meander duration.
        _a, _b = validate_domain(domain, process_name="Meander FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (meander duration)"
        )

        return _core.meander_fpt(
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _max_duration,  # This should be the fixed duration of the meander
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
        max_duration: real = 1000,  # Meander duration
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Meander FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (meander duration)"
        )

        result = (
            _core.meander_fpt_raw_moment(
                self.diffusion_coefficient,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.meander_fpt_central_moment(
                self.diffusion_coefficient,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
        )

        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,  # Meander duration
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Meander Occupation Time")
        _duration = validate_positive_float_param(
            duration, "duration (meander duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.meander_occupation_time(
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Meander Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(
            duration, "duration (meander duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.meander_occupation_time_raw_moment(
                self.diffusion_coefficient,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.meander_occupation_time_central_moment(
                self.diffusion_coefficient,
                (_a, _b),
                _duration,
                _order,
                _particles,
            )
        )

        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(
            duration, "duration (meander duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.meander_tamsd(
            self.diffusion_coefficient,
            _duration,
            _delta,
            _step_size,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(
            duration, "duration (meander duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.meander_eatamsd(
            self.diffusion_coefficient,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
