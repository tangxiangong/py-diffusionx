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


class LevyWalk(ContinuousProcess):
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

        Raises:
            TypeError: If alpha, beta, or start_position are not numbers.
            ValueError: If alpha is not in (0, 2] or beta is not positive (general assumption).
        """
        try:
            _alpha = ensure_float(alpha)
            _beta = ensure_float(beta)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (0 < _alpha <= 2):
            raise ValueError(f"alpha (jump exponent) must be in (0, 2], got {_alpha}")
        if (
            _beta <= 0
        ):  # General assumption, might need refinement based on specific model
            raise ValueError(
                f"beta (waiting/velocity param) must be positive, got {_beta}"
            )

        self.alpha = _alpha
        self.beta = _beta
        self.start_position = _start_position

    def simulate(
        self,
        duration: real,
        step_size: float = 0.01,  # step_size interpretation can vary for LW
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.levy_walk_simulate(
            self.start_position,
            self.alpha,
            self.beta,
            _duration,
            _step_size,  # step_size might be a time resolution or number of steps for _core
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
            _core.levy_walk_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                _duration,
                _step_size,
                _order,
                _particles,
            )
            if not center
            else _core.levy_walk_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="LevyWalk FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.levy_walk_fpt(
            self.start_position,
            self.alpha,
            self.beta,
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="LevyWalk FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.levy_walk_fpt_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.levy_walk_fpt_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="LevyWalk Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.levy_walk_occupation_time(
            self.start_position,
            self.alpha,
            self.beta,
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
        _a, _b = validate_domain(domain, process_name="LevyWalk Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.levy_walk_occupation_time_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.levy_walk_occupation_time_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
        )

        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        step_size: float = 0.01,
        quad_order: int = 32,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.levy_walk_tamsd(
            self.start_position,
            self.alpha,
            self.beta,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.levy_walk_eatamsd(
            self.start_position,
            self.alpha,
            self.beta,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
