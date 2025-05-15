from diffusionx import _core
from typing import Union, Optional
from .basic import ContinuousProcess
from .utils import (
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
import numpy as np

real = Union[float, int]


class BrownianExcursion(ContinuousProcess):
    def __init__(self):
        """
        Initialize a Brownian Excursion object.
        A Brownian excursion is a Brownian motion conditioned to be positive and to return to 0 at a specified duration.
        """

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.be_simulate(
            _duration,
            _step_size,
        )

    def moment(
        self, duration: real, order: int, particles: int = 10_000, step_size: float = 0.01, central: bool = True
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.be_raw_moment(
            _duration,
            _step_size,
            _order,
            _particles,
        ) if not central else _core.be_central_moment(
            _duration,
            _step_size,
            _order,
            _particles,
        )
        return result

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
        step_size: float = 0.01,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Be FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (excursion duration)"
        )

        return _core.be_fpt(
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = True,
        particles: int = 10_000,
        max_duration: real = 1000,
        step_size: float = 0.01,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Brownian excursion FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (excursion duration)"
        )

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.be_fpt_raw_moment(
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        ) if not central else _core.be_fpt_central_moment(
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        )
        return result


    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Brownian excursion Occupation Time")
        _duration = validate_positive_float_param(
            duration, "duration (excursion duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.be_occupation_time(
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = True,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Be Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(
            duration, "duration (excursion duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.be_occupation_time_raw_moment(
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        ) if not central else _core.be_occupation_time_central_moment(
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
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
            duration, "duration (excursion duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.be_tamsd(
            _duration,
            _delta,
            _step_size,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(
            duration, "duration (excursion duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.be_eatamsd(
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
