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


class Cauchy(ContinuousProcess):
    def __init__(
        self,
        start_position: real = 0.0,
    ):
        """
        Initialize a Cauchy process object.

        Args:
            start_position (real, optional): Starting position. Defaults to 0.0.

        Raises:
            TypeError: If start_position is not a number.
        """
        try:
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        self.start_position = _start_position

    def simulate(
        self,
        duration: real,
        step_size: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.cauchy_simulate(
            self.start_position,
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

        result = _core.cauchy_raw_moment(
            self.start_position,
            _duration,
            _step_size,
            _order,
            _particles,
        ) if not central else _core.cauchy_central_moment(
            self.start_position,
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
        _a, _b = validate_domain(domain, process_name="Cauchy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.cauchy_fpt(
            self.start_position,
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
        _a, _b = validate_domain(domain, process_name="Cauchy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")
        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.cauchy_fpt_raw_moment(
            self.start_position,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        ) if not central else _core.cauchy_fpt_central_moment(
            self.start_position,
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
        _a, _b = validate_domain(domain, process_name="Cauchy Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.cauchy_occupation_time(
            self.start_position,
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
        _a, _b = validate_domain(domain, process_name="Cauchy Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.cauchy_occupation_time_raw_moment(
            self.start_position,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        ) if not central else _core.cauchy_occupation_time_central_moment(
            self.start_position,
            self.scale,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.cauchy_tamsd(
            self.start_position,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.cauchy_eatamsd(
            self.start_position,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )


class AsymmetricCauchy(ContinuousProcess):
    def __init__(
        self,
        beta: real = 0.0,
        start_position: real = 0.0,
    ):
        """
        Initialize an Asymmetric Cauchy process object.

        Args:
            beta (real, optional): Skewness parameter. Must be in [-1, 1]. Defaults to 0.0 (symmetric Cauchy).
            start_position (real, optional): Starting position. Defaults to 0.0.
            
        Raises:
            TypeError: If start_position or beta are not numbers.
            ValueError: If beta is not in [-1, 1].
        """
        try:
            _start_position = ensure_float(start_position)
            _beta = ensure_float(beta)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (-1 <= _beta <= 1):
            raise ValueError(
                f"beta (skewness) must be in the range [-1, 1], got {_beta}"
            )

        self.start_position = _start_position
        self.beta = _beta

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_cauchy_simulate(
            self.start_position,
            self.beta,
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

        result = _core.asymmetric_cauchy_raw_moment(
            self.start_position,
            self.beta,
            _duration,
            _step_size,
            _order,
            _particles,
        ) if not central else _core.asymmetric_cauchy_central_moment(
            self.start_position,
            self.beta,
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
        _a, _b = validate_domain(domain, process_name="AsymmetricCauchy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.asymmetric_cauchy_fpt(
            self.start_position,
            self.beta,
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
        _a, _b = validate_domain(domain, process_name="AsymmetricCauchy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")
        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")
        
        result = _core.asymmetric_cauchy_fpt_raw_moment(
            self.start_position,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        ) if not central else _core.asymmetric_cauchy_fpt_central_moment(
            self.start_position,
            self.beta,
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
        _a, _b = validate_domain(
            domain, process_name="AsymmetricCauchy Occupation Time"
        )
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_cauchy_occupation_time(
            self.start_position,
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
        central: bool = True,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="AsymmetricCauchy Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.asymmetric_cauchy_occupation_time_raw_moment(
            self.start_position,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        ) if not central else _core.asymmetric_cauchy_occupation_time_central_moment(
            self.start_position,
            self.beta,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_cauchy_tamsd(
            self.start_position,
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
        particles: int = 10_000,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_cauchy_eatamsd(
            self.start_position,
            self.beta,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
