from diffusionx import _core
from typing import Union, Optional
from .basic import PointProcess
from .utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
import numpy as np


real = Union[float, int]


class CTRW(PointProcess):
    def __init__(
        self,
        alpha: real = 1.0,
        beta: real = 2.0,
        start_position: real = 0.0,
    ):
        """
        Continuous Time Random Walk.

        Args:
            alpha (real, optional): Waiting time exponent (0, 1]. Defaults to 1.0.
            beta (real, optional): Jump length exponent (0, 2]. Defaults to 2.0.
            start_position (real, optional): Starting position. Defaults to 0.0.

        Raises:
            TypeError: If alpha, beta, or start_position are not numbers.
            ValueError: If alpha is not in (0, 1] or beta is not in (0, 2].
        """
        try:
            _alpha = ensure_float(alpha)
            _beta = ensure_float(beta)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (0 < _alpha <= 1):
            raise ValueError(f"alpha must be in the range (0, 1], got {_alpha}")
        if not (0 < _beta <= 2):
            raise ValueError(f"beta must be in the range (0, 2], got {_beta}")

        self.alpha = _alpha
        self.beta = _beta
        self.start_position = _start_position

    def simulate(self, duration: Optional[real] = None, num_step: Optional[int] = None, _step_size: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the CTRW based on total duration.

        Note: `step_size` is not used by this simulation method but is kept for
              consistency with the StochasticProcess interface.

        Args:
            duration (real, optional): Total duration of the simulation.
            num_step (int, optional): Number of steps in the simulation.
            step_size (real, optional): Ignored by this method. Defaults to 0.01.

        Raises:
            TypeError: If duration or num_step is not a number.
            ValueError: If duration or num_step is not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: Times and positions of the CTRW.
        """
        if duration is not None:
            try:
                _duration = ensure_float(duration)
            except TypeError as e:
                raise TypeError(f"duration must be a number. Error: {e}") from e

            if _duration <= 0:
                raise ValueError(f"duration must be positive, got {_duration}")
            # step_size is intentionally not validated here as it's documented as unused.
            return _core.ctrw_simulate_duration(
                self.alpha,
                self.beta,
                self.start_position,
                _duration,
            )
        elif num_step is not None:
            if not isinstance(num_step, int):
                raise TypeError(
                    f"num_step must be an integer, got {type(num_step).__name__}"
                )
            if num_step <= 0:
                raise ValueError(f"num_step must be positive, got {num_step}")
            return _core.ctrw_simulate_step(
                self.alpha,
                self.beta,
                self.start_position,
                num_step,
            )
        else:
            raise ValueError("Either duration or num_step must be provided")
    
    def moment(self, duration: real, order: int, central: bool = True, particles: int = 10_000) -> float: 
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        
        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.ctrw_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            _duration,
            _order,
            _particles,
        ) if not central else _core.ctrw_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            _duration,
            _order,
            _particles,
        )
        return result
            
    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time of the CTRW.

        Args:
            domain (tuple[real, real]): Domain (a, b) for FPT. a must be less than b.
            max_duration (real, optional): Maximum duration to simulate for FPT. Defaults to 1000.

        Raises:
            TypeError: If domain elements or max_duration are not numbers.
            ValueError: If domain is invalid (a >= b) or max_duration not positive.

        Returns:
            Optional[float]: The FPT, or None if max_duration reached first.
        """
        _a, _b = validate_domain(domain, process_name="CTRW FPT")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.ctrw_fpt(
            self.alpha,
            self.beta,
            self.start_position,
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
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="CTRW FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")    
        
        result = _core.ctrw_fpt_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            (_a, _b),
            _order,
            _particles,
            _max_duration,
        ) if not central else _core.ctrw_fpt_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            (_a, _b),
            _order,
            _particles,
            _max_duration,
        )
        return result
    
    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ) -> float:
        """
        Calculate the occupation time of the CTRW in a domain.

        Args:
            domain (tuple[real, real]): Domain (a, b). a must be less than b.
            duration (real): Total simulation duration.

        Raises:
            TypeError: If domain elements or duration are not numbers.
            ValueError: For invalid domain or non-positive duration.

        Returns:
            float: Occupation time.
        """
        _a, _b = validate_domain(domain, process_name="CTRW Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")

        return _core.ctrw_occupation_time(
            self.alpha,
            self.beta,
            self.start_position,
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
    ) -> float:
        _a, _b = validate_domain(domain, process_name="CTRW Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = _core.ctrw_occupation_time_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            (_a, _b),
            _duration,
            _order,
            _particles,
        ) if not central else _core.ctrw_occupation_time_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            (_a, _b),
            _duration,
            _order,
            _particles,
        )
        return result
