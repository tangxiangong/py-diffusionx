from typing import Callable, Union, Optional
from numpy import ndarray
from .. import _core
from ..simulation.utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)

# Define real for consistency if used, though parameters are often explicitly float/int here
real = Union[float, int]


class Langevin:
    """
    Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dW(t), x(0) = x0

    The underlying implementation has been optimized with an efficient callback mechanism
    for interaction between Python functions and Rust code.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t).
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t).
    start_position : float, optional
        Initial position x0. Defaults to 0.0.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: real = 0.0,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"start_position must be a number. Error: {e}") from e

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = _start_position

    def simulate(self, duration: real, time_step: real) -> tuple[ndarray, ndarray]:
        """
        Simulate the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration or time_step are not numbers.
        ValueError
            If duration or time_step are not positive.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "langevin_simulate"):
            raise NotImplementedError(
                "langevin_simulate not implemented in _core module"
            )
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            _duration,
            _time_step,
        )

    def moment(
        self, duration: real, order: int, central: bool = True, particles: int = 10_000, time_step: real = 0.01
    ) -> float:
        """
        Calculate the moment of the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        order : int
            Order of the moment (must be non-negative).
        particles : int
            Number of particles to simulate (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration, order, particles, or time_step have incorrect types.
        ValueError
            If parameters have invalid values.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        float
            Raw moment.
        """
        if not hasattr(_core, "langevin_raw_moment"):
            raise NotImplementedError(
                "langevin_raw_moment not implemented in _core module"
            )

        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        result = _core.langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        ) if not central else _core.langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
                _duration,
                order,
                particles,
                _time_step,
            )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Langevin FPT")
        _time_step = validate_positive_float_param(time_step, "time_step")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            _time_step,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Langevin FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = _core.langevin_fpt_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        ) if not central else _core.langevin_fpt_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Langevin Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            _time_step,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Langevin Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        if _order == 0:
            return 1.0

        result = _core.langevin_occupation_time_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        ) if not central else _core.langevin_occupation_time_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        )
        return result
    
    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            _duration,
            _delta,
            _time_step,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            _duration,
            _delta,
            _particles, 
            _time_step,
            quad_order,
        )


class GeneralizedLangevin:
    """
    Generalized Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dL_alpha(t), x(0) = x0

    dL_alpha(t) is a Lévy stable noise source.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t).
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t).
    start_position : real, optional
        Initial position x0. Defaults to 0.0.
    alpha : real, optional
        Stability index of the stable noise (0, 2]. Defaults to 1.5.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: real = 0.0,
        alpha: real = 1.5,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            _start_position = ensure_float(start_position)
            _alpha = ensure_float(alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < _alpha <= 2):
            raise ValueError(
                f"alpha (stability index) must be in the range (0, 2], got {_alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = _start_position
        self.alpha = _alpha

    def simulate(self, duration: real, time_step: float = 0.01) -> tuple[ndarray, ndarray]:
        """
        Simulate the Generalized Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration or time_step are not numbers.
        ValueError
            If duration or time_step are not positive.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "generalized_langevin_simulate"):
            raise NotImplementedError(
                "generalized_langevin_simulate not implemented in _core module"
            )
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.generalized_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            _time_step,
        )

    def moment(
        self, duration: real, order: int, central: bool = True, particles: int = 10_000, time_step: real = 0.01
    ) -> float:
        """
        Calculate the moment of the Generalized Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        order : int
            Order of the moment (must be non-negative).
        particles : int
            Number of particles (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If parameters have incorrect types.
        ValueError
            If parameters have invalid values.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        float
            Raw moment.
        """
        if not hasattr(_core, "generalized_langevin_raw_moment"):
            raise NotImplementedError(
                "generalized_langevin_raw_moment not implemented in _core module"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        result = _core.generalized_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            order,
            particles,
            _time_step,
        ) if not central else _core.generalized_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            order,
            particles,
            _time_step,
        )
        return result

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="GeneralizedLangevin FPT")
        _time_step = validate_positive_float_param(time_step, "time_step")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.generalized_langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            _time_step,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="GeneralizedLangevin FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = _core.generalized_langevin_fpt_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        ) if not central else _core.generalized_langevin_fpt_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="GeneralizedLangevin Occupation Time"
        )
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.generalized_langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            _time_step,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="GeneralizedLangevin Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        if _order == 0:
            return 1.0

        result = _core.generalized_langevin_occupation_time_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        ) if not central else _core.generalized_langevin_occupation_time_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        )
        return result
    
    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.generalized_langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            _duration,
            _delta,
            _time_step,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.generalized_langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            _duration,
            _delta,
            _particles,
            _time_step,
            quad_order,
        )
        

class SubordinatedLangevin:
    """
    Subordinated Langevin equation driven by a subordinator process for operational time.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, u) where u is operational time.
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, u) where u is operational time.
    subordinator_alpha : real, optional
        Alpha parameter for the subordinator (0, 1). Defaults to 0.7.
    start_position : real, optional
        Initial position x0. Defaults to 0.0.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        subordinator_alpha: real = 0.7,
        start_position: real = 0.0,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            _start_position = ensure_float(start_position)
            _subordinator_alpha = ensure_float(subordinator_alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and subordinator_alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < _subordinator_alpha < 1):
            raise ValueError(
                f"subordinator_alpha must be in the range (0, 1), got {_subordinator_alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.subordinator_alpha = _subordinator_alpha
        self.start_position = _start_position

    def simulate(self, duration: real, time_step: float = 0.01) -> tuple[ndarray, ndarray]:
        """
        Simulate the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.simulate)
        """
        if not hasattr(_core, "subordinated_langevin_simulate"):
            raise NotImplementedError(
                "subordinated_langevin_simulate not implemented in _core module"
            )
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.subordinated_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
            _time_step,
        )

    def moment(
        self, duration: real, order: int, central: bool = True, particles: int = 10_000, time_step: float = 0.01
    ) -> float:
        """
        Calculate the raw moment of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.raw_moment)
        """
        if not hasattr(_core, "subordinated_langevin_raw_moment"):
            raise NotImplementedError(
                "subordinated_langevin_raw_moment not implemented in _core module"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        result = _core.subordinated_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        ) if not central else _core.subordinated_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
        )
        return result
    
    def fpt(
        self, domain: tuple[real, real], max_duration: real = 1000
    ) -> Optional[float]:  # No time_step here as per _core
        _a, _b = validate_domain(domain, process_name="SubordinatedLangevin FPT")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.subordinated_langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(
            domain, process_name="SubordinatedLangevin FPT raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = _core.subordinated_langevin_fpt_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        ) if not central else _core.subordinated_langevin_fpt_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _max_duration,
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="SubordinatedLangevin Occupation Time"
        )
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.subordinated_langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            _time_step,
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
        time_step: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="SubordinatedLangevin Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _time_step = validate_positive_float_param(time_step, "time_step")

        if _order == 0:
            return 1.0

        result = _core.subordinated_langevin_occupation_time_raw_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        ) if not central else _core.subordinated_langevin_occupation_time_central_moment(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (_a, _b),
            _order,
            _particles,
            _time_step,
            _duration,
        )
        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.subordinated_langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.subordinator_alpha,
            _duration,
            _delta,
            _time_step,
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
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _time_step = validate_positive_float_param(time_step, "time_step")

        return _core.subordinated_langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.subordinator_alpha,
            _duration,
            _delta,
            _particles,
            _time_step,
            quad_order,
        )
