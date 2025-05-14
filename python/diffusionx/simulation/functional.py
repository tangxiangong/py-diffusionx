from .basic import StochasticProcess
from .bm import Bm
from .levy import Levy, Subordinator, InvSubordinator
from .fbm import Fbm
from .ctrw import CTRW
from .poisson import Poisson
from .langevin import Langevin, GeneralizedLangevin, SubordinatedLangevin
from typing import Union, Optional
from .utils import ensure_float

real = Union[int, float]


class FPT:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        max_duration: real = 1000,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )

        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )

        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e

        # Specific domain validation (e.g., a < b or Poisson rules) is now handled
        # by the underlying stochastic process methods via utils.validate_domain.
        # self.domain = (a, b) # Will be set later after all validations

        try:
            self.max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(f"max_duration must be a number. Error: {e}") from e
        if self.max_duration <= 0:
            raise ValueError("max_duration must be positive")

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)  # Store the float-converted domain

    def simulate(self) -> Optional[float]:
        match self.sp:
            case Bm():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case Levy():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case Fbm():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case Subordinator():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case InvSubordinator():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case CTRW():
                result = self.sp.fpt(self.domain, self.max_duration)
                return result
            case Poisson():
                result = self.sp.fpt(self.domain, self.max_duration)
                return result
            case Langevin():
                result = self.sp.fpt(
                    self.domain,
                    time_step=self.step_size,
                    max_duration=self.max_duration,
                )
                return result
            case GeneralizedLangevin():
                result = self.sp.fpt(
                    self.domain,
                    time_step=self.step_size,
                    max_duration=self.max_duration,
                )
                return result
            case SubordinatedLangevin():
                result = self.sp.fpt(self.domain, self.max_duration)
                return result
            case _:
                raise NotImplementedError(
                    f"FPT calculation is not implemented for process type {type(self.sp).__name__}"
                )


class OccupationTime:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )

        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )

        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e

        # Specific domain validation (e.g., a < b or Poisson rules) is now handled
        # by the underlying stochastic process methods via utils.validate_domain.
        # self.domain = (a, b) # Will be set later

        try:
            _duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        else:
            self.duration = _duration

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)  # Store the float-converted domain

    def simulate(self) -> float:
        match self.sp:
            case Bm():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case Levy():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case Fbm():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case Subordinator():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case InvSubordinator():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case CTRW():
                return self.sp.occupation_time(self.domain, self.duration)
            case Poisson():
                return self.sp.occupation_time(self.domain, self.duration)
            case Langevin():
                return self.sp.occupation_time(
                    self.domain, self.duration, time_step=self.step_size
                )
            case GeneralizedLangevin():
                return self.sp.occupation_time(
                    self.domain, self.duration, time_step=self.step_size
                )
            case SubordinatedLangevin():
                return self.sp.occupation_time(
                    self.domain, self.duration, time_step=self.step_size
                )
            case _:
                raise NotImplementedError(
                    f"OccupationTime calculation is not implemented for process type {type(self.sp).__name__}"
                )


class FPTRawMoment:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        order: int,
        particles: int,
        max_duration: real = 1000,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e
        # For Poisson, domain interpretation (counts vs. spatial) is handled by the underlying process method.
        # Here, we only validate basic structure. Specific domain validation (e.g. a < b) is in the process methods.
        # This was already the case, so no change needed here for domain logic itself.

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
            self.max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(f"max_duration must be a number. Error: {e}") from e
        if self.max_duration <= 0:
            raise ValueError("max_duration must be positive")

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            # step_size is not used by CTRW/Poisson, but still validated if provided
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)
        self.order = order
        self.particles = particles

    def simulate(self) -> Optional[float]:
        match self.sp:
            case Bm() | Fbm() | Levy() | Subordinator() | InvSubordinator():
                return self.sp.fpt_raw_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.step_size,
                    self.max_duration,
                )
            case CTRW() | Poisson():
                return self.sp.fpt_raw_moment(
                    self.domain, self.order, self.particles, self.max_duration
                )
            case Langevin() | GeneralizedLangevin() | SubordinatedLangevin():
                return self.sp.fpt_raw_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    max_duration=self.max_duration,
                    time_step=self.step_size,
                )
            case _:
                raise NotImplementedError(
                    f"FPTRawMoment calculation is not implemented for process type {type(self.sp).__name__}"
                )


class FPTCentralMoment:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        order: int,
        particles: int,
        max_duration: real = 1000,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e
        # Specific domain logic (a < b etc.) is handled by underlying process methods.

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
            self.max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(f"max_duration must be a number. Error: {e}") from e
        if self.max_duration <= 0:
            raise ValueError("max_duration must be positive")

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)
        self.order = order
        self.particles = particles

    def simulate(self) -> Optional[float]:
        match self.sp:
            case Bm() | Fbm() | Levy() | Subordinator() | InvSubordinator():
                return self.sp.fpt_central_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.step_size,
                    self.max_duration,
                )
            case CTRW() | Poisson():
                return self.sp.fpt_central_moment(
                    self.domain, self.order, self.particles, self.max_duration
                )
            case Langevin() | GeneralizedLangevin() | SubordinatedLangevin():
                return self.sp.fpt_central_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    max_duration=self.max_duration,
                    time_step=self.step_size,
                )
            case _:
                raise NotImplementedError(
                    f"FPTCentralMoment calculation is not implemented for process type {type(self.sp).__name__}"
                )


class OccupationTimeRawMoment:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e
        # Specific domain logic (a < b etc.) is handled by underlying process methods.

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
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        else:
            self.duration = _duration

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)
        self.order = order
        self.particles = particles

    def simulate(
        self,
    ) -> (
        float
    ):  # Assuming occupation time moments always return float, not Optional[float]
        match self.sp:
            case Bm() | Fbm() | Levy() | Subordinator() | InvSubordinator():
                return self.sp.occupation_time_raw_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.duration,
                    self.step_size,
                )
            case CTRW() | Poisson():
                return self.sp.occupation_time_raw_moment(
                    self.domain, self.order, self.particles, self.duration
                )
            case Langevin() | GeneralizedLangevin() | SubordinatedLangevin():
                return self.sp.occupation_time_raw_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.duration,
                    time_step=self.step_size,
                )
            case _:
                raise NotImplementedError(
                    f"OccupationTimeRawMoment calculation is not implemented for process type {type(self.sp).__name__}"
                )


class OccupationTimeCentralMoment:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise TypeError(
                f"sp must be an instance of StochasticProcess, got {type(sp).__name__}"
            )
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise TypeError(
                f"domain must be a tuple of two real numbers, got {type(domain).__name__}"
            )
        try:
            a = ensure_float(domain[0])
            b = ensure_float(domain[1])
        except TypeError as e:
            raise TypeError(
                f"Domain elements must be numbers convertible to float. Error: {e}"
            ) from e
        # Specific domain logic (a < b etc.) is handled by underlying process methods.

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
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        else:
            self.duration = _duration

        try:
            self.step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        self.sp = sp
        self.domain = (a, b)
        self.order = order
        self.particles = particles

    def simulate(self) -> float:  # Assuming occupation time moments always return float
        match self.sp:
            case Bm() | Fbm() | Levy() | Subordinator() | InvSubordinator():
                return self.sp.occupation_time_central_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.duration,
                    self.step_size,
                )
            case CTRW() | Poisson():
                return self.sp.occupation_time_central_moment(
                    self.domain, self.order, self.particles, self.duration
                )
            case Langevin() | GeneralizedLangevin() | SubordinatedLangevin():
                return self.sp.occupation_time_central_moment(
                    self.domain,
                    self.order,
                    self.particles,
                    self.duration,
                    time_step=self.step_size,
                )
            case _:
                raise NotImplementedError(
                    f"OccupationTimeCentralMoment calculation is not implemented for process type {type(self.sp).__name__}"
                )
