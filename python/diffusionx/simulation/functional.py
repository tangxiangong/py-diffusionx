from .basic import StochasticProcess
from .bm import Bm
from .levy import Levy
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

        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )

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

    def simulate(self) -> Optional[float]:
        match self.sp:
            case Bm():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
                return result
            case Levy():
                result = self.sp.fpt(self.domain, self.step_size, self.max_duration)
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

        if a >= b:
            raise ValueError(
                f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]."
            )

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
            case _:
                raise NotImplementedError(
                    f"OccupationTime calculation is not implemented for process type {type(self.sp).__name__}"
                )
