from .basic import ContinuousProcess, PointProcess
from typing import Union, Optional
from .utils import ensure_float

real = Union[int, float]


class FPT:
    def __init__(
        self,
        sp: ContinuousProcess | PointProcess,
        domain: tuple[real, real],
    ):
        if not isinstance(sp, ContinuousProcess):
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
        self.sp = sp
        self.domain = (a, b)  # Store the float-converted domain

    def simulate(self, max_duration: real = 1000,
        step_size: float = 0.01) -> Optional[float]:
        try:
            max_duration = ensure_float(max_duration)
        except TypeError as e:
            raise TypeError(f"max_duration must be a number. Error: {e}") from e
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")

        try:
            step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(f"step_size must be a number. Error: {e}") from e
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        if isinstance(self.sp, ContinuousProcess):
            try: 
                result = self.sp.fpt(self.domain, max_duration, step_size)
            except NotImplementedError:
                raise NotImplementedError(
                        f"FPT calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        elif isinstance(self.sp, PointProcess):
            try: 
                result = self.sp.fpt(self.domain, max_duration)
            except NotImplementedError:
                raise NotImplementedError(
                        f"FPT calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        else:
            raise ValueError("sp must be a ContinuousProcess or PointProcess")
        return result
    
    def moment(self, order: int, central: bool = True, particles: int = 10_000, max_duration: real = 1000, step_size: float = 0.01) -> float: 
        if isinstance(self.sp, ContinuousProcess):
            try: 
                result = self.sp.fpt_moment(self.domain, order, central, particles, max_duration, step_size)
            except NotImplementedError:
                raise NotImplementedError(
                        f"FPT moment calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        elif isinstance(self.sp, PointProcess):
            try: 
                result = self.sp.fpt_moment(self.domain, order, central, particles, max_duration)
            except NotImplementedError:
                raise NotImplementedError(
                        f"FPT moment calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        else:
            raise ValueError("sp must be a ContinuousProcess or PointProcess")
        return result

class OccupationTime:
    def __init__(
        self,
        sp: ContinuousProcess | PointProcess,
        domain: tuple[real, real],
    ):
        if not isinstance(sp, ContinuousProcess | PointProcess):
            raise TypeError(
                f"sp must be an instance of ContinuousProcess or PointProcess, got {type(sp).__name__}"
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

        self.sp = sp
        self.domain = (a, b)  # Store the float-converted domain

    def simulate(self, duration: real, step_size: float = 0.01) -> float:
        try:
            duration = ensure_float(duration)
        except TypeError as e:
            raise TypeError(f"duration must be a number. Error: {e}") from e
        if duration <= 0:
            raise ValueError("duration must be positive")

        if isinstance(self.sp, ContinuousProcess):
            try:
                step_size = ensure_float(step_size)
            except TypeError as e:
                raise TypeError(f"step_size must be a number. Error: {e}") from e
            if step_size <= 0:
                raise ValueError("step_size must be positive")

            try: 
                result = self.sp.occupation_time(self.domain, duration, step_size)
            except NotImplementedError:
                raise NotImplementedError(
                        f"Occupation time calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        elif isinstance(self.sp, PointProcess):
            try: 
                result = self.sp.occupation_time(self.domain, duration)
            except NotImplementedError:
                raise NotImplementedError(
                    f"Occupation time calculation is not implemented for process type {type(self.sp).__name__}"
                )
        else:
            raise ValueError("sp must be a ContinuousProcess or PointProcess")
        return result

    def moment(self, duration: real, order: int, central: bool = True, particles: int = 10_000, step_size: float = 0.01) -> float: 
        if isinstance(self.sp, ContinuousProcess):
            try: 
                result = self.sp.occupation_time_moment(self.domain, duration, order, central, particles, step_size)
            except NotImplementedError:
                raise NotImplementedError(
                        f"Occupation time moment calculation is not implemented for process type {type(self.sp).__name__}"
                    )
        elif isinstance(self.sp, PointProcess):
            try: 
                result = self.sp.occupation_time_moment(self.domain, duration, order, central, particles)
            except NotImplementedError:
                raise NotImplementedError(
                    f"Occupation time moment calculation is not implemented for process type {type(self.sp).__name__}"
                )
        else:
            raise ValueError("sp must be a ContinuousProcess or PointProcess")
        return result
