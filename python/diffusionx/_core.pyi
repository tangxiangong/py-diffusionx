import numpy as np
from typing import Callable

def exp_rand(scale: float = 1.0) -> float: ...
def exp_rands(n: int, scale: float = 1.0) -> np.ndarray: ...
def uniform_rand_float(
    low: float = 0.0, high: float = 1.0, end: bool = False
) -> float: ...
def uniform_rands_float(
    n: int, low: float = 0.0, high: float = 1.0, end: bool = False
) -> np.ndarray: ...
def uniform_rand_int(low: int, high: int, end: bool = False) -> int: ...
def uniform_rands_int(n: int, low: int, high: int, end: bool = False) -> np.ndarray: ...
def normal_rand(mu: float = 0.0, sigma: float = 1.0) -> float: ...
def normal_rands(n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray: ...
def poisson_rand(lambda_: float = 1.0) -> int: ...
def poisson_rands(n: int, lambda_: float = 1.0) -> np.ndarray: ...
def stable_rand(
    alpha: float, beta: float, sigma: float = 1.0, mu: float = 0.0
) -> float: ...
def stable_rands(
    n: int, alpha: float, beta: float, sigma: float = 1.0, mu: float = 0.0
) -> np.ndarray: ...
def skew_stable_rand(alpha: float) -> float: ...
def skew_stable_rands(n: int, alpha: float) -> np.ndarray: ...
def bool_rand(p: float = 0.5) -> bool: ...
def bool_rands(n: int, p: float = 0.5) -> np.ndarray: ...

# Brownian Motion
def bm_simulate(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def bm_raw_moment(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bm_central_moment(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bm_fpt(
    start_position: float,
    diffusion_coefficient: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def bm_fpt_raw_moment(
    start_position: float,
    diffusion_coefficient: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def bm_fpt_central_moment(
    start_position: float,
    diffusion_coefficient: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def bm_occupation_time(
    start_position: float,
    diffusion_coefficient: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def bm_occupation_time_raw_moment(
    start_position: float,
    diffusion_coefficient: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def bm_occupation_time_central_moment(
    start_position: float,
    diffusion_coefficient: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def bm_tamsd(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def bm_eatamsd(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Levy Process
def levy_simulate(
    start_position: float,
    alpha: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def levy_fpt(
    start_position: float,
    alpha: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def levy_occupation_time(
    start_position: float,
    alpha: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def levy_fpt_raw_moment(
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def levy_fpt_central_moment(
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def levy_occupation_time_raw_moment(
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def levy_occupation_time_central_moment(
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def levy_tamsd(
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def levy_eatamsd(
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Poisson Process
def poisson_simulate_duration(
    lambda_: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def poisson_simulate_step(
    lambda_: float,
    num_step: int,
) -> tuple[np.ndarray, np.ndarray]: ...
def poisson_raw_moment(
    lambda_: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def poisson_central_moment(
    lambda_: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def poisson_fpt(
    lambda_: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def poisson_occupation_time(
    lambda_: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def poisson_fpt_raw_moment(
    lambda_: float, domain: tuple[float, float], order: int, particles: int, max_duration: float
) -> float | None: ...
def poisson_fpt_central_moment(
    lambda_: float, domain: tuple[float, float], order: int, particles: int, max_duration: float
) -> float | None: ...
def poisson_occupation_time_raw_moment(
    lambda_: float, domain: tuple[float, float], duration: float, order: int, particles: int
) -> float: ...
def poisson_occupation_time_central_moment(
    lambda_: float, domain: tuple[float, float], duration: float, order: int, particles: int
) -> float: ...

# Subordinator Process
def subordinator_simulate(
    alpha: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def subordinator_fpt(
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
    step_size: float,
) -> float | None: ...
def subordinator_occupation_time(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    step_size: float,
) -> float: ...
def subordinator_fpt_raw_moment(
    alpha: float, domain: tuple[float, float], order: int, particles: int, max_duration: float, step_size: float
) -> float | None: ...
def subordinator_fpt_central_moment(
    alpha: float, domain: tuple[float, float], order: int, particles: int, max_duration: float, step_size: float
) -> float | None: ...
def subordinator_occupation_time_raw_moment(
    alpha: float, domain: tuple[float, float], duration: float, order: int, particles: int, step_size: float
) -> float: ...
def subordinator_occupation_time_central_moment(
    alpha: float, domain: tuple[float, float], duration: float, order: int, particles: int, step_size: float
) -> float: ...
def inv_subordinator_simulate(
    alpha: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def inv_subordinator_raw_moment(
    alpha: float, duration: float, order: int, particles: int, step_size: float
) -> float: ...
def inv_subordinator_central_moment(
    alpha: float, duration: float, order: int, particles: int, step_size: float
) -> float: ...
def inv_subordinator_fpt(
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
    step_size: float,
) -> float | None: ...
def inv_subordinator_fpt_raw_moment(
    alpha: float, domain: tuple[float, float], order: int, particles: int, max_duration: float, step_size: float
) -> float | None: ...
def inv_subordinator_fpt_central_moment(
    alpha: float, domain: tuple[float, float], order: int, particles: int, max_duration: float, step_size: float
) -> float | None: ...
def inv_subordinator_occupation_time(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    step_size: float,
) -> float: ...
def inv_subordinator_occupation_time_raw_moment(
    alpha: float, domain: tuple[float, float], duration: float, order: int, particles: int, step_size: float
) -> float: ...
def inv_subordinator_occupation_time_central_moment(
    alpha: float, domain: tuple[float, float], duration: float, order: int, particles: int, step_size: float
) -> float: ...

# Fractional Brownian motion
def fbm_simulate(
    start_position: float,
    hurst_exponent: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def fbm_raw_moment(
    start_position: float,
    hurst_exponent: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def fbm_central_moment(
    start_position: float,
    hurst_exponent: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def fbm_fpt(
    start_position: float,
    hurst_exponent: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def fbm_fpt_raw_moment(
    start_position: float,
    hurst_exponent: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def fbm_fpt_central_moment(
    start_position: float,
    hurst_exponent: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def fbm_occupation_time(
    start_position: float,
    hurst_exponent: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def fbm_occupation_time_raw_moment(
    start_position: float,
    hurst_exponent: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def fbm_occupation_time_central_moment(
    start_position: float,
    hurst_exponent: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def fbm_tamsd(
    start_position: float,
    hurst_exponent: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def fbm_eatamsd(
    start_position: float,
    hurst_exponent: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Continuous Time Random Walk
def ctrw_simulate_duration(
    alpha: float,
    beta: float,
    start_position: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def ctrw_simulate_step(
    alpha: float,
    beta: float,
    start_position: float,
    num_step: int,
) -> tuple[np.ndarray, np.ndarray]: ...
def ctrw_raw_moment(
    alpha: float,
    beta: float,
    start_position: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def ctrw_central_moment(
    alpha: float,
    beta: float,
    start_position: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def ctrw_fpt(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def ctrw_fpt_raw_moment(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
) -> float | None: ...
def ctrw_fpt_central_moment(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
) -> float | None: ...
def ctrw_occupation_time(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def ctrw_occupation_time_raw_moment(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def ctrw_occupation_time_central_moment(
    alpha: float,
    beta: float,
    start_position: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
) -> float: ...

# Langevin Process
def langevin_simulate(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    duration: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def langevin_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def langevin_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def langevin_fpt(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def langevin_fpt_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def langevin_fpt_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def langevin_occupation_time(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def langevin_occupation_time_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def langevin_occupation_time_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def langevin_tamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    duration: float,
    delta: float,
    time_step: float,
    quad_order: int,
) -> float: ...
def langevin_eatamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    duration: float,
    delta: float,
    particles: int,
    time_step: float,
    quad_order: int,
) -> float: ...

# Generalized Langevin Process
def generalized_langevin_simulate(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def generalized_langevin_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def generalized_langevin_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def generalized_langevin_fpt(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
    time_step: float,
) -> float | None: ...
def generalized_langevin_fpt_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def generalized_langevin_fpt_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def generalized_langevin_occupation_time(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    time_step: float,
) -> float: ...
def generalized_langevin_occupation_time_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def generalized_langevin_occupation_time_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def generalized_langevin_tamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    time_step: float,
    quad_order: int,
) -> float: ...
def generalized_langevin_eatamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    particles: int,
    time_step: float,
    quad_order: int,
) -> float: ...

# Subordinated Langevin Process
def subordinated_langevin_simulate(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def subordinated_langevin_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def subordinated_langevin_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def subordinated_langevin_fpt(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def subordinated_langevin_fpt_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def subordinated_langevin_fpt_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    time_step: float,
) -> float | None: ...
def subordinated_langevin_occupation_time(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    time_step: float,
) -> float: ...
def subordinated_langevin_occupation_time_raw_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def subordinated_langevin_occupation_time_central_moment(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    time_step: float,
) -> float: ...
def subordinated_langevin_tamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    time_step: float,
    quad_order: int,
) -> float: ...
def subordinated_langevin_eatamsd(
    drift_func: Callable[[float, float], float],
    diffusion_func: Callable[[float, float], float],
    start_position: float,
    alpha: float,
    duration: float,
    delta: float,
    particles: int,
    time_step: float,
    quad_order: int,
) -> float: ...
