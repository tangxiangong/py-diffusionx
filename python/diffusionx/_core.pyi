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
    lambda_: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
) -> float | None: ...
def poisson_fpt_central_moment(
    lambda_: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
) -> float | None: ...
def poisson_occupation_time_raw_moment(
    lambda_: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def poisson_occupation_time_central_moment(
    lambda_: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
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
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def subordinator_fpt_central_moment(
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def subordinator_occupation_time_raw_moment(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def subordinator_occupation_time_central_moment(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
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
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def inv_subordinator_fpt_central_moment(
    alpha: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def inv_subordinator_occupation_time(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    step_size: float,
) -> float: ...
def inv_subordinator_occupation_time_raw_moment(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def inv_subordinator_occupation_time_central_moment(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
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

# Asymmetric Levy Process
def asymmetric_levy_simulate(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def asymmetric_levy_fpt(
    start_position: float,
    alpha: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def asymmetric_levy_fpt_raw_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def asymmetric_levy_fpt_central_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    max_duration: float,
    step_size: float,
) -> float | None: ...
def asymmetric_levy_occupation_time(
    start_position: float,
    alpha: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def asymmetric_levy_occupation_time_raw_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def asymmetric_levy_occupation_time_central_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    duration: float,
    order: int,
    particles: int,
    step_size: float,
) -> float: ...
def asymmetric_levy_tamsd(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def asymmetric_levy_eatamsd(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Brownian Bridge
def bb_simulate(
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def bb_raw_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bb_central_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bb_fpt(
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def bb_fpt_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def bb_fpt_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def bb_occupation_time(
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def bb_occupation_time_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def bb_occupation_time_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def bb_tamsd(
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def bb_eatamsd(
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Brownian Excursion
def be_simulate(
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def be_raw_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def be_central_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def be_fpt(
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def be_fpt_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def be_fpt_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def be_occupation_time(
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def be_occupation_time_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def be_occupation_time_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def be_tamsd(
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def be_eatamsd(
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Brownian Meander
def meander_simulate(
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def meander_raw_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def meander_central_moment(
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def meander_fpt(
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def meander_fpt_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def meander_fpt_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def meander_occupation_time(
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def meander_occupation_time_raw_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def meander_occupation_time_central_moment(
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def meander_tamsd(
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def meander_eatamsd(
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Cauchy Process
def cauchy_simulate(
    start_position: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def cauchy_raw_moment(
    start_position: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def cauchy_central_moment(
    start_position: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def cauchy_fpt(
    start_position: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def cauchy_fpt_raw_moment(
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def cauchy_fpt_central_moment(
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def cauchy_occupation_time(
    start_position: float,
    scale: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def cauchy_occupation_time_raw_moment(
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def cauchy_occupation_time_central_moment(
    start_position: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def cauchy_tamsd(
    start_position: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def cauchy_eatamsd(
    start_position: float,
    scale: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Asymmetric Cauchy Process
def asymmetric_cauchy_simulate(
    start_position: float,
    beta: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def asymmetric_cauchy_raw_moment(
    start_position: float,
    beta: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def asymmetric_cauchy_central_moment(
    start_position: float,
    beta: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def asymmetric_cauchy_fpt(
    start_position: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def asymmetric_cauchy_fpt_raw_moment(
    start_position: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def asymmetric_cauchy_fpt_central_moment(
    start_position: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def asymmetric_cauchy_occupation_time(
    start_position: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def asymmetric_cauchy_occupation_time_raw_moment(
    start_position: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def asymmetric_cauchy_occupation_time_central_moment(
    start_position: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def asymmetric_cauchy_tamsd(
    start_position: float,
    beta: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def asymmetric_cauchy_eatamsd(
    start_position: float,
    beta: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Gamma Process
def gamma_simulate(
    start_position: float,
    shape: float,
    rate: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def gamma_raw_moment(
    start_position: float,
    shape: float,
    rate: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def gamma_central_moment(
    start_position: float,
    shape: float,
    rate: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def gamma_fpt(
    start_position: float,
    shape: float,
    rate: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def gamma_fpt_raw_moment(
    start_position: float,
    shape: float,
    rate: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def gamma_fpt_central_moment(
    start_position: float,
    shape: float,
    rate: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def gamma_occupation_time(
    start_position: float,
    shape: float,
    rate: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def gamma_occupation_time_raw_moment(
    start_position: float,
    shape: float,
    rate: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def gamma_occupation_time_central_moment(
    start_position: float,
    shape: float,
    rate: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def gamma_tamsd(
    start_position: float,
    shape: float,
    rate: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def gamma_eatamsd(
    start_position: float,
    shape: float,
    rate: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Geometric Brownian Motion
def gb_simulate(
    start_value: float,
    mu: float,
    sigma: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def gb_raw_moment(
    start_value: float,
    mu: float,
    sigma: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def gb_central_moment(
    start_value: float,
    mu: float,
    sigma: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def gb_fpt(
    start_value: float,
    mu: float,
    sigma: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def gb_fpt_raw_moment(
    start_value: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def gb_fpt_central_moment(
    start_value: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def gb_occupation_time(
    start_value: float,
    mu: float,
    sigma: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def gb_occupation_time_raw_moment(
    start_value: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def gb_occupation_time_central_moment(
    start_value: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def gb_tamsd(
    start_value: float,
    mu: float,
    sigma: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def gb_eatamsd(
    start_value: float,
    mu: float,
    sigma: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Levy Walk
def levy_walk_simulate(
    start_position: float,
    alpha: float,  # Parameter for jump length distribution
    beta: float,  # Parameter for waiting time distribution or velocity
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def levy_walk_raw_moment(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def levy_walk_central_moment(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def levy_walk_fpt(
    start_position: float,
    alpha: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def levy_walk_fpt_raw_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def levy_walk_fpt_central_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def levy_walk_occupation_time(
    start_position: float,
    alpha: float,
    beta: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def levy_walk_occupation_time_raw_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def levy_walk_occupation_time_central_moment(
    start_position: float,
    alpha: float,
    beta: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def levy_walk_tamsd(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def levy_walk_eatamsd(
    start_position: float,
    alpha: float,
    beta: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...

# Ornstein-Uhlenbeck Process
def ou_simulate(
    start_position: float,
    theta: float,  # Mean reversion rate
    mu: float,  # Long-term mean
    sigma: float,  # Volatility
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def ou_raw_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def ou_central_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def ou_fpt(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float | None: ...
def ou_fpt_raw_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def ou_fpt_central_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    max_duration: float,
) -> float | None: ...
def ou_occupation_time(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def ou_occupation_time_raw_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def ou_occupation_time_central_moment(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    domain: tuple[float, float],
    order: int,
    particles: int,
    step_size: float,
    duration: float,
) -> float: ...
def ou_tamsd(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    duration: float,
    delta: float,
    step_size: float,
    quad_order: int,
) -> float: ...
def ou_eatamsd(
    start_position: float,
    theta: float,
    mu: float,
    sigma: float,
    duration: float,
    delta: float,
    particles: int,
    step_size: float,
    quad_order: int,
) -> float: ...
