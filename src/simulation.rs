#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::XPyResult;
use diffusionx::simulation::{
    continuous::{Bm, Fbm, GeneralizedLangevin, InvSubordinator, Langevin, Levy, Subordinator},
    jump::{CTRW, Poisson},
    prelude::*,
};
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::prelude::*;

/// 封装从Python调用函数的辅助方法，处理错误情况
fn call_py_func(func: &PyObject, args: (f64, f64)) -> f64 {
    Python::with_gil(|py| {
        func.call1(py, args)
            .and_then(|result| result.extract::<f64>(py))
            .unwrap_or(0.0)
    })
}

type PyArrayPair<'py> = (Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix1>>);

/// 封装Langevin计算所需参数，同时支持模拟和矩计算
#[pyclass]
struct LangevinParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl LangevinParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

/// 封装GeneralizedLangevin计算所需参数
#[pyclass]
struct GeneralizedLangevinParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl GeneralizedLangevinParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        alpha: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

/// 封装SubordinatedLangevin矩计算所需参数
#[pyclass]
struct SubordinatedLangevinParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl SubordinatedLangevinParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

/// 封装Langevin矩计算所需参数
#[pyclass]
struct LangevinMomentParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl LangevinMomentParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

/// 封装GeneralizedLangevin矩计算所需参数
#[pyclass]
struct GeneralizedLangevinMomentParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl GeneralizedLangevinMomentParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        alpha: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

/// 封装SubordinatedLangevin矩计算所需参数
#[pyclass]
struct SubordinatedLangevinMomentParams {
    #[pyo3(get, set)]
    drift_func: PyObject,
    #[pyo3(get, set)]
    diffusion_func: PyObject,
    #[pyo3(get, set)]
    start_position: f64,
    #[pyo3(get, set)]
    duration: f64,
    #[pyo3(get, set)]
    order: i32,
    #[pyo3(get, set)]
    particles: usize,
    #[pyo3(get, set)]
    time_step: f64,
}

#[pymethods]
impl SubordinatedLangevinMomentParams {
    #[new]
    fn new(
        drift_func: PyObject,
        diffusion_func: PyObject,
        start_position: f64,
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> Self {
        Self {
            drift_func,
            diffusion_func,
            start_position,
            duration,
            order,
            particles,
            time_step,
        }
    }
}

#[pyfunction]
pub fn bm_simulate(
    py: Python,
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let (times, positions) = bm.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn bm_raw_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn bm_central_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn bm_fpt(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn bm_occupation_time(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_simulate(
    py: Python,
    start_position: f64,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let levy = Levy::new(start_position, alpha)?;
    let (times, positions) = levy.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn levy_fpt(
    start_position: f64,
    alpha: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_occupation_time(
    start_position: f64,
    alpha: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_simulate_duration(
    py: Python,
    lambda: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_duration(duration)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn poisson_simulate_step(
    py: Python,
    lambda: f64,
    num_step: usize,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_step(num_step)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn poisson_raw_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_central_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.central_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_fpt(lambda: f64, domain: (f64, f64), max_duration: f64) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.fpt(domain, max_duration)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_occupation_time(lambda: f64, domain: (f64, f64), duration: f64) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.occupation_time(domain, duration)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let subordinator = Subordinator::new(alpha)?;
    let (times, positions) = subordinator.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn inv_subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let (times, positions) = inv_subordinator.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn inv_subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn inv_subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_simulate(
    py: Python,
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let (times, positions) = fbm.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn fbm_raw_moment(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_central_moment(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_fpt(
    start_position: f64,
    hurst_exponent: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_occupation_time(
    start_position: f64,
    hurst_exponent: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_simulate_duration(
    py: Python,
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let (times, positions) = ctrw.simulate(duration)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn ctrw_simulate_step(
    py: Python,
    alpha: f64,
    beta: f64,
    start_position: f64,
    num_step: usize,
) -> XPyResult<PyArrayPair<'_>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let (times, positions) = ctrw.simulate_with_step(num_step)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn ctrw_raw_moment(
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_central_moment(
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.central_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_fpt(
    alpha: f64,
    beta: f64,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.fpt(domain, max_duration)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_occupation_time(
    alpha: f64,
    beta: f64,
    start_position: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.occupation_time(domain, duration)?;
    Ok(result)
}

/// Py function wrapper for Langevin simulation
#[pyfunction]
pub fn langevin_simulate(
    py: Python,
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XPyResult<PyArrayPair<'_>> {
    // 创建参数结构体
    let params = LangevinParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order: 0,     // 不使用
        particles: 0, // 不使用
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, params.start_position)?
    };

    // 模拟过程
    let (times, positions) = langevin.simulate(params.duration, params.time_step)?;

    // 转换为 Python 数组
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);

    Ok((times_array, positions_array))
}

/// Py function wrapper for GeneralizedLangevin simulation
#[pyfunction]
pub fn generalized_langevin_simulate(
    py: Python,
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    time_step: f64,
) -> XPyResult<PyArrayPair<'_>> {
    // 创建参数结构体
    let params = GeneralizedLangevinParams {
        drift_func,
        diffusion_func,
        start_position,
        alpha,
        duration,
        order: 0,     // 不使用
        particles: 0, // 不使用
        time_step,
    };

    // 创建 GeneralizedLangevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, params.start_position, params.alpha)?
    };

    // 模拟过程
    let (times, positions) = langevin.simulate(params.duration, params.time_step)?;

    // 转换为 Python 数组
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);

    Ok((times_array, positions_array))
}

/// Py function wrapper for GeneralizedLangevin raw moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, alpha, duration, order, particles, time_step))]
pub fn generalized_langevin_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = GeneralizedLangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        alpha,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 GeneralizedLangevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, params.start_position, params.alpha)?
    };

    let result = langevin.raw_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}

/// Py function wrapper for GeneralizedLangevin central moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, alpha, duration, order, particles, time_step))]
pub fn generalized_langevin_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = GeneralizedLangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        alpha,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 GeneralizedLangevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, params.start_position, params.alpha)?
    };

    let result = langevin.central_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}

/// Py function wrapper for Langevin raw moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, duration, order, particles, time_step))]
pub fn langevin_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = LangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, params.start_position)?
    };

    let result = langevin.raw_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}

/// Py function wrapper for Langevin central moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, duration, order, particles, time_step))]
pub fn langevin_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = LangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, params.start_position)?
    };

    let result = langevin.central_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}

/// Py function wrapper for SubordinatedLangevin simulation
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, duration, time_step))]
pub fn subordinated_langevin_simulate(
    py: Python,
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XPyResult<PyArrayPair<'_>> {
    // 创建参数结构体
    let params = SubordinatedLangevinParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order: 0,     // 不使用
        particles: 0, // 不使用
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        // 创建 SubordinatedLangevin 实例
        Langevin::new(drift, diffusion, params.start_position)?
    };

    // 模拟过程
    let (times, positions) = langevin.simulate(params.duration, params.time_step)?;

    // 转换为 Python 数组
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);

    Ok((times_array, positions_array))
}

/// Py function wrapper for SubordinatedLangevin raw moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, duration, order, particles, time_step))]
pub fn subordinated_langevin_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = SubordinatedLangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        // 创建 SubordinatedLangevin 实例
        Langevin::new(drift, diffusion, params.start_position)?
    };

    let result = langevin.raw_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}

/// Py function wrapper for SubordinatedLangevin central moment
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, start_position, duration, order, particles, time_step))]
pub fn subordinated_langevin_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
    time_step: f64,
) -> XPyResult<f64> {
    // 创建参数结构体
    let params = SubordinatedLangevinMomentParams {
        drift_func,
        diffusion_func,
        start_position,
        duration,
        order,
        particles,
        time_step,
    };

    // 创建 Langevin 实例
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&params.drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&params.diffusion_func, (x, t)) };

        // 创建 SubordinatedLangevin 实例
        Langevin::new(drift, diffusion, params.start_position)?
    };

    let result = langevin.central_moment(
        params.duration,
        params.order,
        params.particles,
        params.time_step,
    )?;
    Ok(result)
}
