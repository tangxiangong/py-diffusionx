use crate::{
    XPyResult,
    simulation::{PyArrayPair, call_py_func, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{GeneralizedLangevin, Langevin},
    prelude::*,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn langevin_simulate(
    py: Python,
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let (times, positions) = langevin.simulate(duration, time_step)?;

    Ok(vec_to_pyarray(py, times, positions))
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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let (times, positions) = langevin.simulate(duration, time_step)?;

    Ok(vec_to_pyarray(py, times, positions))
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.raw_moment(duration, order, particles, time_step)?;
    Ok(result)
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.central_moment(duration, order, particles, time_step)?;
    Ok(result)
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let result = langevin.raw_moment(duration, order, particles, time_step)?;
    Ok(result)
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let result = langevin.central_moment(duration, order, particles, time_step)?;
    Ok(result)
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let (times, positions) = langevin.simulate(duration, time_step)?;

    Ok(vec_to_pyarray(py, times, positions))
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let result = langevin.raw_moment(duration, order, particles, time_step)?;
    Ok(result)
}

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
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let result = langevin.central_moment(duration, order, particles, time_step)?;
    Ok(result)
}
