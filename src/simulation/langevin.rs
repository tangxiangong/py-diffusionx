use crate::{
    XPyResult,
    simulation::{PyArrayPair, call_py_func, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{GeneralizedLangevin, Langevin, SubordinatedLangevin},
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

#[pyfunction]
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
pub fn langevin_fpt(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };

    let result = langevin.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_fpt_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_fpt_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_occupation_time(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let result = langevin.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_occupation_time_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_occupation_time_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_tamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let result = langevin.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn langevin_eatamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        Langevin::new(drift, diffusion, start_position)?
    };
    let result = langevin.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
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
pub fn generalized_langevin_fpt(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_fpt_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_fpt_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_occupation_time(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_occupation_time_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_occupation_time_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_tamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn generalized_langevin_eatamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        GeneralizedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_simulate(
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

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let (times, positions) = langevin.simulate(duration, time_step)?;

    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn subordinated_langevin_raw_moment(
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

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.raw_moment(duration, order, particles, time_step)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_central_moment(
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

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.central_moment(duration, order, particles, time_step)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_fpt(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };

    let result = langevin.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_fpt_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_fpt_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let fpt = FirstPassageTime::new(&langevin, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_occupation_time(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_occupation_time_raw_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_occupation_time_central_moment(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let oc = OccupationTime::new(&langevin, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_tamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinated_langevin_eatamsd(
    drift_func: PyObject,
    diffusion_func: PyObject,
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let langevin = {
        let drift = |x: f64, t: f64| -> f64 { call_py_func(&drift_func, (x, t)) };

        let diffusion = |x: f64, t: f64| -> f64 { call_py_func(&diffusion_func, (x, t)) };

        SubordinatedLangevin::new(drift, diffusion, start_position, alpha)?
    };
    let result = langevin.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
