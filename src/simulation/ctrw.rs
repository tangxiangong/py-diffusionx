use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{jump::CTRW, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn ctrw_simulate_duration(
    py: Python,
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let (times, positions) = ctrw.simulate_with_duration(duration)?;
    Ok(vec_to_pyarray(py, times, positions))
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
    Ok(vec_to_pyarray(py, times, positions))
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
