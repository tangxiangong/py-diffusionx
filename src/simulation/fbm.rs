use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::Fbm, prelude::*};
use pyo3::prelude::*;

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
    Ok(vec_to_pyarray(py, times, positions))
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
