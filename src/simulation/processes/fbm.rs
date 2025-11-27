use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::FBm, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn fbm_simulate(
    py: Python<'_>,
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
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
    let fbm = FBm::new(start_position, hurst_exponent)?;
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
    let fbm = FBm::new(start_position, hurst_exponent)?;
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
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let result = fbm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_fpt_raw_moment(
    start_position: f64,
    hurst_exponent: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let fpt = FirstPassageTime::new(&fbm, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_fpt_central_moment(
    start_position: f64,
    hurst_exponent: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let fpt = FirstPassageTime::new(&fbm, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
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
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let result = fbm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_occupation_time_raw_moment(
    start_position: f64,
    hurst_exponent: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let oc = OccupationTime::new(&fbm, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_occupation_time_central_moment(
    start_position: f64,
    hurst_exponent: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let oc = OccupationTime::new(&fbm, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_tamsd(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let result = fbm.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_eatamsd(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let fbm = FBm::new(start_position, hurst_exponent)?;
    let result = fbm.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
