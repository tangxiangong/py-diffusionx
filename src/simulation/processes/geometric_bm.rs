use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::GeometricBm, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn gb_simulate(
    py: Python<'_>,
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let (times, positions) = gb.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn gb_raw_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_central_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_fpt(
    start_position: f64,
    mu: f64,
    sigma: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_fpt_raw_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let fpt = FirstPassageTime::new(&gb, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_fpt_central_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let fpt = FirstPassageTime::new(&gb, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_occupation_time(
    start_position: f64,
    mu: f64,
    sigma: f64,
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_occupation_time_raw_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let oc = OccupationTime::new(&gb, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_occupation_time_central_moment(
    start_position: f64,
    mu: f64,
    sigma: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let oc = OccupationTime::new(&gb, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_tamsd(
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn gb_eatamsd(
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let gb = GeometricBm::new(start_position, mu, sigma)?;
    let result = gb.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
