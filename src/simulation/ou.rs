use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::OrnsteinUhlenbeck, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn ou_simulate(
    py: Python,
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let (times, positions) = ou.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn ou_raw_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_central_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_fpt(
    theta: f64,
    sigma: f64,
    start_position: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_fpt_raw_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let fpt = FirstPassageTime::new(&ou, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_fpt_central_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let fpt = FirstPassageTime::new(&ou, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_occupation_time(
    theta: f64,
    sigma: f64,
    start_position: f64,
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_occupation_time_raw_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let oc = OccupationTime::new(&ou, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_occupation_time_central_moment(
    theta: f64,
    sigma: f64,
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let oc = OccupationTime::new(&ou, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_tamsd(
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn ou_eatamsd(
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let ou = OrnsteinUhlenbeck::new(theta, sigma, start_position)?;
    let result = ou.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
