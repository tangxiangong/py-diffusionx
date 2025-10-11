use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::Gamma, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn gamma_simulate(
    py: Python<'_>,
    shape: f64,
    rate: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let gamma = Gamma::new(shape, rate)?;
    let (times, positions) = gamma.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn gamma_raw_moment(
    shape: f64,
    rate: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_central_moment(
    shape: f64,
    rate: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_fpt(
    shape: f64,
    rate: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_fpt_raw_moment(
    shape: f64,
    rate: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gamma = Gamma::new(shape, rate)?;
    let fpt = FirstPassageTime::new(&gamma, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_fpt_central_moment(
    shape: f64,
    rate: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let gamma = Gamma::new(shape, rate)?;
    let fpt = FirstPassageTime::new(&gamma, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_occupation_time(
    shape: f64,
    rate: f64,
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_occupation_time_raw_moment(
    shape: f64,
    rate: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let oc = OccupationTime::new(&gamma, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_occupation_time_central_moment(
    shape: f64,
    rate: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let oc = OccupationTime::new(&gamma, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_tamsd(
    shape: f64,
    rate: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn gamma_eatamsd(
    shape: f64,
    rate: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let gamma = Gamma::new(shape, rate)?;
    let result = gamma.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
