use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{AsymmetricLevy, Levy},
    prelude::*,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn levy_simulate(
    py: Python<'_>,
    start_position: f64,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let levy = Levy::new(start_position, alpha)?;
    let (times, positions) = levy.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
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
pub fn levy_fpt_raw_moment(
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let levy = Levy::new(start_position, alpha)?;
    let fpt = FirstPassageTime::new(&levy, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_fpt_central_moment(
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let levy = Levy::new(start_position, alpha)?;
    let fpt = FirstPassageTime::new(&levy, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
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
pub fn levy_occupation_time_raw_moment(
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let oc = OccupationTime::new(&levy, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_occupation_time_central_moment(
    start_position: f64,
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let oc = OccupationTime::new(&levy, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_tamsd(
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_eatamsd(
    start_position: f64,
    alpha: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_simulate(
    py: Python<'_>,
    start_position: f64,
    alpha: f64,
    beta: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let (times, positions) = levy.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn asymmetric_levy_fpt(
    start_position: f64,
    alpha: f64,
    beta: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let result = levy.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_fpt_raw_moment(
    start_position: f64,
    alpha: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let fpt = FirstPassageTime::new(&levy, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_fpt_central_moment(
    start_position: f64,
    alpha: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let fpt = FirstPassageTime::new(&levy, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_occupation_time(
    start_position: f64,
    alpha: f64,
    beta: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let result = levy.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_occupation_time_raw_moment(
    start_position: f64,
    alpha: f64,
    beta: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let oc = OccupationTime::new(&levy, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_occupation_time_central_moment(
    start_position: f64,
    alpha: f64,
    beta: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
    step_size: f64,
) -> XPyResult<f64> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let oc = OccupationTime::new(&levy, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_tamsd(
    start_position: f64,
    alpha: f64,
    beta: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let result = levy.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_levy_eatamsd(
    start_position: f64,
    alpha: f64,
    beta: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let levy = AsymmetricLevy::new(start_position, alpha, beta)?;
    let result = levy.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
