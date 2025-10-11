use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{AsymmetricCauchy, Cauchy},
    prelude::*,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn cauchy_simulate(
    py: Python<'_>,
    start_position: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let cauchy = Cauchy::new(start_position);
    let (times, positions) = cauchy.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn cauchy_raw_moment(
    start_position: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_central_moment(
    start_position: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_fpt(
    start_position: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_fpt_raw_moment(
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = Cauchy::new(start_position);
    let fpt = FirstPassageTime::new(&cauchy, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_fpt_central_moment(
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = Cauchy::new(start_position);
    let fpt = FirstPassageTime::new(&cauchy, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_occupation_time(
    start_position: f64,
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_occupation_time_raw_moment(
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let oc = OccupationTime::new(&cauchy, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_occupation_time_central_moment(
    start_position: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let oc = OccupationTime::new(&cauchy, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_tamsd(
    start_position: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn cauchy_eatamsd(
    start_position: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let cauchy = Cauchy::new(start_position);
    let result = cauchy.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_simulate(
    py: Python<'_>,
    start_position: f64,
    beta: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let (times, positions) = cauchy.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn asymmetric_cauchy_raw_moment(
    start_position: f64,
    beta: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_central_moment(
    start_position: f64,
    beta: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_fpt(
    start_position: f64,
    beta: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_fpt_raw_moment(
    start_position: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let fpt = FirstPassageTime::new(&cauchy, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_fpt_central_moment(
    start_position: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let fpt = FirstPassageTime::new(&cauchy, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_occupation_time(
    start_position: f64,
    beta: f64,
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_occupation_time_raw_moment(
    start_position: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let oc = OccupationTime::new(&cauchy, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_occupation_time_central_moment(
    start_position: f64,
    beta: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let oc = OccupationTime::new(&cauchy, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_tamsd(
    start_position: f64,
    beta: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn asymmetric_cauchy_eatamsd(
    start_position: f64,
    beta: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let cauchy = AsymmetricCauchy::new(start_position, beta)?;
    let result = cauchy.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
