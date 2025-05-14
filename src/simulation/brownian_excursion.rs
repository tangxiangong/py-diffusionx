use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn be_simulate(py: Python, duration: f64, step_size: f64) -> XPyResult<PyArrayPair<'_>> {
    let be = BrownianExcursion;
    let (times, positions) = be.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn be_raw_moment(
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let result = be.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_central_moment(
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let result = be.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_fpt(step_size: f64, domain: (f64, f64)) -> XPyResult<Option<f64>> {
    let be = BrownianExcursion;
    let result = be.fpt(domain, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_fpt_raw_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let be = BrownianExcursion;
    let fpt = FirstPassageTime::new(&be, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_fpt_central_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let be = BrownianExcursion;
    let fpt = FirstPassageTime::new(&be, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_occupation_time(domain: (f64, f64), step_size: f64, duration: f64) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let result = be.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_occupation_time_raw_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let oc = OccupationTime::new(&be, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_occupation_time_central_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let oc = OccupationTime::new(&be, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn be_tamsd(duration: f64, delta: f64, step_size: f64, quad_order: usize) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let result = be.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[pyfunction]
pub fn be_eatamsd(
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let be = BrownianExcursion;
    let result = be.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
