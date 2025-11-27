use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{InvSubordinator, Subordinator},
    prelude::*,
};
use pyo3::prelude::*;
#[cfg(feature = "stub_gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_simulate(
    py: Python<'_>,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let subordinator = Subordinator::new(alpha)?;
    let (times, positions) = subordinator.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_fpt_raw_moment(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let subordinator = Subordinator::new(alpha)?;
    let fpt = FirstPassageTime::new(&subordinator, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_fpt_central_moment(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let subordinator = Subordinator::new(alpha)?;
    let fpt = FirstPassageTime::new(&subordinator, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_occupation_time_raw_moment(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let subordinator = Subordinator::new(alpha)?;
    let oc = OccupationTime::new(&subordinator, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn subordinator_occupation_time_central_moment(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let subordinator = Subordinator::new(alpha)?;
    let oc = OccupationTime::new(&subordinator, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_simulate(
    py: Python<'_>,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let (times, positions) = inv_subordinator.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_raw_moment(
    alpha: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_central_moment(
    alpha: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_fpt_raw_moment(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let fpt = FirstPassageTime::new(&inv_subordinator, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_fpt_central_moment(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let fpt = FirstPassageTime::new(&inv_subordinator, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_occupation_time_raw_moment(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let oc = OccupationTime::new(&inv_subordinator, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn inv_subordinator_occupation_time_central_moment(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let oc = OccupationTime::new(&inv_subordinator, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}
