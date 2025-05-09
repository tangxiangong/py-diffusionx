use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{
    continuous::{InvSubordinator, Subordinator},
    prelude::*,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let subordinator = Subordinator::new(alpha)?;
    let (times, positions) = subordinator.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

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

#[pyfunction]
pub fn inv_subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let (times, positions) = inv_subordinator.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

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
