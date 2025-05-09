use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::Levy, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn levy_simulate(
    py: Python,
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
