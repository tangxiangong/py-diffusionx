use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{jump::Poisson, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn poisson_simulate_duration(
    py: Python,
    lambda: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_duration(duration)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn poisson_simulate_step(
    py: Python,
    lambda: f64,
    num_step: usize,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_step(num_step)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn poisson_raw_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_central_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.central_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_fpt(lambda: f64, domain: (f64, f64), max_duration: f64) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.fpt(domain, max_duration)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_occupation_time(lambda: f64, domain: (f64, f64), duration: f64) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.occupation_time(domain, duration)?;
    Ok(result)
}
