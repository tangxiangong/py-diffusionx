use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::LevyWalk, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn levy_walk_simulate(
    py: Python<'_>,
    alpha: f64,
    velocity: f64,
    start_position: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
    let (times, positions) = levy_walk.simulate_with_duration(duration)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[pyfunction]
pub fn levy_walk_raw_moment(
    alpha: f64,
    velocity: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
    let result = levy_walk.raw_moment(duration, order, particles, 0.1)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_walk_central_moment(
    alpha: f64,
    velocity: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
    let result = levy_walk.central_moment(duration, order, particles, 0.1)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_walk_fpt(
    alpha: f64,
    velocity: f64,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
    let result = levy_walk.fpt(domain, max_duration, 0.1)?;
    Ok(result)
}
