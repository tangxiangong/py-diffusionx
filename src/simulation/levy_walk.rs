use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{point::LevyWalk, prelude::*};
use pyo3::prelude::*;

#[pyfunction]
pub fn levy_walk_simulate(
    py: Python,
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
    let result = levy_walk.raw_moment(duration, order, particles)?;
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
    let result = levy_walk.central_moment(duration, order, particles)?;
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
    let result = levy_walk.fpt(domain, max_duration)?;
    Ok(result)
}

// #[pyfunction]
// pub fn levy_walk_fpt_raw_moment(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     domain: (f64, f64),
//     order: i32,
//     particles: usize,
//     max_duration: f64,
// ) -> XPyResult<Option<f64>> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let fpt = FirstPassageTime::new(&levy_walk, domain)?;
//     let result = fpt.raw_moment(order, particles, max_duration)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_fpt_central_moment(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     domain: (f64, f64),
//     order: i32,
//     particles: usize,
//     step_size: f64,
//     max_duration: f64,
// ) -> XPyResult<Option<f64>> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let fpt = FirstPassageTime::new(&levy_walk, domain)?;
//     let result = fpt.central_moment(order, particles, max_duration, step_size)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_occupation_time(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     domain: (f64, f64),
//     step_size: f64,
//     duration: f64,
// ) -> XPyResult<f64> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let result = levy_walk.occupation_time(domain, duration, step_size)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_occupation_time_raw_moment(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     domain: (f64, f64),
//     order: i32,
//     particles: usize,
//     step_size: f64,
//     duration: f64,
// ) -> XPyResult<f64> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let oc = OccupationTime::new(&levy_walk, domain, duration)?;
//     let result = oc.raw_moment(order, particles, step_size)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_occupation_time_central_moment(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     domain: (f64, f64),
//     order: i32,
//     particles: usize,
//     step_size: f64,
//     duration: f64,
// ) -> XPyResult<f64> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let oc = OccupationTime::new(&levy_walk, domain, duration)?;
//     let result = oc.central_moment(order, particles, step_size)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_tamsd(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     delta: f64,
//     duration: f64,
//     step_size: f64,
//     quad_order: usize,
// ) -> XPyResult<f64> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let result = levy_walk.tamsd(duration, delta, step_size, quad_order)?;
//     Ok(result)
// }

// #[pyfunction]
// pub fn levy_walk_eatamsd(
//     alpha: f64,
//     velocity: f64,
//     start_position: f64,
//     duration: f64,
//     delta: f64,
//     particles: usize,
//     step_size: f64,
//     quad_order: usize,
// ) -> XPyResult<f64> {
//     let levy_walk = LevyWalk::new(alpha, velocity, start_position)?;
//     let result = levy_walk.eatamsd(duration, delta, particles, step_size, quad_order)?;
//     Ok(result)
// }
