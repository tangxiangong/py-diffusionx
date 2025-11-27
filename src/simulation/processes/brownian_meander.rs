use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_simulate(
    py: Python<'_>,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let bm = BrownianMeander;
    let (times, positions) = bm.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_raw_moment(
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let result = bm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_central_moment(
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let result = bm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_fpt(step_size: f64, domain: (f64, f64)) -> XPyResult<Option<f64>> {
    let bm = BrownianMeander;
    let result = bm.fpt(domain, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_fpt_raw_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = BrownianMeander;
    let fpt = FirstPassageTime::new(&bm, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_fpt_central_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = BrownianMeander;
    let fpt = FirstPassageTime::new(&bm, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_occupation_time(
    domain: (f64, f64),
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let result = bm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_occupation_time_raw_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let oc = OccupationTime::new(&bm, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_occupation_time_central_moment(
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let oc = OccupationTime::new(&bm, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_tamsd(
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let result = bm.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn meander_eatamsd(
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let bm = BrownianMeander;
    let result = bm.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
