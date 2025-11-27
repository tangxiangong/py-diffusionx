use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{continuous::Bm, prelude::*};
use pyo3::prelude::*;
#[cfg(feature = "stub_gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_simulate(
    py: Python<'_>,
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let (times, positions) = bm.simulate(duration, step_size)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_raw_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_central_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_fpt(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_fpt_raw_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let fpt = FirstPassageTime::new(&bm, domain)?;
    let result = fpt.raw_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_fpt_central_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let fpt = FirstPassageTime::new(&bm, domain)?;
    let result = fpt.central_moment(order, particles, max_duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_occupation_time(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_occupation_time_raw_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let oc = OccupationTime::new(&bm, domain, duration)?;
    let result = oc.raw_moment(order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_occupation_time_central_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    domain: (f64, f64),
    order: i32,
    particles: usize,
    step_size: f64,
    duration: f64,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let oc = OccupationTime::new(&bm, domain, duration)?;
    let result = oc.central_moment(order, particles, step_size)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_tamsd(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    delta: f64,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.tamsd(duration, delta, step_size, quad_order)?;
    Ok(result)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn bm_eatamsd(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    delta: f64,
    particles: usize,
    step_size: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.eatamsd(duration, delta, particles, step_size, quad_order)?;
    Ok(result)
}
