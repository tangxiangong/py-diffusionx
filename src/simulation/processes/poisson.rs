use crate::{
    XPyResult,
    simulation::{PyArrayPair, vec_to_pyarray},
};
use diffusionx::simulation::{point::Poisson, prelude::*};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_simulate_duration(
    py: Python<'_>,
    lambda_: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda_)?;
    let (times, positions) = poisson.simulate_with_duration(duration)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_simulate_step(
    py: Python<'_>,
    lambda_: f64,
    num_step: usize,
) -> XPyResult<PyArrayPair<'_>> {
    let poisson = Poisson::new(lambda_)?;
    let (times, positions) = poisson.simulate_with_step(num_step)?;
    Ok(vec_to_pyarray(py, times, positions))
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_raw_moment(
    lambda_: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda_)?;
    let result = poisson.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_central_moment(
    lambda_: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda_)?;
    let result = poisson.central_moment(duration, order, particles)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_fpt(lambda_: f64, domain: (f64, f64), max_duration: f64) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda_)?;
    let result = poisson.fpt(domain, max_duration)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_fpt_raw_moment(
    lambda_: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda_)?;
    let fpt = FirstPassageTime::new(&poisson, domain)?;
    let result = fpt.raw_moment_p(order, particles, max_duration)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_fpt_central_moment(
    lambda_: f64,
    domain: (f64, f64),
    max_duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda_)?;
    let fpt = FirstPassageTime::new(&poisson, domain)?;
    let result = fpt.central_moment_p(order, particles, max_duration)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_occupation_time(lambda_: f64, domain: (f64, f64), duration: f64) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda_)?;
    let result = poisson.occupation_time(domain, duration)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_occupation_time_raw_moment(
    lambda_: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda_)?;
    let oc = OccupationTime::new(&poisson, domain, duration)?;
    let result = oc.raw_moment_p(order, particles)?;
    Ok(result)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn poisson_occupation_time_central_moment(
    lambda_: f64,
    domain: (f64, f64),
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda_)?;
    let oc = OccupationTime::new(&poisson, domain, duration)?;
    let result = oc.central_moment_p(order, particles)?;
    Ok(result)
}
