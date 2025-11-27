use crate::{XPyResult, simulation::PyArrayPair};
use diffusionx::{XResult, simulation::prelude::*};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PyContinuousProcessWrapper {
    py_callable: Arc<Py<PyAny>>,
}

impl PyContinuousProcessWrapper {
    pub fn new(py_callable: Py<PyAny>) -> Self {
        Self {
            py_callable: Arc::new(py_callable),
        }
    }
}

impl ContinuousProcess for PyContinuousProcessWrapper {
    fn start(&self) -> f64 {
        Python::attach(|py| {
            self.py_callable
                .bind(py)
                .call_method0("start")
                .expect("Failed to call Python start method")
                .extract()
                .expect("Failed to extract start result")
        })
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
        let (time, position) = Python::attach(|py| {
            let (t, x) = self
                .py_callable
                .call_method1(py, "simulate", (duration, time_step))
                .and_then(|result| result.extract::<PyArrayPair<'_>>(py))
                .expect("Failed to call Python simulate method");

            let time = t.to_vec().expect("Failed to convert time array to Vec");
            let position = x.to_vec().expect("Failed to convert position array to Vec");

            (time, position)
        });

        Ok((time, position))
    }
}
unsafe impl Send for PyContinuousProcessWrapper {}
unsafe impl Sync for PyContinuousProcessWrapper {}

#[pyfunction]
pub fn moment(
    process: Bound<'_, PyAny>,
    central: bool,
    order: i32,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(if central {
        py_wrapper.central_moment(duration, order, particles, time_step)?
    } else {
        py_wrapper.raw_moment(duration, order, particles, time_step)?
    })
}

#[pyfunction]
pub fn mean(
    process: Bound<'_, PyAny>,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.mean(duration, particles, time_step)?)
}

#[pyfunction]
pub fn msd(
    process: Bound<'_, PyAny>,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.msd(duration, particles, time_step)?)
}

#[pyfunction]
pub fn fpt(
    process: Bound<'_, PyAny>,
    domain: (f64, f64),
    max_duration: f64,
    time_step: f64,
) -> XPyResult<Option<f64>> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.fpt(domain, max_duration, time_step)?)
}

#[pyfunction]
pub fn occupation_time(
    process: Bound<'_, PyAny>,
    domain: (f64, f64),
    duration: f64,
    time_step: f64,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.occupation_time(domain, duration, time_step)?)
}

#[pyfunction]
pub fn tamsd(
    process: Bound<'_, PyAny>,
    duration: f64,
    delta: f64,
    time_step: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.tamsd(duration, delta, time_step, quad_order)?)
}

#[pyfunction]
pub fn eatamsd(
    process: Bound<'_, PyAny>,
    duration: f64,
    delta: f64,
    particles: usize,
    time_step: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    let py_wrapper = PyContinuousProcessWrapper::new(process.unbind());
    Ok(py_wrapper.eatamsd(duration, delta, particles, time_step, quad_order)?)
}
