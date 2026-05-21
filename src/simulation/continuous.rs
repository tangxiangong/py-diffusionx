use crate::{XPyError, XPyResult, simulation::PyArrayPair};
use gauss_quad::GaussLegendre;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
#[cfg(feature = "stub_gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;
use std::{num::NonZero, sync::Arc};

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn moment(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    central: bool,
    order: i32,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    if central {
        central_moment(py, simulate_fn, order, duration, time_step, particles)
    } else {
        raw_moment(py, simulate_fn, order, duration, time_step, particles)
    }
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn mean(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let values: XPyResult<Vec<f64>> = (0..particles)
        .into_par_iter()
        .map(|_| endpoint(&simulate, duration, time_step))
        .collect();

    Ok(values?.into_iter().sum::<f64>() / particles as f64)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn msd(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let values: XPyResult<Vec<f64>> = (0..particles)
        .into_par_iter()
        .map(|_| {
            let x_vec = simulate_positions(&simulate, duration, time_step)?;
            let start = *x_vec
                .first()
                .ok_or_else(|| value_error("simulate returned no positions"))?;
            let end = *x_vec
                .last()
                .ok_or_else(|| value_error("simulate returned no positions"))?;
            Ok((end - start) * (end - start))
        })
        .collect();

    Ok(values?.into_iter().sum::<f64>() / particles as f64)
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn tamsd(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    duration: f64,
    delta: f64,
    time_step: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    validate_tamsd_args(duration, delta, time_step, quad_order)?;
    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let legendre_quad = GaussLegendre::new(
        NonZero::new(quad_order).ok_or_else(|| value_error("quad_order must be positive"))?,
    );
    let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
    let nodes_weights = nodes_weights_transform(0.0, duration - delta, &nodes_weights_pairs);
    let values: XPyResult<Vec<f64>> = nodes_weights
        .into_par_iter()
        .map(|(node, weight)| {
            lagged_square_displacement(&simulate, node + delta, delta, time_step)
                .map(|value| value * weight)
        })
        .collect();

    Ok(values?.into_iter().sum::<f64>() / (duration - delta))
}

#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn eatamsd(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    duration: f64,
    delta: f64,
    particles: usize,
    time_step: f64,
    quad_order: usize,
) -> XPyResult<f64> {
    validate_tamsd_args(duration, delta, time_step, quad_order)?;
    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let values: XPyResult<Vec<f64>> = (0..particles)
        .into_par_iter()
        .map(|_| {
            let legendre_quad = GaussLegendre::new(
                NonZero::new(quad_order)
                    .ok_or_else(|| value_error("quad_order must be positive"))?,
            );
            let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
            let nodes_weights =
                nodes_weights_transform(0.0, duration - delta, &nodes_weights_pairs);
            let values: XPyResult<Vec<f64>> = nodes_weights
                .into_par_iter()
                .map(|(node, weight)| {
                    lagged_square_displacement(&simulate, node + delta, delta, time_step)
                        .map(|value| value * weight)
                })
                .collect();
            Ok(values?.into_iter().sum::<f64>() / (duration - delta))
        })
        .collect();

    Ok(values?.into_iter().sum::<f64>() / particles as f64)
}

fn raw_moment(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    order: i32,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let values: XPyResult<Vec<f64>> = (0..particles)
        .into_par_iter()
        .map(|_| {
            let end = endpoint(&simulate, duration, time_step)?;
            Ok(if order == 1 { end } else { end.powi(order) })
        })
        .collect();

    Ok(values?.into_iter().sum::<f64>() / particles as f64)
}

fn central_moment(
    py: Python<'_>,
    simulate_fn: Py<PyAny>,
    order: i32,
    duration: f64,
    time_step: f64,
    particles: usize,
) -> XPyResult<f64> {
    let mean = raw_moment(
        py,
        simulate_fn.clone_ref(py),
        1,
        duration,
        time_step,
        particles,
    )?;

    let simulate = Arc::new(simulate_fn.clone_ref(py));

    let values: XPyResult<Vec<f64>> = (0..particles)
        .into_par_iter()
        .map(|_| {
            let end = endpoint(&simulate, duration, time_step)?;
            Ok(if order == 1 {
                end - mean
            } else {
                (end - mean).powi(order)
            })
        })
        .collect();

    Ok(values?.into_iter().sum::<f64>() / particles as f64)
}

fn simulate_positions(simulate: &Py<PyAny>, duration: f64, time_step: f64) -> XPyResult<Vec<f64>> {
    Python::attach(|py| {
        let (_, x) = simulate
            .call_method1(py, "simulate", (duration, time_step))
            .map_err(|error| value_error(format!("Failed to call simulate method: {error}")))?
            .extract::<PyArrayPair<'_>>(py)
            .map_err(|error| value_error(format!("Failed to extract simulate result: {error}")))?;

        x.to_vec().map_err(|error| {
            value_error(format!("Failed to convert position array to Vec: {error}"))
        })
    })
}

fn endpoint(simulate: &Py<PyAny>, duration: f64, time_step: f64) -> XPyResult<f64> {
    simulate_positions(simulate, duration, time_step)?
        .last()
        .copied()
        .ok_or_else(|| value_error("simulate returned no positions"))
}

fn lagged_square_displacement(
    simulate: &Py<PyAny>,
    duration: f64,
    delta: f64,
    time_step: f64,
) -> XPyResult<f64> {
    let x_vec = simulate_positions(simulate, duration, time_step)?;
    let lag_length = (delta / time_step).ceil() as usize;
    let lag_index = x_vec.len().checked_sub(lag_length + 1).ok_or_else(|| {
        value_error("simulate returned too few positions for the requested delta")
    })?;
    let end_position = *x_vec
        .last()
        .ok_or_else(|| value_error("simulate returned no positions"))?;
    let lag_position = x_vec[lag_index];

    Ok((end_position - lag_position) * (end_position - lag_position))
}

fn validate_tamsd_args(
    duration: f64,
    delta: f64,
    time_step: f64,
    quad_order: usize,
) -> XPyResult<()> {
    if quad_order == 0 {
        return Err(value_error("quad_order must be positive"));
    }
    if !duration.is_finite() || !delta.is_finite() || !time_step.is_finite() {
        return Err(value_error("duration, delta, and time_step must be finite"));
    }
    if duration <= 0.0 || delta <= 0.0 || time_step <= 0.0 {
        return Err(value_error(
            "duration, delta, and time_step must be positive",
        ));
    }
    if delta >= duration {
        return Err(value_error("delta must be less than duration"));
    }
    Ok(())
}

fn value_error(message: impl Into<String>) -> XPyError {
    XPyError::ValueError(message.into())
}

fn nodes_weights_transform(
    a: impl Into<f64>,
    b: impl Into<f64>,
    pairs: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let a: f64 = a.into();
    let b: f64 = b.into();
    pairs
        .iter()
        .map(|(node, weight)| {
            let new_weight = weight * (b - a) / 2.0;
            let new_node = (b - a) * node / 2.0 + (b + a) / 2.0;
            (new_node, new_weight)
        })
        .collect()
}
