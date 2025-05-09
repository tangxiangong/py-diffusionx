#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::prelude::*;

/// 封装从Python调用函数的辅助方法，处理错误情况
pub(crate) fn call_py_func(func: &PyObject, args: (f64, f64)) -> f64 {
    let res = Python::with_gil(|py| {
        func.call1(py, args)
            .and_then(|result| result.extract::<f64>(py))
    });
    match res {
        Ok(res) => res,
        Err(e) => {
            panic!("Error calling Python function: {}", e);
        }
    }
}

pub(crate) type PyArrayPair<'py> = (Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix1>>);

pub(crate) fn vec_to_pyarray(py: Python, time: Vec<f64>, position: Vec<f64>) -> PyArrayPair {
    let time_array = time.into_pyarray(py);
    let position_array = position.into_pyarray(py);

    (time_array, position_array)
}

mod bm;
pub use bm::*;

mod levy;
pub use levy::*;

mod poisson;
pub use poisson::*;

mod subordinator;
pub use subordinator::*;

mod fbm;
pub use fbm::*;

mod ctrw;
pub use ctrw::*;

mod langevin;
pub use langevin::*;
