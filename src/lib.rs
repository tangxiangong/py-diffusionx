use pyo3::prelude::*;

mod error;
pub use error::*;

pub mod random;

pub mod simulation;

macro_rules! register_functions {
    // 匹配 a::b::c 形式
    ($m:ident, $($p1:ident::$p2:ident::$func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($p1::$p2::$func, $m)?)?;
        )*
    };
    // 匹配 a::b 形式
    ($m:ident, $($p1:ident::$func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($p1::$func, $m)?)?;
        )*
    };
    // 匹配单独的 a 形式
    ($m:ident, $($func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($func, $m)?)?;
        )*
    };
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    register_functions!(
        m,
        random::exp_rand,
        random::exp_rands,
        random::uniform_rand_float,
        random::uniform_rand_int,
        random::uniform_rands_float,
        random::uniform_rands_int,
        random::normal_rand,
        random::normal_rands,
        random::poisson_rand,
        random::poisson_rands,
        random::stable_rand,
        random::stable_rands,
        random::skew_stable_rand,
        random::skew_stable_rands,
        random::bool_rand,
        random::bool_rands,
        // Brownian Motion
        simulation::bm_simulate,
        simulation::bm_raw_moment,
        simulation::bm_central_moment,
        simulation::bm_fpt,
        simulation::bm_fpt_raw_moment,
        simulation::bm_fpt_central_moment,
        simulation::bm_occupation_time,
        simulation::bm_occupation_time_raw_moment,
        simulation::bm_occupation_time_central_moment,
        simulation::bm_tamsd,
        simulation::bm_eatamsd,
        // Fractional Brownian Motion
        simulation::fbm_simulate,
        simulation::fbm_raw_moment,
        simulation::fbm_central_moment,
        simulation::fbm_fpt,
        simulation::fbm_fpt_raw_moment,
        simulation::fbm_fpt_central_moment,
        simulation::fbm_occupation_time,
        simulation::fbm_occupation_time_raw_moment,
        simulation::fbm_occupation_time_central_moment,
        simulation::fbm_tamsd,
        simulation::fbm_eatamsd,
        // Continuous Time Random Walk
        simulation::ctrw_simulate_duration,
        simulation::ctrw_simulate_step,
        simulation::ctrw_raw_moment,
        simulation::ctrw_central_moment,
        simulation::ctrw_fpt,
        simulation::ctrw_fpt_raw_moment,
        simulation::ctrw_fpt_central_moment,
        simulation::ctrw_occupation_time,
        simulation::ctrw_occupation_time_raw_moment,
        simulation::ctrw_occupation_time_central_moment,
        // Langevin Process
        simulation::langevin_simulate,
        simulation::langevin_raw_moment,
        simulation::langevin_central_moment,
        simulation::langevin_fpt,
        simulation::langevin_fpt_raw_moment,
        simulation::langevin_fpt_central_moment,
        simulation::langevin_occupation_time,
        simulation::langevin_occupation_time_raw_moment,
        simulation::langevin_occupation_time_central_moment,
        simulation::langevin_tamsd,
        simulation::langevin_eatamsd,
        // Generalized Langevin Process
        simulation::generalized_langevin_simulate,
        simulation::generalized_langevin_raw_moment,
        simulation::generalized_langevin_central_moment,
        simulation::generalized_langevin_fpt,
        simulation::generalized_langevin_fpt_raw_moment,
        simulation::generalized_langevin_fpt_central_moment,
        simulation::generalized_langevin_occupation_time,
        simulation::generalized_langevin_occupation_time_raw_moment,
        simulation::generalized_langevin_occupation_time_central_moment,
        simulation::generalized_langevin_tamsd,
        simulation::generalized_langevin_eatamsd,
        // Subordinated Langevin Process
        simulation::subordinated_langevin_simulate,
        simulation::subordinated_langevin_raw_moment,
        simulation::subordinated_langevin_central_moment,
        simulation::subordinated_langevin_fpt,
        simulation::subordinated_langevin_fpt_raw_moment,
        simulation::subordinated_langevin_fpt_central_moment,
        simulation::subordinated_langevin_occupation_time,
        simulation::subordinated_langevin_occupation_time_raw_moment,
        simulation::subordinated_langevin_occupation_time_central_moment,
        simulation::subordinated_langevin_tamsd,
        simulation::subordinated_langevin_eatamsd,
        // Levy Process
        simulation::levy_simulate,
        simulation::levy_fpt,
        simulation::levy_fpt_raw_moment,
        simulation::levy_fpt_central_moment,
        simulation::levy_occupation_time,
        simulation::levy_occupation_time_raw_moment,
        simulation::levy_occupation_time_central_moment,
        simulation::levy_tamsd,
        simulation::levy_eatamsd,
        // Poisson Process
        simulation::poisson_simulate_duration,
        simulation::poisson_simulate_step,
        simulation::poisson_raw_moment,
        simulation::poisson_central_moment,
        simulation::poisson_fpt,
        simulation::poisson_fpt_raw_moment,
        simulation::poisson_fpt_central_moment,
        simulation::poisson_occupation_time,
        simulation::poisson_occupation_time_raw_moment,
        simulation::poisson_occupation_time_central_moment,
        // Subordinator
        simulation::subordinator_simulate,
        simulation::subordinator_fpt,
        simulation::subordinator_fpt_raw_moment,
        simulation::subordinator_fpt_central_moment,
        simulation::subordinator_occupation_time,
        simulation::subordinator_occupation_time_raw_moment,
        simulation::subordinator_occupation_time_central_moment,
        simulation::inv_subordinator_simulate,
        simulation::inv_subordinator_raw_moment,
        simulation::inv_subordinator_central_moment,
        simulation::inv_subordinator_fpt,
        simulation::inv_subordinator_fpt_raw_moment,
        simulation::inv_subordinator_fpt_central_moment,
        simulation::inv_subordinator_occupation_time,
        simulation::inv_subordinator_occupation_time_raw_moment,
        simulation::inv_subordinator_occupation_time_central_moment,
    );

    Ok(())
}
