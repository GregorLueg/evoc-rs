//! Python bindings for `evoc-rs`.
//!
//! Deliberately thin. EVoC is one-shot: there is no index to build and hold, so
//! there are no handle classes, no serialisation and no pickle protocol here.
//! The whole compiled surface is two functions returning numpy arrays, plus the
//! thread-pool and GPU-probe helpers. Defaults, validation and the
//! scikit-learn shaped estimator all live in `evoc_rs/`.
//!
//! Two invariants hold throughout, and both are load-bearing:
//!
//! - Everything expensive runs inside `Python::detach`, so the GIL is dropped
//!   for the whole rayon fan-out. See [`evoc`] for the nesting rule.
//! - Nothing generic reaches `evoc_rs::evoc`. The dispatch macro expands to
//!   concrete `f32` / `f64` code, so the core's trait bounds never have to be
//!   restated and `ann-search-rs` stays out of this crate's manifest.

#![warn(missing_docs)]

use pyo3::prelude::*;

mod convert;
mod error;
mod evoc;
mod gpu_probe;
mod params;
mod pool;

#[cfg(feature = "gpu")]
mod evoc_gpu;

/// Assemble the compiled module.
#[pymodule]
fn _evoc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", evoc_rs::VERSION)?;

    m.add("EvocError", m.py().get_type::<error::EvocError>())?;

    m.add_function(wrap_pyfunction!(pool::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pool::num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_probe::gpu_available, m)?)?;

    m.add_function(wrap_pyfunction!(evoc::run_evoc, m)?)?;

    #[cfg(feature = "gpu")]
    m.add_function(wrap_pyfunction!(evoc_gpu::run_evoc_gpu, m)?)?;

    Ok(())
}
