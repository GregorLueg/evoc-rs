//! Rust errors as Python exceptions.
//!
//! The mapping is deliberately flat. `EvocErrors` only wraps `AnnSearchErrors`,
//! and every condition a caller could trigger by hand (wrong dtype, wrong
//! shape, unknown metric, `k` past the sample count) is rejected in
//! `evoc_rs._validate` before the call. Anything that still comes back from the
//! core is therefore an internal failure, not a bad argument.

use evoc_rs::errors::EvocErrors;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    _evoc_rs,
    EvocError,
    PyException,
    "Raised when the evoc-rs Rust core fails."
);

/// Newtype carrying an [`EvocErrors`] across the orphan rule.
///
/// Doubles as the conversion function: `.map_err(EvocErr)?` in a `PyResult`
/// function, plain `?` in an [`EvocPyResult`] one.
pub(crate) struct EvocErr(pub EvocErrors);

impl From<EvocErrors> for EvocErr {
    fn from(e: EvocErrors) -> Self {
        Self(e)
    }
}

impl From<EvocErr> for PyErr {
    fn from(e: EvocErr) -> PyErr {
        EvocError::new_err(e.0.to_string())
    }
}

/// Result carrying an [`EvocErr`], so a body can use plain `?`.
#[allow(dead_code)] // used by the gpu module only
pub(crate) type EvocPyResult<T> = Result<T, EvocErr>;
