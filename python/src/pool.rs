//! Thread-count control.
//!
//! Not `ThreadPoolBuilder::build_global()`: that succeeds exactly once per
//! process, so a second call from a notebook cell fails for reasons the user
//! cannot see. A pool held behind an `RwLock` and entered with `install` can be
//! replaced as often as anyone likes.

use std::sync::{Arc, RwLock};

use pyo3::prelude::*;

/// The override pool, or `None` when rayon's own global pool should serve.
static POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

/// Run `f` on the override pool if one is set, otherwise on rayon's global one.
///
/// ### Params
///
/// * `f` - The work to run. Must not hold the GIL: call this inside
///   `Python::detach`, never around it.
///
/// ### Returns
///
/// Whatever `f` returns.
pub(crate) fn run<T, F>(f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let pool = POOL.read().expect("thread pool lock poisoned").clone();
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Cap the threads the Rust core uses.
///
/// ### Params
///
/// * `n` - Thread count, or `0` to drop the override and go back to rayon's
///   default of one thread per core.
#[pyfunction]
pub(crate) fn set_num_threads(n: usize) -> PyResult<()> {
    let mut slot = POOL.write().expect("thread pool lock poisoned");

    if n == 0 {
        *slot = None;
        return Ok(());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    *slot = Some(Arc::new(pool));
    Ok(())
}

/// Threads the Rust core will use for the next call.
///
/// ### Returns
///
/// The override thread count if one is set, otherwise rayon's global count.
#[pyfunction]
pub(crate) fn num_threads() -> usize {
    match POOL.read().expect("thread pool lock poisoned").as_ref() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
