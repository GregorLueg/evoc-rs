# Top level

Module-level helpers and the exception type. The estimators are documented
under [Estimators](estimator.md).

::: evoc_rs
    options:
      members:
        - gpu_available
        - num_threads
        - set_num_threads
        - EvocError
        - NotFittedError

## Parameter introspection

`get_params` and `set_params` are reimplemented rather than inherited from
scikit-learn, so `clone`, `GridSearchCV` and `Pipeline` all work by duck-typing
without scikit-learn being an install requirement.

::: evoc_rs._base
    options:
      members:
        - BaseEstimator

## Version strings

`__version__` is this wheel. `__core_version__` is the `evoc-rs` crate it
vendored. The two version independently, so the second is the one that tells you
what the numerics are.

```python
import evoc_rs

evoc_rs.__version__, evoc_rs.__core_version__
```
