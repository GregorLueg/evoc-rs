"""Shared estimator behaviour.

`get_params` and `set_params` introspect the subclass `__init__`, which is all
`sklearn.base.BaseEstimator` does. Doing it here keeps scikit-learn out of the
install requirements while `clone`, `GridSearchCV` and `Pipeline` still work by
duck-typing.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from beartype import beartype


class NotFittedError(ValueError, AttributeError):
    """Raised when a fitted attribute is read before `fit`.

    Inherits from both `ValueError` and `AttributeError` to match
    `sklearn.exceptions.NotFittedError`, so code catching either still works.
    """


class BaseEstimator:
    """Parameter introspection shared by every estimator here.

    Subclasses store their constructor arguments verbatim on `self` under the
    same names. That is the whole contract; nothing else is inspected.
    """

    @classmethod
    def _param_names(cls) -> list[str]:
        """Constructor argument names, in signature order.

        Returns:
            Every named parameter of `__init__` bar `self`.
        """
        signature = inspect.signature(cls.__init__)
        return [
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not param.VAR_KEYWORD
        ]

    @beartype
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor arguments as a dict.

        Args:
            deep: Accepted for scikit-learn compatibility. These estimators
                hold no nested estimators, so it changes nothing.

        Returns:
            Mapping from parameter name to its current value.
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set constructor arguments in place.

        Args:
            **params: Parameter names and their new values.

        Returns:
            The estimator, so calls chain.

        Raises:
            ValueError: If a name is not a constructor parameter.
        """
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                options = ", ".join(sorted(valid))
                raise ValueError(
                    f"{name!r} is not a parameter; expected one of {options}"
                )
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _fitted_labels(self) -> np.ndarray:
        """Guard a fitted attribute read and hand back the labels.

        Returning the array rather than just raising is what lets callers stay
        free of the `ndarray | None` the unfitted class attribute declares.

        Returns:
            The `labels_` array.

        Raises:
            NotFittedError: If `fit` has not run.
        """
        labels = getattr(self, "labels_", None)
        if labels is None:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit first"
            )
        return labels
