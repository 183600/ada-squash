from __future__ import annotations

"""Small scalar-validation predicates shared across the package.

The helpers are intentionally dependency-free so importing them never pulls
in torch, numpy or any analysis module.
"""


def is_boolean_scalar(value) -> bool:
    """Whether ``value`` is a Python or NumPy boolean.

    ``isinstance(np.bool_(True), bool)`` is False. Without this check a NumPy
    boolean would be coerced to 1/0 by ``int()``/``float()`` and enter a count,
    seed or measurement as if it were a real observation.
    """
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')
