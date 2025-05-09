from typing import Union

real = Union[float, int]


def ensure_float(value: real) -> float:
    """Ensure the input value is a float, converting from int if necessary."""
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    else:
        # This case should ideally not be reached if type hints are respected,
        # but as a runtime check for robustness:
        raise TypeError(f"Expected float or int, got {type(value).__name__}")
