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


def validate_order(order: int) -> int:
    """Validate that order is a non-negative integer."""
    if not isinstance(order, int):
        raise TypeError(f"order must be an integer, got {type(order).__name__}")
    if order < 0:
        raise ValueError("order must be non-negative")
    return order


def validate_particles(particles: int) -> int:
    """Validate that particles is a positive integer."""
    if not isinstance(particles, int):
        raise TypeError(f"particles must be an integer, got {type(particles).__name__}")
    if particles <= 0:
        raise ValueError("particles must be positive")
    return particles


def validate_positive_float_param(value: real, param_name: str) -> float:
    """Validate that a parameter is a positive float after conversion."""
    try:
        float_value = ensure_float(value)
    except TypeError as e:
        raise TypeError(f"{param_name} must be a number. Error: {e}") from e
    if float_value <= 0:
        raise ValueError(f"{param_name} must be positive, got {float_value}")
    return float_value


def validate_domain(
    domain: tuple[real, real],
    domain_type: str = "interval",  # "interval", "poisson_fpt", "poisson_occupation"
    process_name: str = "",
) -> tuple[float, float]:
    """Validate domain based on type and convert its elements to float."""
    if not (isinstance(domain, tuple) and len(domain) == 2):
        base_msg = f"domain must be a tuple of two real numbers"
        if process_name:
            base_msg += f" for {process_name}"
        raise TypeError(f"{base_msg}, got {type(domain).__name__}")

    try:
        a = ensure_float(domain[0])
        b = ensure_float(domain[1])
    except TypeError as e:
        base_msg = f"Domain elements must be numbers convertible to float"
        if process_name:
            base_msg += f" for {process_name}"
        raise TypeError(f"{base_msg}. Error: {e}") from e

    if domain_type == "interval":
        if a >= b:
            base_msg = f"Invalid domain [{a}, {b}]; domain[0] must be strictly less than domain[1]"
            if process_name:
                base_msg += f" for {process_name}"
            raise ValueError(base_msg)
    elif domain_type == "poisson_fpt":
        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(
                f"Domain counts for FPT in {process_name or 'Poisson process'} must be integers, got {domain}"
            )
        _a_int, _b_int = int(a), int(b)
        if _a_int < 0 or _b_int < 0:
            raise ValueError(
                f"Domain counts for FPT in {process_name or 'Poisson process'} must be non-negative, got {(_a_int, _b_int)}"
            )
        if _b_int <= _a_int:
            raise ValueError(
                f"Target count domain[1] ({_b_int}) must be greater than start count domain[0] ({_a_int}) for FPT in {process_name or 'Poisson process'}."
            )
    elif domain_type == "poisson_occupation":
        if not (float(a).is_integer() and float(b).is_integer()):
            raise ValueError(
                f"Domain counts for occupation time in {process_name or 'Poisson process'} must be integers, got {domain}"
            )
        _a_int, _b_int = int(a), int(b)
        if _a_int < 0 or _b_int < 0:
            raise ValueError(
                f"Domain counts for occupation time in {process_name or 'Poisson process'} must be non-negative, got {(_a_int, _b_int)}"
            )
        if (
            _b_int < _a_int
        ):  # Allow a == b for occupation in a single state, so check b < a
            raise ValueError(
                f"max_count domain[1] ({_b_int}) must be greater than or equal to min_count domain[0] ({_a_int}) for occupation time in {process_name or 'Poisson process'}."
            )
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")

    return a, b
