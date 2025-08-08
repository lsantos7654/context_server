"""Common validation utilities for domain models."""


def validate_non_negative(value, field_name: str) -> None:
    """Validate that a numeric value is non-negative.

    Args:
        value: The value to validate (can be None)
        field_name: Name of the field for error messages

    Raises:
        ValueError: If value is not None and is negative
    """
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def validate_range_fields(obj, fields: list[str]) -> None:
    """Validate multiple range-related fields are non-negative.

    Args:
        obj: Object containing the fields to validate
        fields: List of field names to validate

    Raises:
        ValueError: If any field is negative
    """
    for field_name in fields:
        value = getattr(obj, field_name, None)
        validate_non_negative(value, field_name)


__all__ = ["validate_non_negative", "validate_range_fields"]
