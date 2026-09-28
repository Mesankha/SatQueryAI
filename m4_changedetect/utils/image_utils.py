"""
Image utility module interface for M4 Change Detection.
Provides helper interfaces for reading satellite imagery, validating bi-temporal pairs,
and processing arrays.
"""

from typing import Any, Union, Tuple
from pathlib import Path


def load_image(image_path: Union[str, Path]) -> Any:
    """
    Interface for loading satellite image file from path.

    :param image_path: Path to satellite image file.
    :return: Loaded image data / array object.
    :raises NotImplementedError: Placeholder for algorithm implementation.
    """
    raise NotImplementedError("Image loading logic to be implemented.")


def validate_image_pair(image_t1: Any, image_t2: Any) -> bool:
    """
    Interface for validating bi-temporal image pair compatibility (dimensions, coordinate system).

    :param image_t1: Image observation at t1.
    :param image_t2: Image observation at t2.
    :return: True if valid pair.
    :raises NotImplementedError: Placeholder for algorithm implementation.
    """
    raise NotImplementedError("Bi-temporal image pair validation to be implemented.")


def clean_array(arr: Any) -> Any:
    """
    Interface for cleaning non-finite (NaN, Inf) values in image arrays.

    :param arr: Input numpy array or image object.
    :return: Cleaned array object.
    :raises NotImplementedError: Placeholder for algorithm implementation.
    """
    raise NotImplementedError("Array cleaning logic to be implemented.")
