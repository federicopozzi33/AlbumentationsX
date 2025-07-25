"""Module containing utility functions and classes for the core Albumentations framework.

This module provides a collection of helper functions and base classes used throughout
the Albumentations library. It includes utilities for shape handling, parameter processing,
data conversion, and serialization. The module defines abstract base classes for data
processors that implement the conversion logic between different data formats used in
the transformation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real
from typing import Any, Literal, cast, overload

import numpy as np

from albumentations.core.label_manager import LabelManager

from .serialization import Serializable
from .type_definitions import PAIR, Number


def get_shape(data: dict[str, Any]) -> tuple[int, int]:
    """Extract height and width dimensions from input data dictionary.

    After grayscale preprocessing, all data has channel dimension at the end.

    Args:
        data (dict[str, Any]): Dictionary containing image or volume data with one of:
            - 'volume': 3D array of shape (D, H, W, C)
            - 'volumes': Batch of 3D arrays of shape (N, D, H, W, C)
            - 'image': 2D array of shape (H, W, C)
            - 'images': Batch of arrays of shape (N, H, W, C)

    Returns:
        tuple[int, int]: (height, width) dimensions

    """
    # After preprocessing, all data has channel dimension at the end
    if "image" in data:
        return _get_shape_from_image(data["image"])
    if "images" in data:
        return _get_shape_from_images(data["images"])
    if "volume" in data:
        return _get_shape_from_volume(data["volume"])
    if "volumes" in data:
        return _get_shape_from_volumes(data["volumes"])

    raise ValueError("No image or volume found in data", data.keys())


def get_volume_shape(data: dict[str, Any]) -> tuple[int, int, int] | None:
    """Extract depth, height, and width dimensions from volume data.

    Args:
        data (dict[str, Any]): Dictionary containing volume data

    Returns:
        tuple[int, int, int] | None: (depth, height, width) dimensions if volume data exists, None otherwise

    """
    if "volume" in data:
        vol = data["volume"]
        # Handle PyTorch tensors
        if _is_torch_tensor(vol):
            if len(vol.shape) == 4:  # (C, D, H, W)
                return int(vol.shape[1]), int(vol.shape[2]), int(vol.shape[3])
            if len(vol.shape) == 3:  # (D, H, W)
                return int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        # Regular numpy array
        return vol.shape[0], vol.shape[1], vol.shape[2]

    if "volumes" in data:
        vols = data["volumes"]
        # Handle PyTorch tensors
        if _is_torch_tensor(vols):
            if len(vols.shape) == 5:  # (N, C, D, H, W)
                return int(vols.shape[2]), int(vols.shape[3]), int(vols.shape[4])
            if len(vols.shape) == 4:  # (N, D, H, W)
                return int(vols.shape[1]), int(vols.shape[2]), int(vols.shape[3])
        # Regular numpy array - take first volume
        return vols[0].shape[0], vols[0].shape[1], vols[0].shape[2]

    return None


def _is_torch_tensor(obj: Any) -> bool:
    """Check if an object is a PyTorch tensor."""
    return hasattr(obj, "__module__") and "torch" in obj.__module__


def _get_shape_from_image(img: np.ndarray) -> tuple[int, int]:
    """Extract shape from a single image."""
    # Check if it's a torch tensor that has been transposed to CHW format
    if _is_torch_tensor(img):
        # PyTorch tensor in CHW format
        if len(img.shape) == 3:  # (C, H, W)
            return int(img.shape[1]), int(img.shape[2])
        if len(img.shape) == 2:  # (H, W) - grayscale without channel
            return int(img.shape[0]), int(img.shape[1])
    # Regular numpy array in HWC format
    return img.shape[0], img.shape[1]


def _get_shape_from_images(imgs: np.ndarray) -> tuple[int, int]:
    """Extract shape from a batch of images."""
    # Check if it's a torch tensor batch
    if _is_torch_tensor(imgs):
        # PyTorch tensor batch in NCHW format
        if len(imgs.shape) == 4:  # (N, C, H, W)
            return int(imgs.shape[2]), int(imgs.shape[3])
        if len(imgs.shape) == 3:  # (N, H, W) - grayscale batch without channel
            return int(imgs.shape[1]), int(imgs.shape[2])
    # Regular numpy array batch in NHWC format - take first image
    return imgs[0].shape[0], imgs[0].shape[1]


def _get_shape_from_volume(vol: np.ndarray) -> tuple[int, int]:
    """Extract shape from a single volume."""
    # Check if it's a torch tensor
    if _is_torch_tensor(vol):
        # PyTorch 3D tensor in CDHW format
        if len(vol.shape) == 4:  # (C, D, H, W)
            return int(vol.shape[2]), int(vol.shape[3])
        if len(vol.shape) == 3:  # (D, H, W) - grayscale volume without channel
            return int(vol.shape[1]), int(vol.shape[2])
    # Regular numpy array in DHWC format
    return vol.shape[1], vol.shape[2]


def _get_shape_from_volumes(vols: np.ndarray) -> tuple[int, int]:
    """Extract shape from a batch of volumes."""
    # Check if it's a torch tensor batch
    if _is_torch_tensor(vols):
        # PyTorch 3D tensor batch in NCDHW format
        if len(vols.shape) == 5:  # (N, C, D, H, W)
            return int(vols.shape[3]), int(vols.shape[4])
        if len(vols.shape) == 4:  # (N, D, H, W) - grayscale volume batch without channel
            return int(vols.shape[2]), int(vols.shape[3])
    # Regular numpy array batch in NDHWC format - take first volume
    return vols[0].shape[1], vols[0].shape[2]


def format_args(args_dict: dict[str, Any]) -> str:
    """Format a dictionary of arguments into a string representation.

    Args:
        args_dict (dict[str, Any]): Dictionary of argument names and values.

    Returns:
        str: Formatted string of arguments in the form "key1='value1', key2=value2".

    """
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)


class Params(Serializable, ABC):
    """Base class for parameter handling in transforms.

    Args:
        format (Any): The format of the data this parameter object will process.
        label_fields (Sequence[str] | None): List of fields that are joined with the data, such as labels.

    """

    def __init__(self, format: Any, label_fields: Sequence[str] | None):  # noqa: A002
        self.format = format
        self.label_fields = label_fields

    def to_dict_private(self) -> dict[str, Any]:
        """Return a dictionary containing the private parameters of this object.

        Returns:
            dict[str, Any]: Dictionary with format and label_fields parameters.

        """
        return {"format": self.format, "label_fields": self.label_fields}


class DataProcessor(ABC):
    """Abstract base class for data processors.

    Data processors handle the conversion, validation, and filtering of data
    during transformations.

    Args:
        params (Params): Parameters for data processing.
        additional_targets (dict[str, str] | None): Dictionary mapping additional target names to their types.

    """

    def __init__(self, params: Params, additional_targets: dict[str, str] | None = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        self.is_sequence_input: dict[str, bool] = {}
        self.label_manager = LabelManager()

        if additional_targets is not None:
            self.add_targets(additional_targets)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        """Return the default name of the data field.

        Returns:
            str: Default data field name.

        """
        raise NotImplementedError

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Add targets to transform them the same way as one of existing targets."""
        for k, v in additional_targets.items():
            if v == self.default_data_name and k not in self.data_fields:
                self.data_fields.append(k)

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        """Validate input data before processing.

        Args:
            data (dict[str, Any]): Input data dictionary to validate.

        """

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        """Validate transforms before applying them.

        Args:
            transforms (Sequence[object]): Sequence of transforms to validate.

        """

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process data after transformation.

        Args:
            data (dict[str, Any]): Data dictionary after transformation.

        Returns:
            dict[str, Any]: Processed data dictionary.

        """
        shape: tuple[int, int] | tuple[int, int, int] = get_shape(data)

        # For xyz keypoints, get full 3D shape if available
        if hasattr(self.params, "format") and self.params.format == "xyz":
            volume_shape = get_volume_shape(data)
            if volume_shape is not None:
                shape = volume_shape

        data = self._process_data_fields(data, shape)
        data = self.remove_label_fields_from_data(data)
        return self._convert_sequence_inputs(data)

    def _process_data_fields(
        self,
        data: dict[str, Any],
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self._process_single_field(data_name, data[data_name], shape)
        return data

    def _process_single_field(
        self,
        data_name: str,
        field_data: Any,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> Any:
        field_data = self.filter(field_data, shape)

        if data_name == "keypoints" and len(field_data) == 0:
            field_data = self._create_empty_keypoints_array()

        return self.check_and_convert(field_data, shape, direction="from")

    def _create_empty_keypoints_array(self) -> np.ndarray:
        return np.array([], dtype=np.float32).reshape(0, len(self.params.format))

    def _convert_sequence_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            if self.is_sequence_input.get(data_name, False):
                data[data_name] = data[data_name].tolist()
        return data

    def preprocess(self, data: dict[str, Any]) -> None:
        """Process data before transformation.

        Args:
            data (dict[str, Any]): Data dictionary to preprocess.

        """
        shape = get_shape(data)

        for data_name in set(self.data_fields) & set(data.keys()):  # Convert list of lists to numpy array if necessary
            if isinstance(data[data_name], Sequence):
                self.is_sequence_input[data_name] = True
                data[data_name] = np.array(data[data_name], dtype=np.float32)
            else:
                self.is_sequence_input[data_name] = False

        data = self.add_label_fields_to_data(data)
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.check_and_convert(data[data_name], shape, direction="to")

    def check_and_convert(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        """Check and convert data between Albumentations and external formats.

        Args:
            data (np.ndarray): Input data array.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.
            direction (Literal["to", "from"], optional): Conversion direction.
                "to" converts to Albumentations format, "from" converts from it.
                Defaults to "to".

        Returns:
            np.ndarray: Converted data array.

        """
        if self.params.format == "albumentations":
            self.check(data, shape)
            return data

        process_func = self.convert_to_albumentations if direction == "to" else self.convert_from_albumentations

        return process_func(data, shape)

    @abstractmethod
    def filter(self, data: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
        """Filter data based on shapes.

        Args:
            data (np.ndarray): Data to filter.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Filtered data.

        """

    @abstractmethod
    def check(self, data: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> None:
        """Validate data structure against shape requirements.

        Args:
            data (np.ndarray): Data to validate.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        """

    @abstractmethod
    def convert_to_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert data from external format to Albumentations internal format.

        Args:
            data (np.ndarray): Data in external format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Data in Albumentations format.

        """

    @abstractmethod
    def convert_from_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert data from Albumentations internal format to external format.

        Args:
            data (np.ndarray): Data in Albumentations format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Data in external format.

        """

    def add_label_fields_to_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add label fields to data arrays.

        This method processes label fields and joins them with the corresponding data arrays.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Data with label fields added.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                continue
            data[data_name] = self._process_label_fields(data, data_name)

        return data

    def _process_label_fields(self, data: dict[str, Any], data_name: str) -> np.ndarray:
        data_array = data[data_name]
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                self._validate_label_field_length(data, data_name, label_field)
                encoded_labels = self.label_manager.process_field(data_name, label_field, data[label_field])
                data_array = np.hstack((data_array, encoded_labels))
                del data[label_field]
        return data_array

    def _validate_label_field_length(self, data: dict[str, Any], data_name: str, label_field: str) -> None:
        if len(data[data_name]) != len(data[label_field]):
            raise ValueError(
                f"The lengths of {data_name} and {label_field} do not match. "
                f"Got {len(data[data_name])} and {len(data[label_field])} respectively.",
            )

    def remove_label_fields_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove label fields from data arrays and restore them as separate entries.

        Args:
            data (dict[str, Any]): Input data dictionary with combined label fields.

        Returns:
            dict[str, Any]: Data with label fields extracted as separate entries.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                self._handle_empty_data_array(data)
                continue
            self._remove_label_fields(data, data_name)

        return data

    def _handle_empty_data_array(self, data: dict[str, Any]) -> None:
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                data[label_field] = self.label_manager.handle_empty_data()

    def _remove_label_fields(self, data: dict[str, Any], data_name: str) -> None:
        if self.params.label_fields is None:
            return

        data_array = data[data_name]
        num_label_fields = len(self.params.label_fields)
        non_label_columns = data_array.shape[1] - num_label_fields

        for idx, label_field in enumerate(self.params.label_fields):
            encoded_labels = data_array[:, non_label_columns + idx]
            data[label_field] = self.label_manager.restore_field(data_name, label_field, encoded_labels)

        data[data_name] = data_array[:, :non_label_columns]


def validate_args(
    low: float | Sequence[int] | Sequence[float] | None,
    bias: float | None,
) -> None:
    """Validate that 'low' and 'bias' parameters are not used together.

    Args:
        low (float | Sequence[int] | Sequence[float] | None): Lower bound value.
        bias (float | None): Bias value to be added to both min and max values.

    Raises:
        ValueError: If both 'low' and 'bias' are provided.

    """
    if low is not None and bias is not None:
        raise ValueError("Arguments 'low' and 'bias' cannot be used together.")


def process_sequence(param: Sequence[Number]) -> tuple[Number, Number]:
    """Process a sequence and return it as a (min, max) tuple.

    Args:
        param (Sequence[Number]): Sequence of numeric values.

    Returns:
        tuple[Number, Number]: Tuple containing (min_value, max_value) from the sequence.

    Raises:
        ValueError: If the sequence doesn't contain exactly 2 elements.

    """
    if len(param) != PAIR:
        raise ValueError("Sequence must contain exactly 2 elements.")
    return min(param), max(param)


def process_scalar(param: Number, low: Number | None) -> tuple[Number, Number]:
    """Process a scalar value and optional low bound into a (min, max) tuple.

    Args:
        param (Number): Scalar numeric value.
        low (Number | None): Optional lower bound.

    Returns:
        tuple[Number, Number]: Tuple containing (min_value, max_value) where:
            - If low is provided: (low, param) if low < param else (param, low)
            - If low is None: (-param, param) creating a symmetric range around zero

    """
    if isinstance(low, Real):
        return (low, param) if low < param else (param, low)
    return -param, param


def apply_bias(min_val: Number, max_val: Number, bias: Number) -> tuple[Number, Number]:
    """Apply a bias to both values in a range.

    Args:
        min_val (Number): Minimum value.
        max_val (Number): Maximum value.
        bias (Number): Value to add to both min and max.

    Returns:
        tuple[Number, Number]: Tuple containing (min_val + bias, max_val + bias).

    """
    return bias + min_val, bias + max_val


def ensure_int_output(
    min_val: Number,
    max_val: Number,
    param: Number,
) -> tuple[int, int] | tuple[float, float]:
    """Ensure output is of the same type (int or float) as the input parameter.

    Args:
        min_val (Number): Minimum value.
        max_val (Number): Maximum value.
        param (Number): Original parameter used to determine the output type.

    Returns:
        tuple[int, int] | tuple[float, float]: Tuple with values converted to int if param is int,
        otherwise values remain as float.

    """
    return (int(min_val), int(max_val)) if isinstance(param, int) else (float(min_val), float(max_val))


def ensure_contiguous_output(arg: np.ndarray | Sequence[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    """Ensure that numpy arrays are contiguous in memory.

    Args:
        arg (np.ndarray | Sequence[np.ndarray]): A numpy array or sequence of numpy arrays.

    Returns:
        np.ndarray | list[np.ndarray]: Contiguous array(s) with the same data.

    """
    if isinstance(arg, np.ndarray):
        arg = np.ascontiguousarray(arg)
    elif isinstance(arg, Sequence):
        arg = list(map(ensure_contiguous_output, arg))
    return arg


@overload
def to_tuple(
    param: int | tuple[int, int],
    low: int | tuple[int, int] | None = None,
    bias: float | None = None,
) -> tuple[int, int]: ...


@overload
def to_tuple(
    param: float | tuple[float, float],
    low: float | tuple[float, float] | None = None,
    bias: float | None = None,
) -> tuple[float, float]: ...


def to_tuple(
    param: float | tuple[float, float] | tuple[int, int],
    low: float | tuple[float, float] | tuple[int, int] | None = None,
    bias: float | None = None,
) -> tuple[float, float] | tuple[int, int]:
    """Convert input argument to a min-max tuple.

    This function processes various input types and returns a tuple representing a range.
    It handles single values, sequences, and can apply optional low bounds or biases.

    Args:
        param (tuple[float, float] | float | tuple[int, int] | int): The primary input value. Can be:
            - A single int or float: Converted to a symmetric range around zero.
            - A tuple of two ints or two floats: Used directly as min and max values.

        low (tuple[float, float] | float | None, optional): A lower bound value. Used when param is a single value.
            If provided, the result will be (low, param) or (param, low), depending on which is smaller.
            Cannot be used together with 'bias'. Defaults to None.

        bias (float | int | None, optional): A value to be added to both elements of the resulting tuple.
            Cannot be used together with 'low'. Defaults to None.

    Returns:
        tuple[int, int] | tuple[float, float]: A tuple representing the processed range.
            - If input is int-based, returns tuple[int, int]
            - If input is float-based, returns tuple[float, float]

    Raises:
        ValueError: If both 'low' and 'bias' are provided.
        TypeError: If 'param' is neither a scalar nor a sequence of 2 elements.

    Examples:
        >>> to_tuple(5)
        (-5, 5)
        >>> to_tuple(5.0)
        (-5.0, 5.0)
        >>> to_tuple((1, 10))
        (1, 10)
        >>> to_tuple(5, low=3)
        (3, 5)
        >>> to_tuple(5, bias=1)
        (-4, 6)

    Notes:
        - When 'param' is a single value and 'low' is not provided, the function creates a symmetric range around zero.
        - The function preserves the type (int or float) of the input in the output.
        - If a sequence is provided, it must contain exactly 2 elements.

    """
    validate_args(low, bias)

    if isinstance(param, Sequence):
        min_val, max_val = process_sequence(param)
    elif isinstance(param, Real):
        min_val, max_val = process_scalar(param, cast("Real", low))
    else:
        raise TypeError("Argument 'param' must be either a scalar or a sequence of 2 elements.")

    if bias is not None:
        min_val, max_val = apply_bias(min_val, max_val, bias)

    return ensure_int_output(min_val, max_val, param if isinstance(param, (int, float)) else min_val)
