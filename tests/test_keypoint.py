from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import cv2

import albumentations as A
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.utils import angle_2pi_range
from albumentations.core.keypoints_utils import (
    angle_to_2pi_range,
    check_keypoints,
    convert_keypoints_from_albumentations,
    convert_keypoints_to_albumentations,
    filter_keypoints,
)
from albumentations.core.transforms_interface import BasicTransform
from tests.conftest import RECTANGULAR_UINT8_IMAGE


@pytest.mark.parametrize(
    "input_angles, expected_angles",
    [
        (np.array([0, np.pi, 2 * np.pi]), np.array([0, np.pi, 0])),
        (np.array([-np.pi, 3 * np.pi]), np.array([np.pi, np.pi])),
        (np.array([0.5, 2.5, 7.5]), np.array([0.5, 2.5, 7.5 - 2 * np.pi])),
        (np.array([]), np.array([])),
        (np.array([10 * np.pi, 100 * np.pi]), np.array([0, 0])),
    ],
)
def test_angle_to_2pi_range(input_angles, expected_angles):
    result = angle_to_2pi_range(input_angles)
    np.testing.assert_allclose(result, expected_angles, atol=1e-7)


def test_angle_to_2pi_range_large_array():
    input_angles = np.random.uniform(-10 * np.pi, 10 * np.pi, 1000)
    result = angle_to_2pi_range(input_angles)
    assert np.all((result >= 0) & (result < 2 * np.pi))
    np.testing.assert_allclose((input_angles - result) % (2 * np.pi), 0, atol=1e-8)


def test_angle_to_2pi_range_precision():
    small_angle = np.array([1e-10])
    result = angle_to_2pi_range(small_angle)
    np.testing.assert_allclose(result, small_angle, atol=1e-15)


def test_angle_to_2pi_range_negative_zero():
    input_angles = np.array([-0.0, 0.0])
    result = angle_to_2pi_range(input_angles)
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)
    assert not np.signbit(result[0])  # Ensure -0.0 is converted to +0.0


@pytest.mark.parametrize(
    "keypoints, image_shape, expected_error",
    [
        # Valid keypoints
        (np.array([[10, 20, 0, 0.5], [30, 40, 0, 1.5]]), (100, 100), None),
        (np.array([[0, 0, 0, 0], [99, 99, 0, math.pi]]), (100, 100), None),
        # Invalid x coordinate
        (np.array([[100, 50, 1.0]]), (100, 100), "Expected x for keypoint"),
        (np.array([[-1, 50, 1.0]]), (100, 100), "Expected x for keypoint"),
        # Invalid y coordinate
        (np.array([[50, 100, 1.0]]), (100, 100), "Expected y for keypoint"),
        (np.array([[50, -1, 1.0]]), (100, 100), "Expected y for keypoint"),
        # Multiple invalid keypoints
        (np.array([[100, 50, 1.0], [50, 100, 1.0]]), (100, 100), "Expected x for keypoint"),
        # Keypoints without angle
        (np.array([[10, 20], [30, 40]]), (100, 100), None),
    ],
)
def test_check_keypoints(keypoints, image_shape, expected_error):
    if expected_error is None:
        check_keypoints(keypoints, image_shape)  # Should not raise an error
    else:
        with pytest.raises(ValueError) as exc_info:
            check_keypoints(keypoints, image_shape)
        assert expected_error in str(exc_info.value)


@pytest.mark.parametrize(
    "keypoints, image_shape",
    [
        (np.array([[10, 20, 0.5, 1.0], [30, 40, 1.5, 2.0]]), (100, 100)),
        (np.array([[0, 0, 0, 1.0], [99, 99, math.pi, 2.0]]), (100, 100)),
    ],
)
def test_check_keypoints_with_scale(keypoints, image_shape):
    check_keypoints(keypoints, image_shape)  # Should not raise an error


@pytest.mark.parametrize(
    "keypoints, image_shape",
    [
        (np.array([[10, 20, 0.5, 1.0, 1], [30, 40, 1.5, 2.0, 2]]), (100, 100)),
        (np.array([[0, 0, 0, 1.0, 1], [99, 99, math.pi, 2.0, 2]]), (100, 100)),
    ],
)
def test_check_keypoints_with_extra_data(keypoints, image_shape):
    check_keypoints(keypoints, image_shape)  # Should not raise an error


@pytest.mark.parametrize(
    "keypoints, image_shape, remove_invisible, expected",
    [
        # Test case 1: All keypoints are visible
        (
            np.array([[10, 20, 0.5], [30, 40, 1.0], [50, 60, 1.5]]),
            (100, 100),
            True,
            np.array([[10, 20, 0.5], [30, 40, 1.0], [50, 60, 1.5]]),
        ),
        # Test case 2: Some keypoints are outside the image
        (
            np.array([[-10, 20, 0.5], [30, 40, 1.0], [110, 60, 1.5], [50, 120, 2.0]]),
            (100, 100),
            True,
            np.array([[30, 40, 1.0]]),
        ),
        # Test case 3: All keypoints are outside the image
        (
            np.array([[-10, -20, 0.5], [110, 120, 1.0]]),
            (100, 100),
            True,
            np.array([], dtype=float).reshape(0, 3),
        ),
        # Test case 4: remove_invisible is False
        (
            np.array([[-10, 20, 0.5], [30, 40, 1.0], [110, 60, 1.5]]),
            (100, 100),
            False,
            np.array([[-10, 20, 0.5], [30, 40, 1.0], [110, 60, 1.5]]),
        ),
        # Test case 5: Empty input array
        (
            np.array([], dtype=float).reshape(0, 3),
            (100, 100),
            True,
            np.array([], dtype=float).reshape(0, 3),
        ),
        # Test case 6: Keypoints with additional data
        (
            np.array([[10, 20, 0.5, 1], [30, 40, 1.0, 2], [110, 60, 1.5, 3]]),
            (100, 100),
            True,
            np.array([[10, 20, 0.5, 1], [30, 40, 1.0, 2]]),
        ),
    ],
)
def test_filter_keypoints(keypoints, image_shape, remove_invisible, expected):
    result = filter_keypoints(keypoints, image_shape, remove_invisible)
    np.testing.assert_array_equal(result, expected)


def test_filter_keypoints_with_float_coordinates():
    keypoints = np.array([[0.5, 0.5, 0.5], [99.9, 99.9, 1.0], [100.1, 100.1, 1.5]])
    image_shape = (100, 100)
    remove_invisible = True
    expected = np.array([[0.5, 0.5, 0.5], [99.9, 99.9, 1.0]])
    result = filter_keypoints(keypoints, image_shape, remove_invisible)
    np.testing.assert_array_almost_equal(result, expected)


def test_filter_keypoints_with_int_image_shape():
    keypoints = np.array([[10, 20, 0.5], [30, 40, 1.0], [50, 60, 1.5]])
    image_shape = (100, 100)
    remove_invisible = True
    expected = np.array([[10, 20, 0.5], [30, 40, 1.0], [50, 60, 1.5]])
    result = filter_keypoints(keypoints, image_shape, remove_invisible)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "keypoints, source_format, image_shape, check_validity, angle_in_degrees, expected",
    [
        # Test case 1: xy format
        (
            np.array([[10, 20], [30, 40]]),
            "xy",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, 0, 0], [30, 40, 0, 0, 0]]),
        ),
        # Test case 2: yx format
        (
            np.array([[20, 10], [40, 30]]),
            "yx",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, 0, 0], [30, 40, 0, 0, 0]]),
        ),
        # Test case 3: xya format with degrees
        (
            np.array([[10, 20, 45], [30, 40, 90]]),
            "xya",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, np.pi / 4, 0], [30, 40, 0, np.pi / 2, 0]]),
        ),
        # Test case 4: xya format with radians
        (
            np.array([[10, 20, np.pi / 4], [30, 40, np.pi / 2]]),
            "xya",
            (100, 100),
            False,
            False,
            np.array([[10, 20, 0, np.pi / 4, 0], [30, 40, 0, np.pi / 2, 0]]),
        ),
        # Test case 5: xys format
        (
            np.array([[10, 20, 2], [30, 40, 3]]),
            "xys",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, 0, 2], [30, 40, 0, 0, 3]]),
        ),
        # Test case 6: xyas format with degrees
        (
            np.array([[10, 20, 45, 2], [30, 40, 90, 3]]),
            "xyas",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, np.pi / 4, 2], [30, 40, 0, np.pi / 2, 3]]),
        ),
        # Test case 7: xysa format with degrees
        (
            np.array([[10, 20, 2, 45], [30, 40, 3, 90]]),
            "xysa",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, np.pi / 4, 2], [30, 40, 0, np.pi / 2, 3]]),
        ),
        # Test case 8: with additional columns
        (
            np.array([[10, 20, 45, 2, 1], [30, 40, 90, 3, 2]]),
            "xyas",
            (100, 100),
            False,
            True,
            np.array([[10, 20, 0, np.pi / 4, 2, 1], [30, 40, 0, np.pi / 2, 3, 2]]),
        ),
    ],
)
def test_convert_keypoints_to_albumentations(
    keypoints,
    source_format,
    image_shape,
    check_validity,
    angle_in_degrees,
    expected,
):
    result = convert_keypoints_to_albumentations(
        keypoints,
        source_format,
        image_shape,
        check_validity,
        angle_in_degrees,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_convert_keypoints_to_albumentations_invalid_format():
    with pytest.raises(ValueError, match="Unknown source_format"):
        convert_keypoints_to_albumentations(np.array([[10, 20]]), "invalid_format", (100, 100))


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, 0),
        (np.pi, np.pi),
        (2 * np.pi, 0),
        (3 * np.pi, np.pi),
        (-np.pi, np.pi),
        (-2 * np.pi, 0),
    ],
)
def test_angle_to_2pi_range(angle, expected):
    result = angle_to_2pi_range(np.array([angle]))
    np.testing.assert_allclose(result, [expected], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "keypoints, source_format, image_shape",
    [
        (np.array([[10, 20], [-10, 30]]), "xy", (100, 100)),
        (np.array([[10, 20], [110, 30]]), "xy", (100, 100)),
        (np.array([[10, -20], [30, 40]]), "xy", (100, 100)),
        (np.array([[10, 120], [30, 40]]), "xy", (100, 100)),
    ],
)
def test_convert_keypoints_to_albumentations_check_validity(keypoints, source_format, image_shape):
    with pytest.raises(ValueError):
        convert_keypoints_to_albumentations(keypoints, source_format, image_shape, check_validity=True)


@pytest.mark.parametrize(
    "keypoints, target_format, image_shape, check_validity, angle_in_degrees, expected",
    [
        # Test case 1: xy format
        (
            np.array([[10, 20, 33, 0, 0], [30, 40, 33, 0, 0]]),
            "xy",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20], [30, 40]]),
        ),
        # Test case 2: yx format
        (
            np.array([[10, 20, 33, 0, 0], [30, 40, 33, 0, 0]]),
            "yx",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[20, 10], [40, 30]]),
        ),
        # Test case 3: xya format with degrees
        (
            np.array([[10, 20, 33, np.pi / 4, 0], [30, 40, 33, np.pi / 2, 0]]),
            "xya",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20, 45], [30, 40, 90]]),
        ),
        # Test case 4: xya format with radians
        (
            np.array([[10, 20, 33, np.pi / 4, 0], [30, 40, 33, np.pi / 2, 0]]),
            "xya",
            {"height": 100, "width": 100},
            False,
            False,
            np.array([[10, 20, np.pi / 4], [30, 40, np.pi / 2]]),
        ),
        # Test case 5: xys format
        (
            np.array([[10, 20, 33, 0, 2], [30, 40, 33, 0, 3]]),
            "xys",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20, 2], [30, 40, 3]]),
        ),
        # Test case 6: xyas format with degrees
        (
            np.array([[10, 20, 33, np.pi / 4, 2], [30, 40, 33, np.pi / 2, 3]]),
            "xyas",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20, 45, 2], [30, 40, 90, 3]]),
        ),
        # Test case 7: xysa format with degrees
        (
            np.array([[10, 20, 33, np.pi / 4, 2], [30, 40, 33, np.pi / 2, 3]]),
            "xysa",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20, 2, 45], [30, 40, 3, 90]]),
        ),
        # Test case 8: with additional columns
        (
            np.array([[10, 20, 33, np.pi / 4, 2, 1], [30, 40, 33, np.pi / 2, 3, 2]]),
            "xyas",
            {"height": 100, "width": 100},
            False,
            True,
            np.array([[10, 20, 45, 2, 1], [30, 40, 90, 3, 2]]),
        ),
    ],
)
def test_convert_keypoints_from_albumentations(
    keypoints,
    target_format,
    image_shape,
    check_validity,
    angle_in_degrees,
    expected,
):
    result = convert_keypoints_from_albumentations(
        keypoints,
        target_format,
        image_shape,
        check_validity,
        angle_in_degrees,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_convert_keypoints_from_albumentations_invalid_format():
    with pytest.raises(ValueError, match="Unknown target_format"):
        convert_keypoints_from_albumentations(np.array([[10, 20, 0, 0]]), "invalid_format", {"height": 100, "width": 100})


@pytest.mark.parametrize(
    "keypoints, source_format, image_shape",
    [
        (np.array([[10, 20], [30, 40]]), "xy", (100, 100)),
        (np.array([[20, 10], [40, 30]]), "yx", (100, 100)),
        (np.array([[10, 20, 45], [30, 40, 90]]), "xya", (100, 100)),
        (np.array([[10, 20, 2], [30, 40, 3]]), "xys", (100, 100)),
        (np.array([[10, 20, 45, 2], [30, 40, 90, 3]]), "xyas", (100, 100)),
        (np.array([[10, 20, 2, 45], [30, 40, 3, 90]]), "xysa", (100, 100)),
        (np.array([[10, 20, 45, 2, 1], [30, 40, 90, 3, 2]]), "xyas", (100, 100)),
    ],
)
def test_keypoint_conversion_roundtrip(keypoints, source_format, image_shape):
    # original => to_albumentations => from_albumentations
    albumentations_format = convert_keypoints_to_albumentations(keypoints, source_format, image_shape)
    result = convert_keypoints_from_albumentations(albumentations_format, source_format, image_shape)
    np.testing.assert_allclose(result, keypoints, rtol=1e-5, atol=1e-8)

    # albumentations => from_albumentations => to_albumentations
    other_format = convert_keypoints_from_albumentations(albumentations_format, source_format, image_shape)
    result = convert_keypoints_to_albumentations(other_format, source_format, image_shape)
    np.testing.assert_allclose(result, albumentations_format, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "keypoints, image_shape",
    [
        (np.array([[10, 20, 1, 0, 0], [-10, 30, 1, 0, 0]]), (100, 100)),
        (np.array([[10, 20, 1, 0, 0], [110, 30, 1, 0, 0]]), (100, 100)),
        (np.array([[10, -20, 1, 0, 0], [30, 40, 1, 0, 0]]), (100, 100)),
        (np.array([[10, 120, 1, 0, 0], [30, 40, 1, 0, 0]]), (100, 100)),
    ],
)
def test_convert_keypoints_from_albumentations_check_validity(keypoints, image_shape):
    with pytest.raises(ValueError):
        convert_keypoints_from_albumentations(keypoints, "xy", image_shape, check_validity=True)


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 40, 50)], "xyas", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "xy", None),
        ([(20, 30, 60, 80)], "yx", [2]),
        ([(20, 30, 60, 80, 99)], "xys", None),
        ([(20, 30, 1, 40, 50)], "xyz", [1]),
    ],
)
def test_compose_with_keypoint_noop(keypoints, keypoint_format: str, labels: int | None) -> None:
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = A.Compose(
            [A.NoOp(p=1.0)],
            keypoint_params={"format": keypoint_format, "label_fields": ["labels"]},
            strict=True,
        )
        transformed = aug(image=image, keypoints=keypoints, labels=labels)
    else:
        aug = A.Compose([A.NoOp(p=1.0)], keypoint_params={"format": keypoint_format}, strict=True)
        transformed = aug(image=image, keypoints=keypoints)

    np.testing.assert_array_equal(transformed["image"], image)
    np.testing.assert_allclose(transformed["keypoints"], keypoints)


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format"],
    [([[20, 30, 40, 50]], "xyas"), (np.array([[20, 30, 40, 50]]), "xyas")],
)
def test_compose_with_keypoint_noop_error_label_fields(keypoints, keypoint_format: str) -> None:
    image = np.ones((100, 100, 3))
    aug = A.Compose(
        [A.NoOp(p=1.0)],
        keypoint_params={"format": keypoint_format, "label_fields": "class_id"},
        strict=True,
    )
    with pytest.raises(Exception):
        aug(image=image, keypoints=keypoints, cls_id=["temp_label"])


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 60, 80)], "xy", {"label": [1]}),
        ([], "xy", {}),
        ([], "xy", {"label": []}),
        ([(20, 30, 60, 80)], "xy", {"id": [3]}),
        ([(20, 30, 60, 80), (30, 40, 40, 50)], "xy", {"id": [3, 1]}),
    ],
)
def test_compose_with_keypoint_noop_label_outside(keypoints, keypoint_format: str, labels: dict[str, Any]) -> None:
    image = np.ones((100, 100, 3))
    aug = A.Compose(
        [A.NoOp(p=1.0)],
        keypoint_params={
            "format": keypoint_format,
            "label_fields": list(labels.keys()),
            },
        strict=True,
    )
    transformed = aug(image=image, keypoints=keypoints, **labels)

    np.testing.assert_array_equal(transformed["image"], image)
    np.testing.assert_allclose(transformed["keypoints"], keypoints)

    for k, v in labels.items():
        assert transformed[k] == v


@pytest.mark.parametrize(
    ["aug", "keypoints", "expected"],
    [
        [A.HorizontalFlip, [[0, 0]], [[2, 0]]],
        [A.HorizontalFlip, [[2, 0]], [[0, 0]]],
        [A.HorizontalFlip, [[0, 2]], [[2, 2]]],
        [A.HorizontalFlip, [[2, 2]], [[0, 2]]],
        [A.VerticalFlip, [[0, 0]], [[0, 2]]],
        [A.VerticalFlip, [[2, 0]], [[2, 2]]],
        [A.VerticalFlip, [[0, 2]], [[0, 0]]],
        [A.VerticalFlip, [[2, 2]], [[2, 0]]],
        [A.HorizontalFlip, [[1, 1]], [[1, 1]]],
        [A.VerticalFlip, [[1, 1]], [[1, 1]]],
    ],
)
def test_keypoint_flips_transform_3x3(aug: BasicTransform, keypoints, expected) -> None:
    transform = A.Compose([aug(p=1)], keypoint_params={"format": "xy"}, strict=True)

    image = np.ones((3, 3, 3))
    transformed = transform(
        image=image,
        keypoints=keypoints,
        labels=np.ones(len(keypoints)),
    )
    np.testing.assert_allclose(expected, transformed["keypoints"])


@pytest.mark.parametrize(
    ["aug", "keypoints", "expected"],
    [
        [A.HorizontalFlip, [[20, 30, 0, 0]], [[79, 30, 180, 0]]],
        [A.HorizontalFlip, [[20, 30, 45, 0]], [[79, 30, 135, 0]]],
        [A.HorizontalFlip, [[20, 30, 90, 0]], [[79, 30, 90, 0]]],
        [A.VerticalFlip, [[20, 30, 0, 0]], [[20, 69, 0, 0]]],
        [A.VerticalFlip, [[20, 30, 45, 0]], [[20, 69, 315, 0]]],
        [A.VerticalFlip, [[20, 30, 90, 0]], [[20, 69, 270, 0]]],
    ],
)
def test_keypoint_transform_format_xyas(aug: BasicTransform, keypoints, expected) -> None:
    transform = A.Compose(
        [aug(p=1)],
        keypoint_params={
            "format": "xyas",
            "angle_in_degrees": True,
            "label_fields": ["labels"],
        },
        strict=True,
    )

    image = np.ones((100, 100, 3))
    transformed = transform(
        image=image,
        keypoints=keypoints,
        labels=np.ones(len(keypoints)),
    )
    np.testing.assert_allclose(expected, transformed["keypoints"])


@pytest.mark.parametrize(
    ["keypoint", "expected", "factor"],
    [
        ((20, 30, 1, math.pi / 2, 0), (20, 30, 1, math.pi / 2, 0), 0),
        ((20, 30, 1, math.pi / 2, 0), (30, 179, 1, 0, 0), 1),
        ((20, 30, 1, math.pi / 2, 0), (179, 69, 1, 3 * math.pi / 2, 0), 2),
        ((20, 30, 1, math.pi / 2, 0), (69, 20, 1, math.pi, 0), 3),
    ],
)
def test_keypoint_rotate90(keypoint, expected, factor: int) -> None:
    actual = fgeometric.keypoints_rot90(np.array([keypoint]), factor, (100, 200))
    np.testing.assert_allclose(actual, [expected], atol=1e-7)


@pytest.mark.parametrize(
    ["keypoint", "expected", "scale"],
    [
        # No scaling (scale = 1)
        [[10.0, 20.0, 1, 0, 1], [10.0, 20.0, 1, 0, 1], 1],

        # Uniform scaling up (scale = 2)
        [[10.0, 20.0, 1, 0, 1], [20.0, 40.0, 1, 0, 2], 2],

        # Uniform scaling down (scale = 0.5)
        [[10.0, 20.0, 1, 0, 1], [5.0, 10.0, 1, 0, 0.5], 0.5],

        # Different coordinates, angle, and initial scale
        [[100.0, 50.0, 1, np.pi/4, 2], [200.0, 100.0, 1, np.pi/4, 4], 2],
    ],
)
def test_keypoint_scale(keypoint, expected, scale: float) -> None:
    """Test keypoint scaling with non-zero coordinates.

    Tests that:
    1. x and y coordinates are scaled correctly
    2. z coordinate remains unchanged
    3. angle remains unchanged (scaling preserves directions)
    4. keypoint scale is multiplied by the scaling factor
    """
    actual = fgeometric.keypoints_scale(np.array([keypoint]), scale, scale)
    np.testing.assert_allclose(actual, [expected], atol=1e-7)


def test_compose_with_additional_targets() -> None:
    image = np.ones((100, 100, 3))
    keypoints = [(10, 10), (50, 50)]
    kp1 = [(15, 15), (55, 55)]

    aug = A.Compose(
        [A.CenterCrop(50, 50, p=1)],
        keypoint_params={"format": "xy"},
        additional_targets={"kp1": "keypoints"},
        strict=True,
    )
    transformed = aug(image=image, keypoints=keypoints, kp1=kp1)
    np.testing.assert_array_equal(transformed["keypoints"], [(25, 25)])
    np.testing.assert_array_equal(transformed["kp1"], [(30, 30)])

    aug = A.Compose([A.CenterCrop(50, 50, p=1)], keypoint_params={"format": "xy"}, strict=True)
    aug.add_targets(additional_targets={"kp1": "keypoints"})
    transformed = aug(image=image, keypoints=keypoints, kp1=kp1)
    np.testing.assert_array_equal(transformed["keypoints"], [(25, 25)])
    np.testing.assert_array_equal(transformed["kp1"], [(30, 30)])


@pytest.mark.parametrize(
    ["angle", "expected"],
    [
        [0, 0],
        [np.pi / 2, np.pi / 2],
        [np.pi, np.pi],
        [3 * np.pi / 2, 3 * np.pi / 2],
        [2 * np.pi, 0],
        [-np.pi / 2, 3 * np.pi / 2],
        [-np.pi, np.pi],
        [-3 * np.pi / 2, np.pi / 2],
        [-2 * np.pi, 0],
    ],
)
def test_angle_to_2pi_range(angle, expected) -> None:
    assert np.isclose(angle_to_2pi_range(angle), expected)


@pytest.mark.parametrize(
    "keypoints, image_shape, expected",
    [
        (
            np.array([[0, 0], [50, 50], [100, 100], [-10, 50], [50, -10], [110, 50], [50, 110]]),
            (100, 100),
            np.array([[0, 0], [50, 50]]),
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [50, 50, 90],
                    [100, 100, 180],
                    [-10, 50, 45],
                    [50, -10, 135],
                    [110, 50, 270],
                    [50, 110, 315],
                ],
            ),
            (100, 100),
            np.array([[0, 0, 0], [50, 50, 90]]),
        ),
        (
            np.array([[10, 10], [20, 20]]),
            (30, 30),
            np.array([[10, 10], [20, 20]]),
        ),
    ],
)
def test_validate_keypoints(keypoints, image_shape, expected):
    result = fgeometric.validate_keypoints(keypoints, image_shape)
    np.testing.assert_array_almost_equal(result, expected)


def test_validate_keypoints_all_invalid():
    keypoints = np.array([[-1, -1], [101, 101]])
    result = fgeometric.validate_keypoints(keypoints, (100, 100))
    assert result.shape == (0, 2)


@pytest.mark.parametrize(
    "keypoint, rows, cols",
    [
        ((100, 150, 33, 0, 1), 300, 400),  # Example keypoint with arbitrary angle and scale
        ((200, 100, 33, np.pi / 4, 0.5), 300, 400),
        ((50, 250, 33, np.pi / 2, 2), 300, 400),
    ],
)
def test_keypoint_vh_flip_equivalence(keypoint, rows, cols):
    keypoints = np.array([keypoint])
    # Perform vertical and then horizontal flip
    hflipped_keypoints = fgeometric.keypoints_hflip(keypoints, cols)
    vhflipped_keypoints = fgeometric.keypoints_vflip(hflipped_keypoints, rows)

    vflipped_keypoints = fgeometric.keypoints_vflip(keypoints, rows)
    hvflipped_keypoints = fgeometric.keypoints_hflip(vflipped_keypoints, cols)

    assert vhflipped_keypoints == pytest.approx(
        hvflipped_keypoints,
    ), "Sequential vflip + hflip not equivalent to hflip + vflip"
    assert vhflipped_keypoints == pytest.approx(
        fgeometric.keypoints_rot90(keypoints, 2, (rows, cols)),
    ), "rot180 not equivalent to vflip + hflip"


def test_swap_tiles_on_keypoints_basic():
    keypoints = np.array([[10, 10], [30, 30], [50, 50]])
    tiles = np.array([[0, 0, 20, 20], [20, 20, 40, 40], [40, 40, 60, 60]])
    mapping = np.array([2, 1, 0])  # Swap first and last tile

    new_keypoints = fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    expected = np.array([[50, 50], [30, 30], [10, 10]])
    np.testing.assert_array_equal(new_keypoints, expected)


def test_swap_tiles_on_keypoints_no_change():
    keypoints = np.array([[10, 10], [30, 30], [50, 50]])
    tiles = np.array([[0, 0, 20, 20], [20, 20, 40, 40], [40, 40, 60, 60]])
    mapping = np.array([0, 1, 2])  # No swaps

    new_keypoints = fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    np.testing.assert_array_equal(new_keypoints, keypoints)


def test_swap_tiles_on_keypoints_out_of_bounds():
    keypoints = np.array([[10, 10], [30, 30], [70, 70]])
    tiles = np.array([[0, 0, 20, 20], [20, 20, 40, 40], [40, 40, 60, 60]])
    mapping = np.array([2, 1, 0])

    with pytest.warns(RuntimeWarning):
        new_keypoints = fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    expected = np.array([[50, 50], [30, 30], [70, 70]])  # Last keypoint unchanged
    np.testing.assert_array_equal(new_keypoints, expected)


def test_swap_tiles_on_keypoints_complex_mapping():
    keypoints = np.array([[5, 5], [15, 15], [25, 25], [35, 35]])
    tiles = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]])
    mapping = np.array([3, 2, 0, 1])

    new_keypoints = fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    expected = np.array([[35, 35], [25, 25], [5, 5], [15, 15]])
    np.testing.assert_array_equal(new_keypoints, expected)


def test_swap_tiles_on_keypoints_empty_input():
    keypoints = np.array([])
    tiles = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
    mapping = np.array([1, 0])

    new_keypoints = fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    assert new_keypoints.size == 0


def dummy_keypoint_func(keypoints):
    return keypoints


@pytest.mark.parametrize(
    "input_keypoints, expected_output",
    [
        (np.array([[1, 2, 33, 0], [3, 4, 33, np.pi], [5, 6, 33, 2 * np.pi]]), np.array([[1, 2, 33, 0], [3, 4, 33, np.pi], [5, 6, 33, 0]])),
        (np.array([[1, 2, 33, -np.pi], [3, 4, 33, 3 * np.pi]]), np.array([[1, 2, 33, np.pi], [3, 4, 33, np.pi]])),
        (np.array([]), np.array([])),
    ],
)
def test_angle_2pi_range(input_keypoints, expected_output):
    wrapped_func = angle_2pi_range(dummy_keypoint_func)
    result = wrapped_func(input_keypoints)
    np.testing.assert_allclose(result, expected_output, atol=1e-7)


@pytest.mark.parametrize(
    "input_keypoints",
    [
        np.array([[1, 2, 33, 0], [4, 5, 33, np.pi], [7, 8, 33, 2 * np.pi]]),
        np.array([]),
    ],
)
def test_angle_2pi_range_preserves_input(input_keypoints):
    def modify_input(keypoints):
        keypoints = keypoints.copy()
        if keypoints.size > 0 and keypoints.shape[1] > 2:
            keypoints[:, 3] += 1
        return keypoints

    wrapped_func = angle_2pi_range(modify_input)
    result = wrapped_func(input_keypoints)

    if input_keypoints.size > 0 and input_keypoints.shape[1] > 2:
        expected = input_keypoints.copy()
        expected[:, 3] = np.mod(expected[:, 3] + 1, 2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-7)
    else:
        np.testing.assert_array_equal(result, input_keypoints)


@pytest.mark.parametrize(
    "transform, params",
    [(A.CoarseDropout, {"hole_height_range": (98, 98), "hole_width_range": (98, 98)})],
)
def test_remove_invisible_keypoints_false(transform, params):
    image = RECTANGULAR_UINT8_IMAGE

    keypoints = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])

    aug = A.Compose([transform(**params, p=1)], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False), strict=True)
    result = aug(image=image, keypoints=keypoints)

    np.testing.assert_array_equal(result["keypoints"], keypoints)

    aug = A.Compose([transform(**params, p=1)], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True), strict=True)
    result = aug(image=image, keypoints=keypoints)

    assert len(result["keypoints"]) == 0


@pytest.mark.parametrize(
    "keypoints, image_shape, matrix, max_width, max_height, keep_size, expected",
    [
        # Test case 1: Identity transformation
        (
            np.array([[10, 10, 1, 0, 1], [50, 50, 1, np.pi / 4, 2]]),
            (100, 100),
            np.eye(3, dtype=np.float32),
            100,
            100,
            True,
            np.array([[10, 10, 1, 0, 1], [50, 50, 1, np.pi / 4, 2]]),
        ),
        # Test case 3: Scaling
        (
            np.array([[10, 10, 1, 0, 1], [50, 50, 1, 0, 1]]),
            (100, 100),
            np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float32),
            200,
            200,
            False,
            np.array([[20, 20, 1, 0, 2], [100, 100, 1, 0, 2]]),
        ),
        # Test case 4: Single keypoint
        (
            np.array([[10, 10, 1, 0, 1]]),
            (100, 100),
            np.eye(3, dtype=np.float32),
            100,
            100,
            True,
            np.array([[10, 10, 1, 0, 1]]),
        ),
    ],
)
def test_perspective_keypoints(keypoints, image_shape, matrix, max_width, max_height, keep_size, expected):
    result = fgeometric.perspective_keypoints(keypoints, image_shape, matrix, max_width, max_height, keep_size)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_perspective_keypoints_empty():
    result = fgeometric.perspective_keypoints(np.array([]), (100, 100), np.eye(3, dtype=np.float32), 100, 100, True)
    assert result.size == 0


@pytest.mark.parametrize(
    "input_angle, expected_angle",
    [
        (3 * np.pi, np.pi),
        (-np.pi / 2, 3 * np.pi / 2),
        (5 * np.pi / 2, np.pi / 2),
    ],
)
def test_perspective_keypoints_angle_wrapping(input_angle, expected_angle):
    keypoints = np.array([[0.5, 0.5, 1, input_angle, 1]])
    result = fgeometric.perspective_keypoints(keypoints, (100, 100), np.eye(3, dtype=np.float32), 100, 100, True)
    np.testing.assert_allclose(result[0, 3], expected_angle, atol=1e-6)


def test_crop_keypoints():
    image = np.random.randint(0, 256, (100, 100, 1), np.uint8)
    keypoints = np.array([(50, 50, 1, 0, 0)])

    aug = A.Crop(0, 0, 80, 80, p=1)
    result = aug(image=image, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], keypoints)

    aug = A.Crop(50, 50, 100, 100, p=1)
    result = aug(image=image, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(0, 0, 1, 0, 0)])


def test_longest_max_size_keypoints():
    img = np.random.randint(0, 256, (50, 10, 1), np.uint8)
    keypoints = np.array([(9, 5, 1, 0, 0)])

    aug = A.LongestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_almost_equal(
        result["keypoints"], [(18, 10, 1, 0, 0)], decimal=5
    )

    aug = A.LongestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_almost_equal(
        result["keypoints"], [(0.9, 0.5, 1, 0, 0)], decimal=5
    )

    aug = A.LongestMaxSize(max_size=50, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(9, 5, 1, 0, 0)])


def test_smallest_max_size_keypoints():
    img = np.random.randint(0, 256, (50, 10, 1), np.uint8)
    keypoints = np.array([(9, 5, 1, 0, 0)])

    aug = A.SmallestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(90, 50, 1, 0, 0)])

    aug = A.SmallestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(4.5, 2.5, 1, 0, 0)])

    aug = A.SmallestMaxSize(max_size=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(9, 5, 1, 0, 0)])


def test_resize_keypoints():
    img = np.random.randint(0, 256, (50, 10, 1), np.uint8)
    keypoints = np.array([(9, 5, 1, 0, 0)])

    aug = A.Resize(height=100, width=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(4.5, 10, 1, 0, 0)])

    aug = A.Resize(height=50, width=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    np.testing.assert_array_equal(result["keypoints"], [(9, 5, 1, 0, 0)])



@pytest.mark.parametrize(
    "image_shape, keypoints, distortion_type",
    [
        # Test case 1: Simple translation
        (
            (100, 100),
            np.array([[50, 50, 1], [25, 75, 1], [75, 25, 1]]),
            "translation"
        ),
        # Test case 2: Single point
        (
            (100, 100),
            np.array([[10, 10, 1]]),
            "identity"
        ),
        # Test case 3: Points with extra attributes
        (
            (100, 100),
            np.array([[50, 50, 1, 0, 2], [25, 75, 1, np.pi/4, 1]]),
            "wave"
        ),
        # Test case 4: Points near borders
        (
            (100, 100),
            np.array([[0, 0, 1], [99, 99, 1], [0, 99, 1], [99, 0, 1]]),
            "scale"
        ),
        # Test case 5: Larger image
        (
            (512, 512),
            np.array([[256, 256, 1], [128, 384, 1], [384, 128, 1]]),
            "rotation"
        ),
    ]
)
def test_keypoint_remap_methods(image_shape, keypoints, distortion_type):
    """Test that both keypoint remapping methods produce similar results."""

    # Generate distortion maps based on type
    h, w = image_shape
    map_x = np.zeros(image_shape, dtype=np.float32)
    map_y = np.zeros(image_shape, dtype=np.float32)

    # Create meshgrid in correct order
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    if distortion_type == "identity":
        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)

    elif distortion_type == "translation":
        map_x = x.astype(np.float32) + 10
        map_y = y.astype(np.float32) + 10

    elif distortion_type == "wave":
        map_x = x + np.sin(y/10.0) * 5  # y affects x-displacement
        map_y = y + np.sin(x/10.0) * 5  # x affects y-displacement
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

    elif distortion_type == "scale":
        scale = 1.5
        map_x = x.astype(np.float32) * scale
        map_y = y.astype(np.float32) * scale

    elif distortion_type == "rotation":
        center = (w/2, h/2)
        angle = 45
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation to meshgrid
        pts = np.column_stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))
        transformed = M.dot(pts.T).T
        map_x = transformed[:, 0].reshape(h, w).astype(np.float32)
        map_y = transformed[:, 1].reshape(h, w).astype(np.float32)

    # Get results from both methods
    result_direct = fgeometric.remap_keypoints(keypoints, map_x, map_y, image_shape)
    result_mask = fgeometric.remap_keypoints_via_mask(keypoints, map_x, map_y, image_shape)

    # Compare results
    assert len(result_direct) == len(result_mask), "Methods returned different numbers of keypoints"

    # Compare coordinates with tolerance
    np.testing.assert_allclose(
        result_direct[:, :2],  # only x,y coordinates
        result_mask[:, :2],
        rtol=1e-3,  # relative tolerance
        atol=2.0,   # absolute tolerance (1 pixel)
        err_msg=f"Methods produced different results for {distortion_type} distortion"
    )

    # Compare extra attributes if present
    if keypoints.shape[1] > 2:
        np.testing.assert_allclose(
            result_direct[:, 2:],
            result_mask[:, 2:],
            rtol=1e-3,
            atol=1e-3,
            err_msg="Methods produced different results for extra attributes"
        )

    # Additional checks for specific distortion types
    if distortion_type == "identity":
        np.testing.assert_allclose(
            result_direct,
            keypoints,
            rtol=1e-3,
            atol=1.0,
            err_msg="Identity transformation changed keypoint positions"
        )
