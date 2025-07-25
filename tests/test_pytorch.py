import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ColorJitter

import albumentations as A
from tests.conftest import RECTANGULAR_UINT8_IMAGE, SQUARE_UINT8_IMAGE, UINT8_IMAGES


@pytest.mark.parametrize("image", UINT8_IMAGES)
def test_torch_to_tensor_v2_augmentations(image):
    mask = image.copy()
    aug = A.ToTensorV2()
    data = aug(image=image, mask=mask, force_apply=True)
    height, width, num_channels = image.shape
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (num_channels, height, width)
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == mask.shape
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


@pytest.mark.parametrize("image", UINT8_IMAGES)
def test_torch_to_tensor_v2_augmentations_with_transpose_2d_mask(image):
    mask = image[:, :, 0].copy()

    aug = A.ToTensorV2(transpose_mask=True)

    data = aug(image=image, mask=mask, force_apply=True)
    image_height, image_width, image_num_channels = image.shape
    mask_height, mask_width = mask.shape

    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        image_num_channels,
        image_height,
        image_width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == (mask_height, mask_width)
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


@pytest.mark.parametrize("image", UINT8_IMAGES)
def test_torch_to_tensor_v2_augmentations_with_transpose_3d_mask(image):
    aug = A.ToTensorV2(transpose_mask=True)
    mask_shape = image.shape[:2] + (4,)
    mask = np.random.randint(low=0, high=256, size=mask_shape, dtype=np.uint8)
    data = aug(image=image, mask=mask, force_apply=True)
    image_height, image_width, image_num_channels = image.shape
    mask_height, mask_width, mask_num_channels = mask.shape
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        image_num_channels,
        image_height,
        image_width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == (
        mask_num_channels,
        mask_height,
        mask_width,
    )
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


def test_additional_targets_for_totensorv2():
    aug = A.Compose([A.ToTensorV2()], additional_targets={"image2": "image", "mask2": "mask"}, strict=True)
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(100, 100, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)

        image1_height, image1_width, image1_num_channels = image1.shape
        image2_height, image2_width, image2_num_channels = image2.shape
        assert isinstance(res["image"], torch.Tensor) and res["image"].shape == (
            image1_num_channels,
            image1_height,
            image1_width,
        )
        assert isinstance(res["image2"], torch.Tensor) and res["image2"].shape == (
            image2_num_channels,
            image2_height,
            image2_width,
        )
        assert isinstance(res["mask"], torch.Tensor) and res["mask"].shape == mask1.shape
        assert isinstance(res["mask2"], torch.Tensor) and res["mask2"].shape == mask2.shape
        assert np.array_equal(res["image"], res["image2"])
        assert np.array_equal(res["mask"], res["mask2"])

    aug = A.Compose([A.ToTensorV2()], strict=True)
    aug.add_targets(additional_targets={"image2": "image", "mask2": "mask"})
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(100, 100, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)

        image1_height, image1_width, image1_num_channels = image1.shape
        image2_height, image2_width, image2_num_channels = image2.shape
        assert isinstance(res["image"], torch.Tensor) and res["image"].shape == (
            image1_num_channels,
            image1_height,
            image1_width,
        )
        assert isinstance(res["image2"], torch.Tensor) and res["image2"].shape == (
            image2_num_channels,
            image2_height,
            image2_width,
        )
        assert isinstance(res["mask"], torch.Tensor) and res["mask"].shape == mask1.shape
        assert isinstance(res["mask2"], torch.Tensor) and res["mask2"].shape == mask2.shape
        assert np.array_equal(res["image"], res["image2"])
        assert np.array_equal(res["mask"], res["mask2"])


def test_torch_to_tensor_v2_on_gray_scale_images():
    aug = A.ToTensorV2()
    grayscale_image = np.random.randint(low=0, high=256, size=(100, 100), dtype=np.uint8)
    data = aug(image=grayscale_image)
    assert isinstance(data["image"], torch.Tensor)
    assert len(data["image"].shape) == 3
    assert data["image"].shape[1:] == grayscale_image.shape
    assert data["image"].dtype == torch.uint8


def test_with_replaycompose() -> None:
    aug = A.ReplayCompose([A.ToTensorV2()])
    kwargs = {
        "image": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
        "mask": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
    }
    res = aug(**kwargs)

    res2 = A.ReplayCompose.replay(res["replay"], **kwargs)
    np.testing.assert_array_equal(res["image"], res2["image"])
    np.testing.assert_array_equal(res["mask"], res2["mask"])

    assert res["image"].dtype == torch.uint8
    assert res["mask"].dtype == torch.uint8
    assert res2["image"].dtype == torch.uint8
    assert res2["mask"].dtype == torch.uint8


@pytest.mark.parametrize(
    ["brightness", "contrast", "saturation", "hue"],
    [
        [1, 1, 1, 0],
        [0.123, 1, 1, 0],
        [1.321, 1, 1, 0],
        [1, 0.234, 1, 0],
        [1, 1.432, 1, 0],
        [1, 1, 0.345, 0],
        [1, 1, 1.543, 0],
    ],
)
def test_color_jitter(brightness, contrast, saturation, hue):
    img = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
    pil_image = Image.fromarray(img)

    transform = A.Compose(
        [
            A.ColorJitter(
                brightness=[brightness, brightness],
                contrast=[contrast, contrast],
                saturation=[saturation, saturation],
                hue=[hue, hue],
                p=1,
            ),
        ],
        strict=True,
    )

    pil_transform = ColorJitter(
        brightness=[brightness, brightness],
        contrast=[contrast, contrast],
        saturation=[saturation, saturation],
        hue=[hue, hue],
    )

    res1 = transform(image=img)["image"]
    res2 = np.array(pil_transform(pil_image))

    assert np.abs(res1.astype(np.int16) - res2.astype(np.int16)).max() <= 2


def test_post_data_check():
    img = np.empty([100, 100, 3], dtype=np.uint8)
    bboxes = np.array([
        [0, 0, 90, 90, 0],
    ])
    keypoints = np.array([
        [90, 90],
        [50, 50],
    ])

    transform = A.Compose(
        [
            A.Resize(50, 50, p=1),
            A.Normalize(p=1),
            A.ToTensorV2(p=1),
        ],
        keypoint_params=A.KeypointParams("xy"),
        bbox_params=A.BboxParams("pascal_voc"),
        seed=137,
        strict=True,
    )

    res = transform(image=img, keypoints=keypoints, bboxes=bboxes)
    np.testing.assert_array_equal(res["keypoints"], [(45, 45), (25, 25)])
    # Use assert_allclose instead of assert_array_equal
    np.testing.assert_allclose(res["bboxes"], [(0, 0, 45, 45, 0)], rtol=1e-5, atol=1e-5)


def test_to_tensor_v2_on_non_contiguous_array():
    # Create a contiguous array
    img = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
    assert img.flags["C_CONTIGUOUS"]

    # Create a non-contiguous array by slicing
    non_contiguous_img = img[::2, ::2, :]
    assert not non_contiguous_img.flags["C_CONTIGUOUS"]

    transform = A.Compose([A.ToTensorV2()], strict=True)
    transformed = transform(image=non_contiguous_img, masks=np.stack([non_contiguous_img] * 2))

    # Additional checks to ensure the transformation worked correctly
    assert isinstance(transformed["image"], torch.Tensor)
    assert transformed["image"].shape == (3, 50, 50)  # Shape changed due to slicing
    assert transformed["image"].dtype == torch.uint8

    # Check masks
    assert isinstance(transformed["masks"], torch.Tensor)
    assert transformed["masks"].shape == (2, 50, 50, 3)  # (N, H, W, C) - masks remain in original format
    assert transformed["masks"].dtype == torch.uint8

    # Optional: Check that the content is correct
    np_transformed = transformed["image"].numpy()
    np.testing.assert_array_equal(np_transformed.transpose(1, 2, 0), non_contiguous_img)


def test_to_tensor_v2_on_non_contiguous_array_with_horizontal_flip():
    transform = A.Compose(
        [
            A.HorizontalFlip(p=1),
            A.ToFloat(max_value=255),
            A.ToTensorV2(),
        ],
        is_check_shapes=False,
        strict=True,
    )

    image = RECTANGULAR_UINT8_IMAGE

    masks = np.stack([image[:, :, 0]] * 2)

    transform(image=image, masks=masks)


def test_to_tensor_v2_on_non_contiguous_array_with_random_rotate90():
    transforms = A.Compose(
        [
            A.RandomRotate90(p=1.0),
            A.ToTensorV2(),
        ],
        strict=True,
    )

    img = np.random.randint(0, 256, (640, 480, 3)).astype(np.uint8)
    masks = np.stack([np.random.randint(0, 2, (640, 480)).astype(np.uint8)] * 4)
    for _ in range(10):
        transformed = transforms(image=img, masks=masks)
        assert isinstance(transformed["image"], torch.Tensor)
        assert isinstance(transformed["masks"][0], torch.Tensor)
        assert transformed["image"].numpy().shape in ((3, 640, 480), (3, 480, 640))
        assert transformed["masks"][0].shape in ((640, 480), (480, 640))


def test_to_tensor_v2_images_masks():
    transform = A.Compose([A.ToTensorV2(p=1)], strict=True)
    image = SQUARE_UINT8_IMAGE
    mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

    transformed = transform(
        image=image,
        mask=mask,
        masks=np.stack([mask] * 2),  # Now passing stacked numpy array
        images=np.stack([image] * 2)  # Stacked numpy array
    )

    # Check outputs are torch.Tensor
    assert isinstance(transformed["image"], torch.Tensor)
    assert isinstance(transformed["mask"], torch.Tensor)
    assert isinstance(transformed["masks"], torch.Tensor)
    assert isinstance(transformed["images"], torch.Tensor)  # Now checking single tensor

    # Check shapes
    assert transformed["image"].shape == (3, 100, 100)  # (C, H, W)
    assert transformed["mask"].shape == (100, 100)  # (H, W)
    assert transformed["masks"].shape == (2, 100, 100)  # (N, H, W)
    assert transformed["images"].shape == (2, 3, 100, 100)  # (N, C, H, W)
