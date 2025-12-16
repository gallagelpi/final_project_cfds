import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from best_library.preprocessing.preprocessing import build_transform, Preprocessing


# ---------- Helpers ----------

def dummy_image(size=(300, 300)):
    """Create a dummy RGB PIL image."""
    array = (np.random.rand(*size, 3) * 255).astype(np.uint8)
    return Image.fromarray(array)


# ---------- build_transform tests ----------

def test_build_transform_returns_compose():
    transform = build_transform()
    assert isinstance(transform, transforms.Compose)


def test_build_transform_contains_resize():
    transform = build_transform()
    resize = transform.transforms[0]
    assert isinstance(resize, transforms.Resize)


def test_build_transform_contains_to_tensor():
    transform = build_transform()
    assert any(isinstance(t, transforms.ToTensor) for t in transform.transforms)


def test_build_transform_contains_normalize():
    transform = build_transform()
    assert any(isinstance(t, transforms.Normalize) for t in transform.transforms)


def test_build_transform_output_shape():
    transform = build_transform(img_size=224)
    img = dummy_image()
    tensor = transform(img)

    assert tensor.shape == (3, 224, 224)


def test_build_transform_output_type():
    transform = build_transform()
    img = dummy_image()
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)


# ---------- Preprocessing class tests ----------

def test_preprocessing_get_transform_returns_compose():
    prep = Preprocessing()
    transform = prep.get_transform()
    assert isinstance(transform, transforms.Compose)


def test_preprocessing_respects_img_size():
    img_size = 128
    prep = Preprocessing(img_size=img_size)
    transform = prep.get_transform()

    img = dummy_image()
    tensor = transform(img)

    assert tensor.shape == (3, img_size, img_size)


def test_preprocessing_normalization_applied():
    prep = Preprocessing()
    transform = prep.get_transform()

    img = dummy_image()
    tensor = transform(img)

    # After normalization, values shouldn't be in [0, 1]
    assert tensor.min() < 0 or tensor.max() > 1
