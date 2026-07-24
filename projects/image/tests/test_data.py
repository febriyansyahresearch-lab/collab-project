import pytest
import numpy as np
from projects.image.src.data import generate_images


def test_generate_images_shape():
    X, y, classes = generate_images(150)
    assert X.shape[0] == 150


def test_generate_images_classes():
    X, y, classes = generate_images(200)
    assert len(set(y)) == 3


def test_generate_images_pixel_range():
    X, y, classes = generate_images(100)
    assert X.min() >= 0.0 and X.max() <= 1.0
