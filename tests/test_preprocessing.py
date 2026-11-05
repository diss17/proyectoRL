"""Tests del preprocesado de fotogramas."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.preprocessing import FrameStack, area_resize, preprocess_frame


def test_preprocess_shapes_and_dtype() -> None:
    frame = np.zeros((144, 160, 4), dtype=np.uint8)
    obs = preprocess_frame(frame, size=(84, 84), grayscale=True)
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.uint8


def test_constant_image_stays_constant() -> None:
    frame = np.full((144, 160, 4), 200, dtype=np.uint8)
    frame[..., 3] = 255
    obs = preprocess_frame(frame, size=(84, 84), grayscale=True)
    assert np.all(obs == 200)


def test_small_sprite_survives_downscaling() -> None:
    """Un sprite de 2x2 no debe desaparecer al pasar de 160x144 a 84x84."""
    frame = np.zeros((144, 160, 4), dtype=np.uint8)
    frame[70:72, 80:82, :3] = 255

    nearest = preprocess_frame(frame, size=(84, 84), grayscale=True)
    assert nearest.max() > 0

    area = area_resize(np.full((144, 160), 0, dtype=np.uint8), (84, 84))
    assert area.shape == (84, 84)


def test_area_resize_preserves_total_intensity() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(144, 160), dtype=np.uint8)

    resized = area_resize(image, (72, 80))
    assert resized.shape == (72, 80)
    # El promedio de área conserva aproximadamente la media global.
    assert abs(float(resized.mean()) - float(image.mean())) < 1.5


def test_area_resize_upscales() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[4:6, 4:6] = 255
    assert area_resize(image, (20, 20)).shape == (20, 20)


def test_frame_stack() -> None:
    stack = FrameStack(n=4, shape=(1, 84, 84))
    frame = np.ones((1, 84, 84), dtype=np.uint8)

    obs = stack.reset(frame)
    assert obs.shape == (4, 84, 84)
    assert stack.observation_shape == (4, 84, 84)

    obs = stack.push(frame * 2)
    assert obs.shape == (4, 84, 84)
    assert np.all(obs[-1] == 2)


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        area_resize(np.zeros((0, 10), dtype=np.uint8), (5, 5))
