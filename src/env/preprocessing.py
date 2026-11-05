"""Preprocesado de fotogramas y apilado temporal."""

from __future__ import annotations

from collections import deque

import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    size: tuple[int, int] = (84, 84),
    grayscale: bool = True,
) -> np.ndarray:
    """Convierte la pantalla de PyBoy (H, W, 4) en una observación (C, H, W) uint8.

    El redimensionado usa promedio de área (no vecino más cercano): la pantalla
    de Game Boy es 160x144 y Kirby mide ~16 px, así que descartar píxeles puede
    borrar proyectiles y enemigos pequeños.
    """
    if frame.ndim == 3:
        rgb = frame[..., :3].astype(np.uint8)
    elif frame.ndim == 2:
        rgb = np.repeat(frame[..., None], 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError(f"Fotograma con forma inesperada: {frame.shape}")

    if grayscale:
        # Luminancia ITU-R BT.601
        gray = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
        resized = area_resize(gray, size)
        return resized[None, ...].astype(np.uint8)

    resized = np.stack(
        [area_resize(rgb[..., c], size) for c in range(3)], axis=0
    )
    return resized.astype(np.uint8)


def area_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona promediando el área exacta de cada celda destino.

    Usa la imagen integral con interpolación lineal en las fronteras
    fraccionarias, así que no depende de librerías externas y preserva mejor
    los objetos pequeños que el vecino más cercano.
    """
    out_h, out_w = int(size[0]), int(size[1])
    src = image.astype(np.float64)
    src_h, src_w = src.shape[:2]

    if src_h == 0 or src_w == 0:
        raise ValueError("Imagen de entrada vacía")

    # ii[y, x] = suma del rectángulo [0:y, 0:x] en coordenadas enteras
    ii = np.pad(src.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    ys = np.linspace(0.0, src_h, out_h + 1)
    xs = np.linspace(0.0, src_w, out_w + 1)

    y0 = np.clip(np.floor(ys).astype(np.int64), 0, src_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, src_w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    rows = ii[y0] * (1.0 - fy) + ii[y0 + 1] * fy          # (out_h+1, src_w+1)
    cols = rows[:, x0] * (1.0 - fx) + rows[:, x0 + 1] * fx  # (out_h+1, out_w+1)

    sums = (
        cols[1:, 1:] - cols[:-1, 1:] - cols[1:, :-1] + cols[:-1, :-1]
    )
    areas = np.outer(np.diff(ys), np.diff(xs))
    return np.clip(np.rint(sums / areas), 0, 255).astype(np.uint8)


class FrameStack:
    """Pila de los últimos `n` fotogramas preprocesados."""

    def __init__(self, n: int, shape: tuple[int, int, int]) -> None:
        self.n = n
        self.shape = shape
        self._frames: deque[np.ndarray] = deque(maxlen=n)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        for _ in range(self.n):
            self._frames.append(frame)
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._frames.append(frame)
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames), axis=0)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.shape[0] * self.n, self.shape[1], self.shape[2])
