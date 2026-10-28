"""
Unit tests of the Halide pipelines against numpy references, on random data.
Widths are not multiples of the vector size, to test the vectorization tails (in-place updates).
"""

import numpy as np
import pytest

from photometricStereoHalide import halidePipelines as hp

HEIGHT, WIDTH = 7, 37


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def randomNormals(rng, height=HEIGHT, width=WIDTH):
    normals = rng.normal(size=(3, height, width)).astype(np.float32)
    return np.ascontiguousarray(normals / np.linalg.norm(normals, axis=0))


@pytest.mark.parametrize("withAmbient", [False, True], ids=["noAmbient", "ambient"])
def test_leastSquares(rng, withAmbient):
    nbImages = 6
    lights = rng.normal(size=(nbImages, 3)).astype(np.float32)
    intensities = rng.uniform(0.5, 1.5, size=(nbImages, 3)).astype(np.float32)
    images = rng.uniform(0, 1, size=(nbImages, 3, HEIGHT, WIDTH)).astype(np.float32)
    ambient = rng.uniform(0, 0.1, size=(3, HEIGHT, WIDTH)).astype(np.float32) if withAmbient else None
    mask = (rng.uniform(size=(HEIGHT, WIDTH)) > 0.3).astype(np.float32)

    pinv = np.linalg.pinv(lights.astype(np.float64)).astype(np.float32)
    acc = np.zeros((9, HEIGHT, WIDTH), dtype=np.float32)
    accumulate = hp.getPipeline(hp.AccumulatePipeline)
    for i in range(nbImages):
        accumulate(acc, images[i], ambient, intensities[i], pinv[:, i])
    normals = np.empty((3, HEIGHT, WIDTH), dtype=np.float32)
    albedo = np.empty((3, HEIGHT, WIDTH), dtype=np.float32)
    hp.getPipeline(hp.FinalizePipeline)(acc, mask, normals, albedo)

    # Numpy reference: per channel least squares solution of L.m = I
    scaled = images - (ambient if withAmbient else 0)
    scaled = scaled / intensities[:, :, None, None]
    pixels = scaled.reshape(nbImages, 3, -1)
    solutions = np.stack([np.linalg.lstsq(lights, pixels[:, c], rcond=None)[0] for c in range(3)])  # (c, j, pixels)
    gray = np.tensordot(np.array(hp.LUMINANCE), solutions, axes=1)
    expectedNormals = (gray / np.linalg.norm(gray, axis=0)).reshape(3, HEIGHT, WIDTH) * mask
    expectedAlbedo = np.linalg.norm(solutions, axis=1).reshape(3, HEIGHT, WIDTH) * mask

    assert np.allclose(normals, expectedNormals, atol=1e-5)
    assert np.allclose(albedo, expectedAlbedo, rtol=1e-4, atol=1e-5)


def test_normalize(rng):
    vectors = rng.normal(size=(3, HEIGHT, WIDTH)).astype(np.float32) * 5
    vectors[:, 0, 0] = 0
    mask = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    mask[1, :] = 0
    out = np.empty_like(vectors)
    hp.getPipeline(hp.NormalizePipeline)(vectors, mask, out)

    expected = vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-30) * mask
    assert np.allclose(out, expected, atol=1e-6)
    assert np.all(out[:, 0, 0] == 0)


def test_rotate(rng):
    normals = randomNormals(rng)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out = np.empty_like(normals)
    hp.getPipeline(hp.RotatePipeline)(normals, rotation, out)
    assert np.allclose(out, np.einsum("ij,jhw->ihw", rotation, normals), atol=1e-6)


def test_normalsToPng(rng):
    normals = randomNormals(rng)
    normals[:, 0, 0] = 0  # Null normal
    normals[:, 0, 1] = [1, -1, 0]  # Extreme values
    normals[:, 0, 2] = [-1, 1, -1]
    normals = np.ascontiguousarray(normals)
    out = np.empty(normals.shape, dtype=np.uint8)
    hp.getPipeline(hp.NormalsToPngPipeline)(normals, out)

    # Reference: AliceVision's convertNormalMap2png, negative values wrapping around in uint8
    n = normals.astype(np.float64)
    expected = np.stack([np.floor(255.0 * (n[0] + 1.0) / 2.0),
                         -np.ceil(255.0 * (n[1] + 1.0) / 2.0),
                         -np.ceil(255.0 * n[2])]).astype(np.int64) % 256
    expected[:, np.sum(n * n, axis=0) == 0] = 0
    assert np.array_equal(out, expected.astype(np.uint8))


def test_robustIterations(rng):
    """ The ADMM iterations converge to the true solution despite sparse outliers, unlike least squares. """
    nbImages = 12
    lights = rng.normal(size=(nbImages, 3)).astype(np.float32)
    truth = rng.normal(size=(3, HEIGHT, WIDTH)).astype(np.float32)
    gray = np.einsum("nj,jhw->nhw", lights, truth)
    outliers = rng.uniform(size=gray.shape) < 0.08
    gray = np.ascontiguousarray(np.where(outliers, gray + rng.uniform(2, 5, size=gray.shape), gray), dtype=np.float32)

    pinv = np.ascontiguousarray(np.linalg.pinv(lights.astype(np.float64)).astype(np.float32))
    solution = np.zeros((3, HEIGHT, WIDTH), dtype=np.float32)
    errors = np.ascontiguousarray(-gray)
    multipliers = np.zeros_like(gray)
    mu = 0.1
    for _ in range(1000):
        hp.getPipeline(hp.RobustSolvePipeline)(gray, errors, multipliers, pinv, mu, solution)
        hp.getPipeline(hp.RobustUpdatePipeline)(gray, multipliers, solution, lights, mu, errors)

    leastSquares = np.einsum("jn,nhw->jhw", pinv, gray)
    robustError = np.median(np.abs(solution - truth))
    assert robustError < 0.1 * np.median(np.abs(leastSquares - truth))
    assert robustError < 0.05
