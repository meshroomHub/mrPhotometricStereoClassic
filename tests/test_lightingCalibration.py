"""
Tests of the lighting calibration: image reading, accuracy of each method against the ground truth lights, parity of
'brightestPoint' with aliceVision_lightingCalibration, and the full chain of Python nodes from the detected spheres.
"""

import json
import os
import shutil

import numpy as np
import OpenImageIO as oiio
import pytest
from pyalicevision import image as avimage

from conftest import angularError, lightErrors, readImage, readLights
from makeSyntheticDataset import SPHERE_ALBEDO, perspectiveSphere, runCommand

from photometricStereoHalide import dataIO
from photometricStereoHalide import lightingCalibration as lc


@pytest.fixture(scope="module")
def imagePath(dataset):
    sfm = dataIO.loadSfmData(dataset.sfmPath)
    return next(v.getImage().getImagePath() for v in sfm.getViews().values() if not dataIO.isAmbient(v.getImage().getImagePath()))


def runPython(dataset, tmp_path, method, detection=None, **kwargs):
    outputPath = tmp_path / f"lights_{method}.json"
    lc.lightingCalibration(dataset.sfmPath, detection or dataset.spheresPath, str(outputPath), method, **kwargs)
    return outputPath


def test_readLuminance(dataset, imagePath):
    """ The region read matches AliceVision's grayscale reading, with zeros outside of the image. """
    img = avimage.Image_float()
    avimage.readImage(imagePath, img, avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION))
    full = np.array(img.getNumpyArray()).reshape(img.height(), img.width())

    region, width, height = lc.readLuminance(imagePath, 10, 20, 110, 90)
    assert (width, height) == (dataset.width, dataset.height)
    assert np.array_equal(region, full[20:90, 10:110])

    region, _, _ = lc.readLuminance(imagePath, -5, -3, 15, 7)
    assert np.all(region[:3] == 0) and np.all(region[:, :5] == 0)
    assert np.array_equal(region[3:, 5:], full[0:7, 0:15])


def test_separableConvolution():
    rng = np.random.default_rng(0)
    image = rng.random((9, 13)).astype(np.float32)
    kernel = lc.triangleKernel(2)
    assert np.allclose(kernel, [0.5, 1.0, 1.5, 1.0, 0.5])

    # Reference: 2D correlation with the borders extended
    padded = np.pad(image, 2, mode="edge")
    expected = sum(kernel[a] * kernel[b] * padded[a:a + 9, b:b + 13] for a in range(5) for b in range(5))
    assert np.allclose(lc.separableConvolution(image, kernel), expected, atol=1e-5)


@pytest.mark.parametrize("method", lc.METHODS)
def test_groundTruthSpheres(dataset, tmp_path, method):
    """ With the exact spheres, every method recovers the lights (the specular highlight slightly biases the fits). """
    outputPath = runPython(dataset, tmp_path, method, visualizationFolder=str(tmp_path))
    errors = lightErrors(outputPath, dataset.lightsPath)
    assert len(errors) == len(readLights(dataset.lightsPath))
    assert np.median(list(errors.values())) < 2.5
    assert np.max(list(errors.values())) < 4

    lights = readLights(outputPath)
    for viewId, light in lights.items():
        assert light["type"] == ("SH" if method == "SH" else "directional")
        assert len(light["direction"]) == (9 if method == "SH" else 3)
        assert np.linalg.norm(light["direction"][:3]) == pytest.approx(1, abs=1e-6)
    # One visualization per image file name (as in AliceVision, images with the same name in different folders share it)
    sfm = dataIO.loadSfmData(dataset.sfmPath)
    stems = {os.path.splitext(os.path.basename(sfm.getViews()[v].getImage().getImagePath()))[0] for v in lights}
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(f"_{method}.png")) == sorted(f"{s}_{method}.png" for s in stems)


@pytest.mark.parametrize("method", ["whiteSphere", "SH"])
def test_intensities(dataset, tmp_path, method):
    """
    The fitted intensity is proportional to the luminance of the light intensity (the photometric stereo only uses
    relative intensities). The factor is the sphere albedo, plus a few percents of specular highlight.
    """
    lights = readLights(runPython(dataset, tmp_path, method))
    groundTruth = readLights(dataset.lightsPath)
    ratios = []
    for viewId, light in lights.items():
        assert light["intensity"] == [light["intensity"][0]] * 3
        expected = SPHERE_ALBEDO * np.dot(lc.LUMINANCE, np.array(groundTruth[viewId]["intensity"], dtype=float))
        ratios.append(light["intensity"][0] / expected)
    assert max(ratios) / min(ratios) < 1.03
    assert np.mean(ratios) == pytest.approx(1, rel=0.1)


def test_saveAsModel(dataset, tmp_path, runHalide, monkeypatch):
    """
    Lights saved as a model are ordered like the images (by path: pose A then pose B), and are applied by position:
    every pose uses the first lights, i.e. the ones calibrated on pose A.
    """
    byView = runPython(dataset, tmp_path, "whiteSphere")
    asModel = tmp_path / "model.json"
    lc.lightingCalibration(dataset.sfmPath, dataset.spheresPath, str(asModel), "whiteSphere", saveAsModel=True)
    with open(asModel) as f:
        modelLights = json.load(f)["lights"]
    assert [light["lightId"] for light in modelLights] == list(range(len(modelLights)))

    outputs = {}
    for name, path in [("byView", byView), ("asModel", asModel)]:
        monkeypatch.setattr(dataset, "lightsPath", str(path))
        outputs[name] = runHalide(name=name)
    normals = {name: {pose: readImage(output / f"{poseId}_normals.exr") for pose, poseId in dataset.poseIds.items()}
               for name, output in outputs.items()}
    assert np.array_equal(normals["byView"]["A"], normals["asModel"]["A"])
    # Same physical lights, calibrated on the other sphere
    assert np.median(angularError(normals["byView"]["B"], normals["asModel"]["B"])) < 0.5


def test_missingSphereAndUnusablePixels(dataset, tmp_path, imagePath):
    """ Views without sphere or without usable pixels are skipped. """
    with open(dataset.spheresPath) as f:
        spheres = json.load(f)
    observations = spheres["shapes"][0]["observations"]
    viewIds = sorted(int(v) for v in readLights(dataset.lightsPath))
    del observations[str(viewIds[0])]
    # Sphere smaller than a pixel: not enough pixels to fit the light
    observations[str(viewIds[1])] = {"center": {"x": 2.0, "y": 2.0}, "radius": 0.6}
    detectionPath = tmp_path / "spheres.json"
    detectionPath.write_text(json.dumps(spheres))

    lights = readLights(runPython(dataset, tmp_path, "whiteSphere", detection=str(detectionPath)))
    assert viewIds[0] not in lights and viewIds[1] not in lights
    assert len(lights) == len(viewIds) - 2


def test_unknownMethod(dataset, imagePath):
    with pytest.raises(ValueError):
        lc.calibrateImage(imagePath, dataset.spheres["A"], "unknown")


def test_renderSphere():
    image = lc.renderSphere(np.array([0.0, 0.0, -1.0]), size=64)
    assert image.shape == (64, 64)
    assert image[0, 0] == 0  # Outside of the sphere
    assert image[32, 32] == pytest.approx(1.0)  # Normal facing the light
    sh = lc.renderSphere(np.array([0.0, 0.0, -1.0, 0.5, 0, 0, 0, 0, 0]), size=64)
    assert sh[32, 32] == pytest.approx((1.0 + 0.5) * np.sqrt(1.25))


def test_brightestPointParity(dataset, tmp_path):
    if not shutil.which("aliceVision_lightingCalibration"):
        pytest.skip("aliceVision_lightingCalibration is not available.")
    pythonFolder, aliceVisionFolder = tmp_path / "python", tmp_path / "aliceVision"
    pythonFolder.mkdir()
    aliceVisionFolder.mkdir()
    pythonPath = runPython(dataset, pythonFolder, "brightestPoint", visualizationFolder=str(pythonFolder))
    aliceVisionPath = aliceVisionFolder / "lights.json"
    runCommand(["aliceVision_lightingCalibration", "--inputPath", dataset.sfmPath, "--inputDetection", dataset.spheresPath,
                "--method", "brightestPoint", "--outputFile", str(aliceVisionPath), "--verboseLevel", "error"])

    with open(pythonPath) as f:
        pythonLights = json.load(f)["lights"]
    with open(aliceVisionPath) as f:
        aliceVisionLights = json.load(f)["lights"]
    # Same lights in the same order
    assert [light["viewId"] for light in pythonLights] == [int(light["viewId"]) for light in aliceVisionLights]
    for pythonLight, aliceVisionLight in zip(pythonLights, aliceVisionLights):
        assert np.allclose(pythonLight["direction"], np.array(aliceVisionLight["direction"], dtype=float), atol=1e-5)
        assert np.allclose(pythonLight["intensity"], np.array(aliceVisionLight["intensity"], dtype=float))

    # Same visualizations (up to the 8-bit rounding)
    for name in os.listdir(aliceVisionFolder):
        if name.endswith(".png"):
            expected = oiio.ImageBuf(str(aliceVisionFolder / name)).get_pixels(oiio.UINT8).astype(int)
            assert np.abs(readImage(pythonFolder / name).astype(int) - expected).max() <= 1, name


@pytest.mark.parametrize("method", ["brightestPoint", "whiteSphere"])
def test_fullChain(dataset, autoDetection, runHalide, tmp_path, monkeypatch, method):
    """ Detected spheres -> lighting calibration -> photometric stereo, all with the Python nodes. """
    lightsPath = runPython(dataset, tmp_path, method, detection=autoDetection)
    # The errors mainly come from the detected sphere positions (the network boxes the lit part of the sphere)
    errors = list(lightErrors(lightsPath, dataset.lightsPath).values())
    assert len(errors) == len(readLights(dataset.lightsPath))
    assert np.median(errors) < 15

    monkeypatch.setattr(dataset, "lightsPath", str(lightsPath))
    output = runHalide()
    for name, poseId in dataset.poseIds.items():
        error = angularError(readImage(output / f"{poseId}_normals.exr"), dataset.groundTruth[name])
        assert np.median(error) < 15


def test_pinholeCamera(dataset):
    sfm = dataIO.loadSfmData(dataset.sfmPath)
    camera = lc.pinholeCamera(sfm, next(iter(sfm.getViews().values())))
    expectedFocal = dataset.width / 2 / np.tan(np.radians(45) / 2)
    assert camera.focal == pytest.approx((expectedFocal, expectedFocal), rel=1e-6)
    assert camera.principalPoint == (dataset.width // 2, dataset.height // 2)


@pytest.mark.parametrize("position", [(160, 120), (60, 50), (270, 200)], ids=["center", "topLeft", "bottomRight"])
def test_perspectiveSphereGeometry(position):
    """ The sphere recovered from its minor axis circle matches a rendered perspective sphere. """
    width, height, focal = 320, 240, 160.0
    normals, views, inside, circle = perspectiveSphere(width, height, (*position, 30), focal)
    sphere = lc.PerspectiveSphere(circle, lc.PinholeCamera((focal, focal), (width // 2, height // 2)))

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    estimatedNormals, estimatedViews, estimatedInside = sphere.geometry(xx, yy)
    # Same silhouette, up to the pixels of its border
    assert np.sum(inside != estimatedInside) < 0.05 * np.sum(inside)
    both = inside & estimatedInside
    assert np.allclose(estimatedViews, views, atol=1e-9)
    assert np.median(angularError(estimatedNormals[both][None], normals[both][None])) < 1
    # The region contains the sphere
    x0, y0, x1, y1 = sphere.region()
    rows, cols = np.nonzero(inside)
    assert x0 <= cols.min() and cols.max() < x1 and y0 <= rows.min() and rows.max() < y1


@pytest.mark.parametrize("method", lc.METHODS)
def test_ellipticEstimation(perspectiveDataset, tmp_path, method):
    """ On spheres seen in perspective, the elliptic estimation recovers the lights, unlike the orthographic model. """
    errors = {}
    for elliptic in (False, True):
        outputPath = tmp_path / f"lights_{elliptic}.json"
        lc.lightingCalibration(perspectiveDataset.sfmPath, perspectiveDataset.spheresPath, str(outputPath), method,
                               ellipticEstimation=elliptic)
        errors[elliptic] = list(lightErrors(outputPath, perspectiveDataset.lightsPath).values())
        assert len(errors[elliptic]) == len(readLights(perspectiveDataset.lightsPath))
    assert np.median(errors[True]) < 2
    assert np.max(errors[True]) < 3
    assert np.median(errors[False]) > 10


def test_ellipticEstimationNeedsPinhole(perspectiveDataset, tmp_path):
    with open(perspectiveDataset.sfmPath) as f:
        sfm = json.load(f)
    for intrinsic in sfm["intrinsics"]:
        intrinsic["type"] = "equidistant"
    sfmPath = tmp_path / "equidistant.sfm"
    sfmPath.write_text(json.dumps(sfm))
    with pytest.raises(RuntimeError, match="pinhole"):
        lc.lightingCalibration(str(sfmPath), perspectiveDataset.spheresPath, str(tmp_path / "lights.json"),
                               "whiteSphere", ellipticEstimation=True)


@pytest.mark.parametrize("elliptic", [False, True], ids=["orthographic", "elliptic"])
def test_brightestPointTinySphere(dataset, imagePath, elliptic):
    sfm = dataIO.loadSfmData(dataset.sfmPath)
    camera = lc.pinholeCamera(sfm, next(iter(sfm.getViews().values()))) if elliptic else None
    assert lc.calibrateImage(imagePath, (100.0, 100.0, 0.4), "brightestPoint", camera=camera) is None
