"""
End-to-end tests of the photometric stereo on the synthetic dataset: accuracy against the ground truth,
outputs, and parity with the AliceVision C++ implementation.
"""

import json
import os

import numpy as np
import pytest
from pyalicevision import sfmData, sfmDataIO

from conftest import angularError, readImage

OUTPUT_SUFFIXES = ["_normals.exr", "_normals.png", "_normals_w.exr", "_normals_w.png",
                   "_albedo.exr", "_albedo.png", "_mask.png"]
OUTPUT_SFMS = ["albedoMaps.sfm", "normalMaps.sfm", "normalMapsPNG.sfm"]


def sfmViews(path):
    with open(path) as f:
        return sorted((v["viewId"], v["poseId"], os.path.basename(v["path"])) for v in json.load(f)["views"])


def test_outputs(dataset, runHalide):
    output = runHalide()
    for poseId in dataset.poseIds.values():
        for suffix in OUTPUT_SUFFIXES:
            assert (output / f"{poseId}{suffix}").is_file(), f"{poseId}{suffix}"
    for sfmName in OUTPUT_SFMS:
        views = sfmViews(output / sfmName)
        # One view per pose: the one whose view id is the pose id
        assert sorted(v[1] for v in views) == sorted(str(p) for p in dataset.poseIds.values())
        assert all(v[0] == v[1] for v in views)


@pytest.mark.parametrize("isRobust", [False, True], ids=["leastSquares", "robust"])
def test_groundTruthNormals(dataset, runHalide, isRobust):
    output = runHalide(isRobust=isRobust)
    for name, poseId in dataset.poseIds.items():
        normals = readImage(output / f"{poseId}_normals.exr")
        assert np.allclose(np.linalg.norm(normals, axis=2), 1, atol=1e-5)
        error = angularError(normals, dataset.groundTruth[name])
        # Errors come from shadows (clamped shading) at grazing angles and from the noise
        assert np.median(error) < 1.5
        assert np.percentile(error, 90) < 10


def test_albedo(dataset, runHalide):
    output = runHalide(maskPath=dataset.maskFolder)
    for poseId in dataset.poseIds.values():
        albedo = readImage(output / f"{poseId}_albedo.exr")
        assert albedo.min() >= 0
        assert albedo.max() == pytest.approx(1.0)

    # Pixels outside of the mask of pose A are null
    poseId = dataset.poseIds["A"]
    mask = readImage(output / f"{poseId}_mask.png")[..., 0] > 0
    assert 0 < mask.mean() < 1
    for suffix in ["_albedo.exr", "_normals.exr"]:
        image = readImage(output / f"{poseId}{suffix}")
        assert np.all(image[~mask] == 0)
        assert np.all(np.any(image[mask] != 0, axis=1))


def test_worldNormals(dataset, runHalide):
    output = runHalide(sfmPath=dataset.posedSfmPath)
    sfm = sfmData.SfMData()
    assert sfmDataIO.load(sfm, dataset.posedSfmPath, sfmDataIO.ALL)
    for view in sfm.getViews().values():
        if view.getViewId() != view.getPoseId():
            continue
        # getPose() returns a copy: keep it alive while reading the rotation
        pose = sfm.getPose(view)
        transform = pose.getTransform()
        rotation = np.array(transform.rotation())  # World to camera
        normals = readImage(output / f"{view.getPoseId()}_normals.exr")
        normalsWorld = readImage(output / f"{view.getPoseId()}_normals_w.exr")
        assert np.allclose(normalsWorld, normals @ rotation, atol=1e-5)
        assert np.allclose(np.linalg.norm(normalsWorld, axis=2), 1, atol=1e-5)


def test_downscale(dataset, runHalide):
    output = runHalide(downscale=2)
    for poseId in dataset.poseIds.values():
        assert readImage(output / f"{poseId}_normals.exr").shape == (dataset.height // 2, dataset.width // 2, 3)


def test_missingImage(dataset, runHalide, tmp_path):
    with open(dataset.sfmPath) as f:
        sfm = json.load(f)
    # A lit image: the ambient image is not read without "removeAmbient"
    view = next(v for v in sfm["views"] if "ambient" not in os.path.basename(v["path"]))
    view["path"] = str(tmp_path / "missing.exr")
    sfmPath = tmp_path / "missing.sfm"
    sfmPath.write_text(json.dumps(sfm))
    with pytest.raises(FileNotFoundError):
        runHalide(sfmPath=str(sfmPath))


def test_noLights(dataset, runHalide, tmp_path, monkeypatch):
    lightsPath = tmp_path / "noLights.json"
    lightsPath.write_text(json.dumps({"lights": []}))
    monkeypatch.setattr(dataset, "lightsPath", str(lightsPath))
    with pytest.raises(RuntimeError, match="No image was processed"):
        runHalide()


@pytest.mark.parametrize("params", [
    {},
    {"mask": True, "removeAmbient": True},
    {"downscale": 2},
], ids=["default", "maskAmbient", "downscale"])
def test_parityWithAliceVision(dataset, runHalide, runAliceVision, params):
    maskPath = dataset.maskFolder if params.get("mask") else ""
    options = {k: v for k, v in params.items() if k != "mask"}
    halideOutput = runHalide(maskPath=maskPath, **options)
    aliceVisionOutput = runAliceVision(maskPath=maskPath, **options)

    # World normals are not compared: they differ because of the in-place rotation of the C++ code.
    # Without camera poses, they are equal to the camera normals, which are compared.
    for poseId in dataset.poseIds.values():
        for suffix in ["_normals.exr", "_albedo.exr"]:
            expected = readImage(aliceVisionOutput / f"{poseId}{suffix}")
            assert np.abs(readImage(halideOutput / f"{poseId}{suffix}") - expected).max() < 1e-5, suffix
        for suffix in ["_normals.png", "_albedo.png"]:
            expected = readImage(aliceVisionOutput / f"{poseId}{suffix}").astype(np.int32)
            difference = np.abs(readImage(halideOutput / f"{poseId}{suffix}").astype(np.int32) - expected)
            # Rounding differences on a few pixels
            assert difference.max() <= 1, suffix
            assert np.mean(difference > 0) < 1e-3, suffix
    for sfmName in OUTPUT_SFMS:
        assert sfmViews(halideOutput / sfmName) == sfmViews(aliceVisionOutput / sfmName)


def test_robustShadows(dataset, runHalide):
    """
    The robust estimation is less sensitive to the shadows than the least squares: better normals on the pixels in
    shadow for some lights, same normals elsewhere.
    (Not compared with AliceVision: its robust estimation stops after 11 iterations, on its least squares solution or,
    depending on the run, on a wrong solution as it starts from an uninitialized matrix.)
    """
    leastSquaresOutput = runHalide(name="leastSquares")
    robustOutput = runHalide(name="robust", isRobust=True)
    for name, poseId in dataset.poseIds.items():
        groundTruth = dataset.groundTruth[name]
        inShadow = np.any(groundTruth @ dataset.directions.T <= 0, axis=2)
        assert 0.02 < np.mean(inShadow) < 0.5
        leastSquaresError = angularError(readImage(leastSquaresOutput / f"{poseId}_normals.exr"), groundTruth)
        robustError = angularError(readImage(robustOutput / f"{poseId}_normals.exr"), groundTruth)
        assert np.median(robustError[inShadow]) < 0.75 * np.median(leastSquaresError[inShadow])
        assert np.median(robustError[~inShadow]) == pytest.approx(np.median(leastSquaresError[~inShadow]), abs=0.1)
