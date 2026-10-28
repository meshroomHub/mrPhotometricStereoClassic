"""
Tests of the light file loaders and of the grouping of the views by pose.
"""

import json

import numpy as np
import pytest

from photometricStereoHalide import dataIO


def writeJson(path, lights):
    path.write_text(json.dumps({"lights": lights}))
    return str(path)


def test_lightsJsonPerView(tmp_path):
    lights = [
        {"viewId": "20", "direction": ["0", "1", "0"], "intensity": ["2", "2", "2"]},
        {"viewId": "10", "direction": ["1", "0", "0"], "intensity": ["1", "1", "1"]},
        {"viewId": "99", "direction": ["0", "0", "1"], "intensity": ["3", "3", "3"]},
    ]
    path = writeJson(tmp_path / "lights.json", lights)
    # Views are given in the image order; view 30 (e.g. the ambient image) has no light
    directions, intensities = dataIO.loadLightsFromJSON(path, [10, 30, 20], 2)
    assert directions == [[1, 0, 0], [0, 1, 0]]
    assert intensities == [[1, 1, 1], [2, 2, 2]]


def test_lightsJsonModel(tmp_path):
    """ Lights saved as a model ("lightId") are applied in the image order. """
    lights = [{"lightId": str(i), "direction": [str(i), "0", "0"], "intensity": ["1", "1", "1"]} for i in range(4)]
    path = writeJson(tmp_path / "lights.json", lights)
    directions, _ = dataIO.loadLightsFromJSON(path, [7, 8, 9], 3)
    assert directions == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]


def test_lightsJsonSphericalHarmonics(tmp_path):
    """ SH lights are reduced to their normalized directional part. """
    lights = [{"viewId": "1", "direction": ["3", "0", "4"] + ["0.5"] * 6, "intensity": ["1", "1", "1"]}]
    directions, _ = dataIO.loadLightsFromJSON(writeJson(tmp_path / "lights.json", lights), [1], 1)
    assert np.allclose(directions, [[0.6, 0, 0.8]])


def test_lightsLP(tmp_path):
    path = tmp_path / "lights.lp"
    path.write_text("2\nIMG_A.png 0.1 0.2 0.3\nimg_b.png 0.4 0.5 0.6\n")
    directions, intensities = dataIO.loadLightsFromLP(str(path), ["/data/img_b.png", "/data/img_a.png", "/data/c.png"])
    # Case insensitive file names, y and z flipped, unit intensities
    assert np.allclose(directions, [[0.4, -0.5, -0.6], [0.1, -0.2, -0.3]])
    assert intensities == [[1, 1, 1], [1, 1, 1]]


def test_lightsFolder(tmp_path):
    np.savetxt(tmp_path / "light_directions.txt", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.savetxt(tmp_path / "light_intensities.txt", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.savetxt(tmp_path / "convertionMatrix.txt", [[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    directions, intensities = dataIO.loadLightsFromFolder(str(tmp_path), 2)
    assert np.allclose(directions, [[0, 1, 0], [1, 0, 0]])
    assert np.allclose(intensities, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_unsupportedLights(tmp_path):
    group = dataIO.PoseGroup(poseId=0, imagePaths=["a.png"])
    with pytest.raises(RuntimeError, match="Unsupported"):
        dataIO.loadLights(str(tmp_path / "lights.txt"), group)


@pytest.mark.parametrize("removeAmbient", [False, True])
def test_groupViewsByPose(dataset, removeAmbient):
    groups = dataIO.groupViewsByPose(dataIO.loadSfmData(dataset.posedSfmPath), removeAmbient)
    assert sorted(g.poseId for g in groups) == sorted(dataset.poseIds.values())
    for group in groups:
        # Sorted by image path (no date metadata), ambient image excluded
        assert len(group.viewIds) == 9
        assert len(group.imagePaths) == 8
        assert group.imagePaths == sorted(group.imagePaths)
        assert not any("ambient" in p for p in group.imagePaths)
        assert (group.ambientPath is not None) == removeAmbient
        assert group.rotation is not None
        assert np.allclose(group.rotation @ group.rotation.T, np.eye(3), atol=1e-6)
