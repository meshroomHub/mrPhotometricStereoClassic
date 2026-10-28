"""
Shared fixtures: a synthetic photometric stereo dataset with known normals (see makeSyntheticDataset.py),
and helpers to run the Halide implementation and the AliceVision C++ one on it.
"""

import logging
import os
import shutil
import sys

import numpy as np
import OpenImageIO as oiio
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from makeSyntheticDataset import createSyntheticDataset, runCommand  # noqa: E402
from photometricStereoHalide.photometricStereo import photometricStereo, PhotometricStereoParameters  # noqa: E402


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """ Synthetic dataset: 2 poses of 8 images (320x240) and an ambient image each, with a mask for pose A only. """
    if not shutil.which("aliceVision_cameraInit"):
        pytest.skip("aliceVision_cameraInit is not available.")
    return createSyntheticDataset(tmp_path_factory.mktemp("dataset"))


@pytest.fixture
def runHalide(dataset, tmp_path):
    """ Run the Halide photometric stereo, return the output folder. """
    def run(maskPath="", sfmPath=None, name="halide", **params):
        outputFolder = tmp_path / name
        outputFolder.mkdir()
        photometricStereo(sfmPath or dataset.sfmPath, dataset.lightsPath, maskPath, str(outputFolder),
                          PhotometricStereoParameters(**params), logger=logging.getLogger("test"))
        return outputFolder
    return run


@pytest.fixture
def runAliceVision(dataset, tmp_path):
    """ Run the AliceVision C++ photometric stereo, return the output folder. """
    if not shutil.which("aliceVision_photometricStereo"):
        pytest.skip("aliceVision_photometricStereo is not available.")

    def run(maskPath="", sfmPath=None, removeAmbient=False, isRobust=False, downscale=1):
        outputFolder = tmp_path / "aliceVision"
        outputFolder.mkdir()
        runCommand(["aliceVision_photometricStereo", "--inputPath", sfmPath or dataset.sfmPath,
                    "--pathToJSONLightFile", dataset.lightsPath, "--maskPath", maskPath,
                    "--removeAmbient", str(int(removeAmbient)), "--isRobust", str(int(isRobust)),
                    "--downscale", str(downscale), "--outputPath", str(outputFolder), "--verboseLevel", "error"])
        return outputFolder
    return run


def readImage(path):
    """ Read an output image as a (H, W, C) array (float for EXR, uint8 values for PNG). """
    buf = oiio.ImageBuf(str(path))
    return buf.get_pixels(oiio.FLOAT if str(path).endswith(".exr") else oiio.UINT8)


def angularError(normals, groundTruth):
    """ Per-pixel angle, in degrees, between two (H, W, 3) normal maps. """
    cosine = np.clip(np.sum(normals.astype(np.float64) * groundTruth, axis=2), -1, 1)
    return np.degrees(np.arccos(cosine))

