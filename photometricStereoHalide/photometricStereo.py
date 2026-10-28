"""
Photometric stereo with Halide: estimate per-pixel normals and albedo from images taken from the
same viewpoint under different known directional lights.

The per-pixel computations are Halide pipelines (see halidePipelines.py); this module orchestrates
them per pose and handles the inputs/outputs.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import dataIO
from . import halidePipelines as hp


@dataclass
class PhotometricStereoParameters:
    removeAmbient: bool = False
    isRobust: bool = False
    downscale: int = 1
    # Robust (ADMM) algorithm parameters: the penalty mu grows by "muGrowth" per iteration up to "muMax", and the
    # iterations stop when the relative primal residual |L.M - I - E| / |I| is below "epsilon"
    mu: float = 0.1
    muGrowth: float = 1.1
    muMax: float = 1e4
    maxIterations: int = 1000
    epsilon: float = 1e-5
    # The primal residual is a full pass on the images: it is only computed every "convergenceCheckStep" iterations
    convergenceCheckStep: int = 10


def solveLeastSquares(imagePaths, lightMat, intensities, ambient, mask, params, logger):
    """
    Classic photometric stereo: per-pixel least squares solution of L.m = I.
    The images are streamed: only one image and the accumulator are in memory at a time.

    Returns:
        (normals, albedo): (3, H, W) arrays, the albedo not being normalized yet.
    """
    pinv = np.linalg.pinv(lightMat.astype(np.float64)).astype(np.float32)  # (3, N)
    height, width = mask.shape
    acc = np.zeros((9, height, width), dtype=np.float32)
    accumulate = hp.getPipeline(hp.AccumulatePipeline)

    for i, path in enumerate(imagePaths):
        logger.debug(f"  Accumulate {path}")
        image = dataIO.readImageRGB(path, params.downscale)
        _checkSize(image, mask, path)
        accumulate(acc, image, ambient, intensities[i], pinv[:, i])

    normals = np.empty((3, height, width), dtype=np.float32)
    albedo = np.empty((3, height, width), dtype=np.float32)
    hp.getPipeline(hp.FinalizePipeline)(acc, mask, normals, albedo)
    return normals, albedo


def solveRobust(imagePaths, lightMat, intensities, ambient, mask, params, logger):
    """
    Robust photometric stereo: L1 fitting of the luminance (sparse errors E, Lagrange multipliers W) with ADMM and an
    increasing penalty, then albedo as the per-channel median of the image / shading ratios.
    All the images are kept in memory.

    Returns:
        (normals, albedo): (3, H, W) arrays, the albedo not being normalized yet.
    """
    nbImages = len(imagePaths)
    height, width = mask.shape
    lights = np.ascontiguousarray(lightMat, dtype=np.float32)  # (N, 3)
    pinv = np.ascontiguousarray(np.linalg.pinv(lightMat.astype(np.float64)).astype(np.float32))  # (3, N)
    inMask = mask > 0.7

    images = np.empty((nbImages, 3, height, width), dtype=np.float32)
    for i, path in enumerate(imagePaths):
        image = dataIO.readImageRGB(path, params.downscale)
        _checkSize(image, mask, path)
        if ambient is not None:
            image -= ambient
        images[i] = image / np.asarray(intensities[i], dtype=np.float32)[:, None, None]

    # Pixels outside of the mask are set to 0: their solution stays 0 and does not affect the convergence test
    gray = np.einsum("c,nchw->nhw", np.asarray(hp.LUMINANCE, dtype=np.float32), images)
    gray *= inMask
    gray = np.ascontiguousarray(gray)

    # Initialization with M = 0
    solution = np.zeros((3, height, width), dtype=np.float32)
    errors = np.ascontiguousarray(-gray)
    multipliers = np.zeros_like(gray)
    grayNorm = max(float(np.linalg.norm(gray)), 1e-30)

    robustSolve = hp.getPipeline(hp.RobustSolvePipeline)
    robustUpdate = hp.getPipeline(hp.RobustUpdatePipeline)
    mu = params.mu
    for k in range(1, params.maxIterations + 1):
        robustSolve(gray, errors, multipliers, pinv, mu, solution)
        robustUpdate(gray, multipliers, solution, lights, mu, errors)

        # Do not use the change of the solution: it is close to 0 during the first iterations, before ADMM moves
        # away from the least squares solution (this stopped the AliceVision implementation after 11 iterations)
        if k % params.convergenceCheckStep == 0:
            primalResidual = np.linalg.norm(np.einsum("nj,jhw->nhw", lights, solution) - gray - errors) / grayNorm
            if primalResidual < params.epsilon:
                logger.info(f"  Robust estimation converged in {k} iterations.")
                break
        mu = min(mu * params.muGrowth, params.muMax)
    else:
        logger.warning(f"  Robust estimation did not converge in {params.maxIterations} iterations.")

    normals = np.empty_like(solution)
    hp.getPipeline(hp.NormalizePipeline)(solution, mask, normals)

    # Albedo: median over the images of the ratio between the pixel value and the shading.
    # Only the images where the pixel is lit (positive shading) are considered.
    shading = np.einsum("nj,jhw->nhw", lights, normals)
    lit = shading > 0
    albedo = np.zeros((3, height, width), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in range(3):
            ratios = np.where(lit, images[:, c] / shading, np.nan)
            albedo[c] = np.nan_to_num(np.nanmedian(ratios, axis=0), nan=0.0)
    albedo *= inMask
    return normals, albedo


def _checkSize(image, mask, path):
    if image.shape[1:] != mask.shape:
        raise RuntimeError(f"Size mismatch: '{path}' is {image.shape[2]}x{image.shape[1]}, "
                           f"expected {mask.shape[1]}x{mask.shape[0]} (mask or first image size).")


def processPose(group: dataIO.PoseGroup, lightMat, intensities, maskPath: Optional[str],
                params: PhotometricStereoParameters, outputFolder: str, logger):
    """ Compute and write the normals and albedo of one pose. """
    firstImage = dataIO.readImageRGB(group.imagePaths[0], params.downscale)
    height, width = firstImage.shape[1:]

    if maskPath:
        mask = dataIO.readMask(maskPath, params.downscale)
    else:
        mask = np.ones((height, width), dtype=np.float32)

    ambient = None
    if group.ambientPath:
        logger.info(f"  Removing ambient light: {group.ambientPath}")
        ambient = dataIO.readImageRGB(group.ambientPath, params.downscale)
        _checkSize(ambient, mask, group.ambientPath)

    solver = solveRobust if params.isRobust else solveLeastSquares
    normals, albedo = solver(group.imagePaths, lightMat, intensities, ambient, mask, params, logger)

    maxAlbedo = albedo.max()
    if maxAlbedo > 0:
        albedo /= maxAlbedo

    prefix = os.path.join(outputFolder, str(group.poseId))
    toPng = hp.getPipeline(hp.NormalsToPngPipeline)
    normalsPng = np.empty(normals.shape, dtype=np.uint8)

    dataIO.writeImageRGB(prefix + "_normals.exr", normals)
    toPng(normals, normalsPng)
    dataIO.writeImageRGB8(prefix + "_normals.png", normalsPng)
    dataIO.writeImageRGB(prefix + "_albedo.exr", albedo)
    dataIO.writeImageRGB(prefix + "_albedo.png", albedo, dataIO.avimage.EStorageDataType_Half)
    dataIO.writeMask(prefix + "_mask.png", (mask > 0.7).astype(np.float32))

    # World coordinates: apply the camera to world rotation if the pose is known
    if group.rotation is not None:
        normalsWorld = np.empty_like(normals)
        hp.getPipeline(hp.RotatePipeline)(normals, group.rotation.T, normalsWorld)
        normals = normalsWorld
    dataIO.writeImageRGB(prefix + "_normals_w.exr", normals)
    toPng(normals, normalsPng)
    dataIO.writeImageRGB8(prefix + "_normals_w.png", normalsPng)


def getMaskPath(maskPath: str, group: dataIO.PoseGroup, logger) -> str:
    """
    Return the mask file of a pose: "maskPath" itself if it is a file, otherwise the mask named after the
    image folder without its "ps_" prefix (e.g. "ps_object/" -> "<maskPath>/object.png").
    Return an empty string if there is no mask.
    """
    if not maskPath or os.path.isfile(maskPath):
        return maskPath
    folderName = os.path.basename(os.path.dirname(group.imagePaths[0]))
    path = os.path.join(maskPath, folderName[3:] + ".png")
    if not os.path.exists(path):
        logger.warning(f"Could not open '{path}'. Every pixel will be used.")
        return ""
    return path


def photometricStereo(sfmPath: str, lightPath: str, maskPath: str, outputFolder: str,
                      params: PhotometricStereoParameters, logger=logging.getLogger(__name__)):
    """
    Run the photometric stereo on every pose of an SfMData file.

    Args:
        sfmPath: input SfMData file.
        lightPath: light file (.json from LightingCalibration, .lp) or folder with light_*.txt files.
        maskPath: mask file, or folder with one mask per pose (see getMaskPath). Empty to use all the pixels.
        outputFolder: output folder.
        params: algorithm parameters.
    """
    sfm = dataIO.loadSfmData(sfmPath)
    groups = dataIO.groupViewsByPose(sfm, params.removeAmbient)

    nbProcessed = 0
    for group in groups:
        logger.info(f"Pose Id: {group.poseId}")
        for path in group.imagePaths:
            logger.info(f" - {path}")

        directions, intensities = dataIO.loadLights(lightPath, group)
        if len(directions) != len(group.imagePaths) or not group.imagePaths:
            if group.imagePaths:
                logger.warning(f"Mismatch between the number of images and the number of light intensities "
                               f"({len(group.imagePaths)} images, {len(directions)} light intensities). This might happen "
                               "when no sphere was detected for an image in the list, and thus light was not calibrated for it.")
            else:
                logger.warning("Empty list, skipping pose.")
            continue
        if len(group.imagePaths) < 3:
            logger.warning(f"Only {len(group.imagePaths)} image(s) for pose {group.poseId}: at least 3 lights are needed. "
                           "Input images need to be located in a folder that starts with 'ps_' to be grouped together.")
            continue

        processPose(group, np.asarray(directions, dtype=np.float32), intensities,
                    getMaskPath(maskPath, group, logger), params, outputFolder, logger)
        nbProcessed += 1

    if nbProcessed == 0:
        raise RuntimeError("No image was processed for the photometric stereo. This might happen if no sphere has been "
                           "detected in any of the input images, or if the light could not be calibrated.")

    dataIO.writeOutputSfmData(sfmPath, outputFolder)
