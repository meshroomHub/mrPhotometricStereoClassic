"""
Detection of the calibration spheres used for the lighting calibration, with ONNX Runtime.

Writes a shape file (JSON) with one circle observation per view, as AliceVision's sphereDetection:
    {"shapes": [{"name": ..., "type": "Circle", "properties": {"color": "green"},
                 "observations": {"<viewId>": {"center": {"x": .., "y": ..}, "radius": .., "type": "matte"}}}]}
"""

import json
import logging
import os
import shutil
from typing import Dict, Optional

import numpy as np
from pyalicevision import image as avimage

from . import dataIO

SHAPE_NAME = "Manual Sphere Detection"


def _writeShapeFile(outputPath: str, observations: Dict[str, dict]):
    shapes = [{"name": SHAPE_NAME, "type": "Circle", "properties": {"color": "green"}, "observations": observations}]
    with open(outputPath, "w") as f:
        json.dump({"shapes": shapes}, f, indent=4)


def _sortedViews(sfm):
    """ Return the (viewId, view) pairs sorted by view id. """
    views = sfm.getViews()
    return [(viewId, views[viewId]) for viewId in sorted(views.keys())]


def _observation(x: float, y: float, radius: float, score: Optional[float] = None) -> dict:
    observation = {"center": {"x": float(x), "y": float(y)}, "radius": float(radius)}
    if score is not None:
        observation["score"] = float(score)
    observation["type"] = "matte"
    return observation


def readImageForDetection(path: str) -> np.ndarray:
    """ Read an image as the network input: (1, 3, H, W) float32 RGB in [0, 1], in sRGB (like AliceVision). """
    dataIO._checkExists(path)
    img = avimage.Image_RGBColor()
    avimage.readImage(path, img, avimage.ImageReadOptions(avimage.EImageColorSpace_SRGB))
    pixels = np.array(img.getNumpyArray(), dtype=np.float32) / 255.0
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def createSession(modelPath: str, useGpu: bool, logger=logging.getLogger(__name__)):
    """ Create the ONNX Runtime session, on GPU if requested and available. """
    import onnxruntime as ort

    modelPath = os.path.expandvars(modelPath)
    if not os.path.isfile(modelPath):
        raise FileNotFoundError(f"The sphere detection model '{modelPath}' does not exist.")
    providers = ["CPUExecutionProvider"]
    if useGpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(modelPath, providers=providers)
    logger.info(f"Sphere detection model: {modelPath} ({', '.join(session.get_providers())})")
    if useGpu and "CUDAExecutionProvider" not in session.get_providers():
        # ONNX Runtime silently falls back to the CPU when the CUDA provider cannot be loaded
        logger.warning("The detection runs on CPU: the ONNX Runtime CUDA provider is not available.")
    return session


def detectSphere(session, imagePath: str, minScore: float):
    """
    Detect the calibration sphere of an image.

    Returns:
        (x, y, radius, score) of the best detection with a score above "minScore", None if there is none.
        The circle is inscribed in the top-left corner of the bounding box, like in AliceVision.
    """
    # Only the boxes and scores are requested: the mask branch of the network is not computed
    boxes, scores = session.run(["boxes", "scores"], {"input": readImageForDetection(imagePath)})
    # Detections are sorted by decreasing score
    for box, score in zip(boxes, scores):
        if score > minScore:
            x1, y1, x2, y2 = box
            radius = min(y2 - y1, x2 - x1) / 2
            return x1 + radius, y1 + radius, radius, score
    return None


def autoDetectSpheres(sfmPath: str, modelPath: str, outputPath: str, minScore: float = 0.0, useGpu: bool = False,
                      logger=logging.getLogger(__name__)):
    """ Detect the calibration sphere in every image (except the ambient ones) and write the shape file. """
    sfm = dataIO.loadSfmData(sfmPath)
    session = createSession(modelPath, useGpu, logger)

    observations = {}
    for viewId, view in _sortedViews(sfm):
        imagePath = view.getImage().getImagePath()
        if dataIO.isAmbient(imagePath):
            continue
        detection = detectSphere(session, imagePath, minScore)
        if detection is None:
            logger.warning(f"No sphere detected for '{imagePath}'.")
            continue
        x, y, radius, score = detection
        logger.info(f"View {viewId}: sphere at ({x:.1f}, {y:.1f}), radius {radius:.1f}, score {score:.3f}")
        observations[str(viewId)] = _observation(x, y, radius, score)

    _writeShapeFile(outputPath, observations)


def writeManualSpheres(sfmPath: str, observations: Dict[str, dict], outputPath: str, fillMissingSpheres: bool,
                       logger=logging.getLogger(__name__)):
    """
    Write the shape file from manually defined spheres.

    Args:
        observations: {viewId: {"center": {"x": .., "y": ..}, "radius": ..}} (e.g. a keyable Circle attribute value).
        fillMissingSpheres: use the sphere of the last view with a sphere (in view id order) for the views without any.
    """
    sfm = dataIO.loadSfmData(sfmPath)
    # Ignore the undefined circles (default values)
    observations = {str(k): v for k, v in observations.items() if v.get("radius", -1) > 0}
    lastKnown = observations[max(observations, key=int)] if observations else None

    outObservations = {}
    for viewId, _ in _sortedViews(sfm):
        observation = observations.get(str(viewId))
        if observation is None:
            if not fillMissingSpheres or lastKnown is None:
                logger.info(f"Sphere shape for view ID {viewId} not found.")
                continue
            logger.info(f"Sphere shape for view ID {viewId} not found: using the last known sphere.")
            observation = lastKnown
        outObservations[str(viewId)] = _observation(observation["center"]["x"], observation["center"]["y"],
                                                    observation["radius"])

    _writeShapeFile(outputPath, outObservations)


def writeSpheresFromFile(sfmPath: str, sphereFile: str, outputPath: str, fillMissingSpheres: bool,
                         logger=logging.getLogger(__name__)):
    """
    Write the shape file from an input shape file: copied as is, or completed with the last sphere of the
    file for the views without any if "fillMissingSpheres" is set.
    """
    if not fillMissingSpheres:
        shutil.copyfile(sphereFile, outputPath)
        return

    with open(sphereFile) as f:
        shapes = json.load(f).get("shapes")
    if not shapes:
        raise RuntimeError(f"Cannot find sphere detection data in '{sphereFile}'.")
    observations = shapes[0].get("observations") or {}
    if not observations:
        raise RuntimeError(f"No sphere in '{sphereFile}'.")

    # The last sphere of the file is used for the missing views
    lastObservation = list(observations.values())[-1]
    sfm = dataIO.loadSfmData(sfmPath)
    for viewId, _ in _sortedViews(sfm):
        if str(viewId) not in observations:
            logger.info(f"Sphere shape for view ID {viewId} not found: using the last known sphere.")
            observations[str(viewId)] = _observation(float(lastObservation["center"]["x"]),
                                                     float(lastObservation["center"]["y"]),
                                                     float(lastObservation["radius"]))
    _writeShapeFile(outputPath, observations)
