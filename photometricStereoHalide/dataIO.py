"""
Input/output helpers for the photometric stereo: SfMData parsing, light files and image read/write.

Images are read and written with AliceVision (pyalicevision) to get exactly the same decoding
(RAW, color spaces...) and encoding as the AliceVision photometric stereo.
They are returned as planar float32 numpy arrays of shape (channels, height, width).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import OpenImageIO as oiio
from pyalicevision import image as avimage
from pyalicevision import sfmData as avsfmData
from pyalicevision import sfmDataIO as avsfmDataIO


@dataclass
class PoseGroup:
    """ All the views sharing the same pose: one image per light, and optionally an ambient image. """
    poseId: int
    viewIds: List[int] = field(default_factory=list)  # All the views of the pose (including the ambient one), sorted
    imagePaths: List[str] = field(default_factory=list)  # Images lit by a calibrated light, sorted
    ambientPath: Optional[str] = None
    rotation: Optional[np.ndarray] = None  # World to camera rotation of the pose, if any


def isAmbient(path: str) -> bool:
    return "ambient" in os.path.splitext(os.path.basename(path))[0].lower()


def loadSfmData(sfmPath: str):
    sfm = avsfmData.SfMData()
    if not avsfmDataIO.load(sfm, sfmPath, avsfmDataIO.ALL):
        raise RuntimeError(f"Cannot load the SfMData file '{sfmPath}'.")
    return sfm


def groupViewsByPose(sfm, removeAmbient: bool) -> List[PoseGroup]:
    """
    Group the views by pose. Inside a pose, the views are sorted by acquisition date
    ("Exif:DateTimeDigitized") if available, by image path otherwise.
    """
    viewsPerPose = {}
    for viewId, view in sfm.getViews().items():
        viewsPerPose.setdefault(view.getPoseId(), []).append(viewId)

    groups = []
    for poseId in sorted(viewsPerPose):
        viewIds = sorted(viewsPerPose[poseId])
        views = sfm.getViews()
        hasMetadata = "Exif:DateTimeDigitized" in dict(views[viewIds[0]].getImage().getMetadata())

        def sortKey(viewId):
            img = views[viewId].getImage()
            return dict(img.getMetadata())["Exif:DateTimeDigitized"] if hasMetadata else img.getImagePath()

        group = PoseGroup(poseId=poseId)
        # Same behavior as a std::map: views with an identical key are deduplicated (the last one wins)
        byKey = {}
        for viewId in viewIds:
            byKey[sortKey(viewId)] = viewId
        group.viewIds = [byKey[k] for k in sorted(byKey)]

        for viewId in group.viewIds:
            path = views[viewId].getImage().getImagePath()
            if not isAmbient(path):
                group.imagePaths.append(path)
            elif removeAmbient:
                group.ambientPath = path

        if sfm.existsPose(views[group.viewIds[0]]):
            # getPose() returns a copy and rotation() a view on its memory: keep the pose alive while copying it
            pose = sfm.getPose(views[group.viewIds[0]])
            transform = pose.getTransform()
            group.rotation = np.array(transform.rotation(), dtype=np.float64)

        groups.append(group)
    return groups


def _floats(values) -> List[float]:
    return [float(v) for v in values]


def loadLightsFromJSON(lightPath: str, viewIds: List[int], nbImages: int):
    """
    Load the light directions and intensities from a LightingCalibration JSON file.
    Lights are either associated to a view ("viewId") or ordered ("lightId", when the calibration
    was saved as a model and is applied in the image order).

    Returns:
        (directions, intensities): lists of 3-float lists.
    """
    with open(lightPath) as f:
        lights = json.load(f)["lights"]

    directions, intensities = [], []

    def addLight(light):
        direction = _floats(light["direction"])
        if len(direction) > 3:
            # Spherical harmonics are not supported yet: lighting is reduced to its normalized directional part
            logging.info("SH will soon be available for use in PS. For now, lighting is reduced to directional.")
            direction = list(np.asarray(direction[:3]) / np.linalg.norm(direction[:3]))
        directions.append(direction[:3])
        intensities.append(_floats(light["intensity"])[:3])

    for viewId in viewIds:
        for light in lights:
            if len(directions) == nbImages:
                break
            if "lightId" in light:
                addLight(light)
            elif int(light.get("viewId", -1)) == viewId:
                addLight(light)
                break
    return directions, intensities


def loadLightsFromLP(lightPath: str, imagePaths: List[str]):
    """
    Load the light directions from a ".lp" file ("<imageFileName> x y z" per line).
    The y and z axes are flipped to match the AliceVision convention, and intensities are set to 1.
    """
    entries = {}
    with open(lightPath) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) >= 4:
                entries.setdefault(tokens[0].lower(), _floats(tokens[1:4]))

    directions, intensities = [], []
    for path in imagePaths:
        values = entries.get(os.path.basename(path).lower())
        if values is not None:
            directions.append([values[0], -values[1], -values[2]])
            intensities.append([1.0, 1.0, 1.0])
    return directions, intensities


def loadLightsFromFolder(folder: str, nbImages: int):
    """
    Load the lights from a folder with "light_directions.txt", "light_intensities.txt"
    and optionally "convertionMatrix.txt" (3x3 matrix applied to the directions).
    """
    intensities = np.loadtxt(os.path.join(folder, "light_intensities.txt"), ndmin=2)[:, :3]
    directions = np.loadtxt(os.path.join(folder, "light_directions.txt"), ndmin=2)[:nbImages, :3]
    conversionPath = os.path.join(folder, "convertionMatrix.txt")
    if os.path.exists(conversionPath):
        directions = directions @ np.loadtxt(conversionPath, ndmin=2).T
    return directions.tolist(), intensities.tolist()


def loadLights(lightPath: str, group: PoseGroup):
    """ Load the lights of a pose group, whatever the format of the light data. """
    if os.path.isdir(lightPath):
        return loadLightsFromFolder(lightPath, len(group.imagePaths))
    extension = os.path.splitext(lightPath)[1].lower()
    if extension == ".json":
        return loadLightsFromJSON(lightPath, group.viewIds, len(group.imagePaths))
    if extension == ".lp":
        return loadLightsFromLP(lightPath, group.imagePaths)
    raise RuntimeError(f"Unsupported light file format: '{lightPath}'.")


def _downscale(pixels: np.ndarray, downscale: int) -> np.ndarray:
    """ Downscale a (H, W, C) float32 image with OIIO's default resize filter, like AliceVision's resizeImage. """
    if downscale <= 1:
        return pixels
    height, width, channels = pixels.shape
    src = oiio.ImageBuf(np.ascontiguousarray(pixels))
    dst = oiio.ImageBuf(oiio.ImageSpec(width // downscale, height // downscale, channels, oiio.FLOAT))
    if not oiio.ImageBufAlgo.resize(dst, src):
        raise RuntimeError(f"Image resize failed: {dst.geterror()}")
    return dst.get_pixels(oiio.FLOAT)


def _checkExists(path: str):
    # pyalicevision aborts the process (uncaught C++ exception) when an image cannot be opened
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The image file '{path}' does not exist.")


def readImageRGB(path: str, downscale: int = 1) -> np.ndarray:
    """ Read an image without color conversion, as a (3, H, W) float32 array. """
    _checkExists(path)
    img = avimage.Image_RGBfColor()
    options = avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION)
    avimage.readImage(path, img, options)
    pixels = _downscale(np.array(img.getNumpyArray(), dtype=np.float32), downscale)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def readMask(path: str, downscale: int = 1) -> np.ndarray:
    """ Read a mask (in sRGB, like AliceVision) as a (H, W) float32 array. """
    _checkExists(path)
    img = avimage.Image_float()
    options = avimage.ImageReadOptions(avimage.EImageColorSpace_SRGB)
    avimage.readImage(path, img, options)
    pixels = np.array(img.getNumpyArray(), dtype=np.float32).reshape(img.height(), img.width(), 1)
    return np.ascontiguousarray(_downscale(pixels, downscale)[:, :, 0])


def _writeOptions(storageDataType):
    # The ImageWriteOptions setters must not be chained: each call returns a temporary copy in Python
    options = avimage.ImageWriteOptions()
    options.toColorSpace(avimage.EImageColorSpace_NO_CONVERSION)
    options.storageDataType(storageDataType)
    return options


def writeImageRGB(path: str, planar: np.ndarray, storageDataType=avimage.EStorageDataType_Float):
    """ Write a (3, H, W) float32 image without color conversion. """
    img = avimage.Image_RGBfColor()
    img.fromNumpyArray(np.ascontiguousarray(planar.transpose(1, 2, 0), dtype=np.float32))
    avimage.writeImage(path, img, _writeOptions(storageDataType))


def writeImageRGB8(path: str, planar: np.ndarray):
    """ Write a (3, H, W) uint8 image. """
    img = avimage.Image_RGBColor()
    img.fromNumpyArray(np.ascontiguousarray(planar.transpose(1, 2, 0), dtype=np.uint8))
    avimage.writeImage(path, img, _writeOptions(avimage.EStorageDataType_Float))


def writeMask(path: str, mask: np.ndarray):
    """ Write a (H, W) float32 mask. """
    img = avimage.Image_float()
    img.fromNumpyArray(np.ascontiguousarray(mask[:, :, None], dtype=np.float32))
    avimage.writeImage(path, img, _writeOptions(avimage.EStorageDataType_Float))


def writeOutputSfmData(sfmPath: str, outputFolder: str):
    """
    Write the albedo and normal maps SfMData files: one view per pose (the view whose id is the pose id)
    pointing to the corresponding output image.
    """
    outputs = {
        "albedoMaps.sfm": "_albedo.exr",
        "normalMaps.sfm": "_normals_w.exr",
        "normalMapsPNG.sfm": "_normals_w.png",
    }
    for sfmName, suffix in outputs.items():
        sfm = loadSfmData(sfmPath)
        views = sfm.getViews()
        for viewId in list(views.keys()):
            view = views[viewId]
            if viewId != view.getPoseId():
                views.erase(viewId)
            else:
                view.getImage().setImagePath(os.path.join(outputFolder, f"{view.getPoseId()}{suffix}"))
        partFlags = avsfmDataIO.VIEWS | avsfmDataIO.INTRINSICS | avsfmDataIO.EXTRINSICS
        if not avsfmDataIO.save(sfm, os.path.join(outputFolder, sfmName), partFlags):
            raise RuntimeError(f"Cannot write the SfMData file '{sfmName}'.")
