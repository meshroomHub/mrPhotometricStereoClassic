"""
Lighting calibration from calibration spheres: one directional light (or spherical harmonics) per image.

Methods:
    - brightestPoint: for shiny spheres. The light is the mirror reflection of the viewing direction (orthographic
      camera) on the normal of the brightest point of the sphere. Same algorithm as AliceVision.
    - whiteSphere: for white matte spheres. Least squares fit of a Lambertian model (I = normal . L) on the sphere pixels
      that are neither in shadow nor saturated. The intensity is |L|.
    - SH: whiteSphere for the directional part, then second order spherical harmonics fit on the residual.

Sphere models:
    - Orthographic (default): the sphere image is the detected circle, the normals are computed from the pixel position
      relative to its center, and the viewing direction is (0, 0, -1) everywhere.
    - Perspective ("ellipticEstimation"): the sphere image is an ellipse, whose minor axis is the detected circle. The
      3D sphere is recovered from the camera intrinsics, and the normals and viewing directions come from the camera
      rays intersecting it.

Only the area of the sphere is read from the images when the file format allows it.
"""

import functools
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import OpenImageIO as oiio
from pyalicevision import camera as avcamera
from pyalicevision import image as avimage

from . import dataIO

METHODS = ["brightestPoint", "whiteSphere", "SH"]
# Luminance weights used by AliceVision to read RGB images as grayscale
LUMINANCE = (0.2126, 0.7152, 0.0722)
# Sphere pixels used by the fitting methods: away from the silhouette, neither in shadow nor saturated
MAX_RELATIVE_DISTANCE = 0.95
MIN_VALUE, MAX_VALUE = 0.2, 0.8
# Same criterion for the perspective model, as the minimum cosine between the normal and the viewing direction
MIN_VIEW_COSINE = float(np.sqrt(1 - MAX_RELATIVE_DISTANCE ** 2))
VISUALIZATION_SIZE = 1024


@dataclass
class PinholeCamera:
    focal: Tuple[float, float]  # (fx, fy) in pixels
    principalPoint: Tuple[float, float]  # (px, py) in pixels

    def rays(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """ Unit directions (..., 3) of the camera rays through the pixels (camera looking towards +z, y down). """
        rays = np.stack([(xx - self.principalPoint[0]) / self.focal[0], (yy - self.principalPoint[1]) / self.focal[1],
                         np.ones_like(xx, dtype=np.float64)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def pinholeCamera(sfm, view) -> PinholeCamera:
    """ Camera of a view: focal and principal point (image center + offset) of its pinhole intrinsic. """
    intrinsics = sfm.getIntrinsics()
    intrinsicId = view.getIntrinsicId()
    if intrinsicId not in intrinsics.keys():
        raise RuntimeError(f"View {view.getViewId()} has no intrinsic: the elliptic estimation needs the camera focal.")
    intrinsic = intrinsics[intrinsicId]
    if not avcamera.isPinhole(intrinsic.getType()):
        raise RuntimeError(f"The elliptic estimation needs a pinhole camera, not '{intrinsic.getTypeStr()}' "
                           f"(view {view.getViewId()}).")
    fx, fy, offsetX, offsetY = np.array(intrinsic.getParameters(), dtype=np.float64)[:4]
    return PinholeCamera((float(fx), float(fy)), (intrinsic.w() // 2 + float(offsetX), intrinsic.h() // 2 + float(offsetY)))


class PerspectiveSphere:
    """
    Unit sphere whose image is an ellipse of center (cx, cy) and minor semi-axis b (the detected circle).

    For a sphere of angular radius alpha whose center direction makes an angle theta with the optical axis, the image
    is an ellipse of minor semi-axis b = f.sin(alpha) / sqrt(cos²(theta) - sin²(alpha)), centered at the distance
    rho = f.sin(theta).cos(theta) / (cos²(theta) - sin²(alpha)) from the principal point, along the direction of the
    sphere center. Inverting: tan(theta) = rho.f / (b² + f²) and sin(alpha) = cos(theta).b / sqrt(b² + f²).
    The sphere radius is 1: the scale does not change the normals.
    """

    def __init__(self, circle, camera: PinholeCamera):
        cx, cy, b = circle
        self.camera = camera
        # Normalized image coordinates (focal = 1)
        u = (cx - camera.principalPoint[0]) / camera.focal[0]
        v = (cy - camera.principalPoint[1]) / camera.focal[1]
        rho, beta = np.hypot(u, v), b / camera.focal[0]
        theta = np.arctan(rho / (beta * beta + 1))
        sinAlpha = np.cos(theta) * beta / np.sqrt(beta * beta + 1)
        phi = np.arctan2(v, u)
        self.center = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]) / sinAlpha
        # Major semi-axis, for the region containing the sphere
        cosAlpha = np.sqrt(1 - sinAlpha * sinAlpha)
        self.majorSemiAxis = camera.focal[0] * sinAlpha * cosAlpha / (np.cos(theta) ** 2 - sinAlpha * sinAlpha)
        self.circle = circle

    def region(self):
        """ Pixel bounds (x0, y0, x1, y1) of the sphere image. """
        cx, cy, _ = self.circle
        halfSize = self.majorSemiAxis + 2
        return (int(np.floor(cx - halfSize)), int(np.floor(cy - halfSize)),
                int(np.ceil(cx + halfSize)) + 1, int(np.ceil(cy + halfSize)) + 1)

    def geometry(self, xx: np.ndarray, yy: np.ndarray):
        """
        Intersect the camera rays of the pixels with the sphere.

        Returns:
            (normals, views, inside): (..., 3) normals and viewing directions (unit vectors towards the camera), and the
            (...) mask of the pixels whose ray hits the sphere. Outside, the normal is the one of the closest point.
        """
        rays = self.camera.rays(xx, yy)
        projection = rays @ self.center
        discriminant = projection * projection - (self.center @ self.center - 1)
        distance = projection - np.sqrt(np.clip(discriminant, 0, None))
        normals = distance[..., None] * rays - self.center
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals, -rays, discriminant > 0


def readLuminance(path: str, x0: int, y0: int, x1: int, y1: int):
    """
    Read the luminance of the region [x0, x1) x [y0, y1) of an image, without color conversion, like AliceVision's
    readImage to a grayscale float image. Only the needed scanlines or tiles are read (through the OIIO image cache),
    except for RAW images which are fully read with AliceVision. Pixels outside of the image are 0.

    Returns:
        (region, width, height): the (y1 - y0, x1 - x0) float32 region and the full image size.
    """
    dataIO._checkExists(path)
    inputFile = oiio.ImageInput.open(path)
    if inputFile is None:
        raise RuntimeError(f"Cannot open '{path}': {oiio.geterror()}")
    spec, formatName = inputFile.spec(), inputFile.format_name()
    inputFile.close()
    width, height, nbChannels = spec.width, spec.height, spec.nchannels
    if nbChannels == 2:
        raise RuntimeError(f"Images with 2 channels are not supported: '{path}'.")

    region = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    # Intersection of the region with the image
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if ix0 >= ix1 or iy0 >= iy1:
        return region, width, height

    if formatName == "raw":
        # RAW images need AliceVision's processing (demosaicing, orientation...)
        img = avimage.Image_float()
        avimage.readImage(path, img, avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION))
        pixels = np.array(img.getNumpyArray(), dtype=np.float32).reshape(img.height(), img.width())[iy0:iy1, ix0:ix1]
    else:
        nbRead = 3 if nbChannels >= 3 else 1
        buf = oiio.ImageBuf(path)
        pixels = buf.get_pixels(oiio.FLOAT, oiio.ROI(ix0, ix1, iy0, iy1, 0, 1, 0, nbRead))
        if pixels is None or buf.has_error:
            raise RuntimeError(f"Cannot read '{path}': {buf.geterror()}")
        pixels = pixels.reshape(iy1 - iy0, ix1 - ix0, nbRead)
        if nbRead == 3:
            # Same operation order as OIIO's channel_sum
            pixels = (pixels[..., 0] * np.float32(LUMINANCE[0]) + pixels[..., 1] * np.float32(LUMINANCE[1])
                      + pixels[..., 2] * np.float32(LUMINANCE[2]))
        else:
            pixels = pixels[..., 0]

    region[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = pixels
    return region, width, height


def triangleKernel(size: int) -> np.ndarray:
    """ AliceVision's triangle kernel of 2 * size + 1 taps (not normalized). """
    i = np.arange(2 * size + 1, dtype=np.float32)
    return np.where(i > size, 1.0 + size - (i - size), 1.0 + i).astype(np.float32) / size


def separableConvolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ Vertical then horizontal convolution, the borders being extended with the edge pixels. """
    half = len(kernel) // 2
    result = image
    for axis in (0, 1):
        padded = np.pad(result, [(half, half) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        length = result.shape[axis]
        result = sum(weight * np.take(padded, np.arange(k, k + length), axis=axis) for k, weight in enumerate(kernel))
    return result.astype(np.float32)


def mirrorReflection(normal: np.ndarray, view: np.ndarray) -> np.ndarray:
    """ Light direction whose mirror reflection on the normal goes towards the camera (view direction). """
    return 2 * normal.dot(view) * normal - view


def brightestPoint(path: str, sphere, perspectiveSphere: Optional[PerspectiveSphere] = None):
    """
    Light direction from the brightest point of a shiny sphere, as in AliceVision: the sphere is cut out of the image,
    smoothed with a triangle kernel, and the light is the mirror reflection of the viewing direction on the normal of
    the maximum. The viewing direction is (0, 0, -1) with the orthographic model, the camera ray with the perspective one.

    Returns:
        (direction, intensity), or None if the sphere is smaller than a pixel.
    """
    if sphere[2] < 1:
        return None
    if perspectiveSphere is not None:
        return _brightestPointPerspective(path, perspectiveSphere)

    cx, cy, radius = sphere
    x0, y0 = int(np.floor(cx - radius)), int(np.floor(cy - radius))
    # std::round: half away from zero
    patchRadius = int(np.floor(radius + 0.5))
    patch, _, _ = readLuminance(path, x0, y0, x0 + 2 * patchRadius, y0 + 2 * patchRadius)

    # Keep the disk of the sphere
    i, j = np.mgrid[0:2 * patchRadius, 0:2 * patchRadius]
    patch[(i - patchRadius) ** 2 + (j - patchRadius) ** 2 > patchRadius * patchRadius + 2] = 0

    kernelSize = int(np.floor(radius / 20 + 0.5))
    if kernelSize > 0:
        patch = separableConvolution(patch, triangleKernel(kernelSize))
    row, col = np.unravel_index(np.argmax(patch), patch.shape)

    nx, ny = (col + x0 - cx) / radius, (row + y0 - cy) / radius
    normal = np.array([nx, ny, -np.sqrt(max(1 - nx * nx - ny * ny, 0.0))])
    return mirrorReflection(normal, np.array([0.0, 0.0, -1.0])), 1.0


def _brightestPointPerspective(path: str, sphere: PerspectiveSphere):
    x0, y0, x1, y1 = sphere.region()
    patch, _, _ = readLuminance(path, x0, y0, x1, y1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    _, _, inside = sphere.geometry(xx, yy)
    patch[~inside] = 0

    kernelSize = int(np.floor(sphere.circle[2] / 20 + 0.5))
    if kernelSize > 0:
        patch = separableConvolution(patch, triangleKernel(kernelSize))
    row, col = np.unravel_index(np.argmax(patch), patch.shape)

    normal, view, _ = sphere.geometry(np.array(float(col + x0)), np.array(float(row + y0)))
    direction = mirrorReflection(normal, view)
    return direction / np.linalg.norm(direction), 1.0


def sphereSamples(path: str, sphere, perspectiveSphere: Optional[PerspectiveSphere] = None):
    """
    Return the normals (N, 3) and values (N,) of the sphere pixels used by the fitting methods.
    With the orthographic model, the normals are computed from the pixel position relative to the sphere center.
    """
    if perspectiveSphere is not None:
        x0, y0, x1, y1 = perspectiveSphere.region()
        patch, width, height = readLuminance(path, x0, y0, x1, y1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        normals, views, inside = perspectiveSphere.geometry(xx, yy)
        valid = (inside & (np.sum(normals * views, axis=-1) > MIN_VIEW_COSINE) & (patch > MIN_VALUE) & (patch < MAX_VALUE)
                 & (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height))
        return normals[valid], patch[valid].astype(np.float64)

    cx, cy, radius = sphere
    x0, y0 = int(np.floor(cx - radius)), int(np.floor(cy - radius))
    x1, y1 = int(np.ceil(cx + radius)) + 1, int(np.ceil(cy + radius)) + 1
    patch, width, height = readLuminance(path, x0, y0, x1, y1)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    nx, ny = (xx - cx) / radius, (yy - cy) / radius
    d2 = nx * nx + ny * ny
    valid = ((d2 < MAX_RELATIVE_DISTANCE ** 2) & (patch > MIN_VALUE) & (patch < MAX_VALUE)
             & (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height))
    normals = np.stack([nx[valid], ny[valid], -np.sqrt(1 - d2[valid])], axis=1)
    return normals, patch[valid].astype(np.float64)


def fitDirectional(normals, values):
    """ Lambertian least squares fit: returns the unit light direction and its intensity. """
    light = np.linalg.lstsq(normals, values, rcond=None)[0]
    intensity = float(np.linalg.norm(light))
    return light / intensity, intensity


def shBasis(normals):
    """ Second order terms of the AliceVision SH basis: 1, nx.ny, nx.nz, ny.nz, nx² - ny², 3.nz² - 1. """
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    return np.stack([np.ones_like(nx), nx * ny, nx * nz, ny * nz, nx * nx - ny * ny, 3 * nz * nz - 1], axis=1)


def calibrateImage(path: str, sphere, method: str, logger=logging.getLogger(__name__),
                   camera: Optional[PinholeCamera] = None):
    """
    Calibrate the light of one image from its sphere (cx, cy, radius in pixels).

    Args:
        camera: if set, use the perspective sphere model (the circle is the minor axis of the sphere image).

    Returns:
        (direction, intensity): the 3 (directional) or 9 (SH) light coefficients and the intensity,
        or None if the sphere does not have enough usable pixels.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown lighting calibration method: '{method}'.")
    perspectiveSphere = PerspectiveSphere(sphere, camera) if camera is not None else None
    if method == "brightestPoint":
        result = brightestPoint(path, sphere, perspectiveSphere)
        if result is None:
            logger.warning(f"The sphere of '{path}' is too small (radius {sphere[2]:.2f}): the light cannot be calibrated.")
        return result

    normals, values = sphereSamples(path, sphere, perspectiveSphere)
    minSamples = 3 if method == "whiteSphere" else 9
    if len(values) < minSamples:
        logger.warning(f"Not enough sphere pixels between {MIN_VALUE} and {MAX_VALUE} in '{path}' "
                       f"({len(values)} pixels): the light cannot be calibrated.")
        return None

    direction, intensity = fitDirectional(normals, values)
    if method == "whiteSphere":
        return direction, intensity
    # SH: second order terms fitted on the residual of the directional part
    secondOrder = np.linalg.lstsq(shBasis(normals), values - intensity * normals @ direction, rcond=None)[0]
    return np.concatenate([direction, secondOrder]), intensity


@functools.lru_cache(maxsize=2)
def _sphereBasis(size: int) -> np.ndarray:
    """ (size * size, 9) normals and SH terms of the visualization sphere, null outside of the sphere. """
    radius, center = size * 0.9 / 2, size // 2
    i, j = np.mgrid[0:size, 0:size].astype(np.float64)
    nx, ny = (j - center) / radius, (i - center) / radius
    d2 = nx * nx + ny * ny
    normals = np.stack([nx, ny, -np.sqrt(np.clip(1 - d2, 0, 1))], axis=-1).reshape(-1, 3)
    basis = np.concatenate([normals, shBasis(normals)], axis=1)
    basis[(d2 >= 1).reshape(-1)] = 0
    return basis


def renderSphere(light: np.ndarray, size: int = VISUALIZATION_SIZE) -> np.ndarray:
    """ Render a sphere lit by a light (3 or 9 coefficients), scaled by the norm of the light, as AliceVision. """
    basis = _sphereBasis(size)[:, :len(light)]
    return ((basis @ light) * np.linalg.norm(light)).reshape(size, size).astype(np.float32)


def loadSpheres(detectionPath: str) -> Dict[str, tuple]:
    """ Read the spheres of a shape file: {viewId: (cx, cy, radius)}. """
    with open(detectionPath) as f:
        shapes = json.load(f).get("shapes")
    if not shapes:
        raise RuntimeError(f"Cannot find sphere detection data in '{detectionPath}'.")
    observations = shapes[0].get("observations") or {}
    return {str(viewId): (float(o["center"]["x"]), float(o["center"]["y"]), float(o["radius"]))
            for viewId, o in observations.items()}


def sortedViews(sfm) -> List[tuple]:
    """
    Return the (viewId, view) pairs sorted like AliceVision: by acquisition date ("Exif:DateTimeDigitized") if
    available, by image path otherwise. Views with the same date are not dropped: they are sorted by image path.
    """
    def sortKey(item):
        image = item[1].getImage()
        metadata = dict(image.getMetadata())
        return metadata.get("Exif:DateTimeDigitized", image.getImagePath()), image.getImagePath()

    views = sfm.getViews()
    return sorted(((viewId, views[viewId]) for viewId in views.keys()), key=sortKey)


def lightingCalibration(sfmPath: str, detectionPath: str, outputPath: str, method: str = "brightestPoint",
                        saveAsModel: bool = False, visualizationFolder: Optional[str] = None,
                        ellipticEstimation: bool = False, logger=logging.getLogger(__name__)):
    """
    Calibrate the light of every image with a sphere and write the lights JSON file.

    Args:
        detectionPath: shape file with the spheres per view (from SphereDetection).
        saveAsModel: identify the lights by their index ("lightId") instead of their view, to reuse the calibration
                     on other datasets acquired with the same lights in the same order.
        visualizationFolder: if set, write "<imageStem>_<method>.png" images of a sphere lit by each estimated light.
        ellipticEstimation: use the perspective sphere model, with the pinhole intrinsics of the views. The detected
                            circles are the minor axes of the sphere images.
    """
    spheres = loadSpheres(detectionPath)
    sfm = dataIO.loadSfmData(sfmPath)

    lights = []
    for viewId, view in sortedViews(sfm):
        imagePath = view.getImage().getImagePath()
        if dataIO.isAmbient(imagePath):
            continue
        sphere = spheres.get(str(viewId))
        if sphere is None:
            logger.warning(f"No detected sphere found for '{imagePath}'.")
            continue

        camera = pinholeCamera(sfm, view) if ellipticEstimation else None
        result = calibrateImage(imagePath, sphere, method, logger, camera)
        if result is None:
            continue
        direction, intensity = result
        logger.info(f"View {viewId}: direction {np.round(direction[:3], 4).tolist()}, intensity {intensity:.4f}")

        light = {"lightId": len(lights)} if saveAsModel else {"viewId": viewId}
        light["type"] = "SH" if method == "SH" else "directional"
        light["direction"] = [float(v) for v in direction]
        light["intensity"] = [float(intensity)] * 3
        lights.append(light)

        if visualizationFolder:
            stem = os.path.splitext(os.path.basename(imagePath))[0]
            dataIO.writeImageGray(os.path.join(visualizationFolder, f"{stem}_{method}.png"), renderSphere(direction))

    if not lights:
        logger.warning("No light could be calibrated.")
    with open(outputPath, "w") as f:
        json.dump({"lights": lights}, f, indent=4)
