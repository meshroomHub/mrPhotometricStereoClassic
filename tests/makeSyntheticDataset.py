#!/usr/bin/env python
"""
Create a synthetic photometric stereo dataset with known normals, albedo, lights and calibration spheres.

Each pose is a scene with a Lambertian tilted plane with Gaussian bumps (the object) and a checkerboard albedo, and a
glossy white calibration sphere (Lambertian plus a sharp specular highlight, as expected by the "brightestPoint"
lighting calibration). It is rendered under directional lights with per-channel intensities, plus an ambient term and
a small noise. The calibration sphere is realistic enough to be found by the sphere detection network.

Output layout:
    <output>/
    ├── images/ps_A/img_00.exr ... img_<N-1>.exr, img_ambient.exr   # One folder per pose
    ├── images/ps_B/...
    ├── masks/A.png                  # Mask of pose A only (pose B uses all the pixels)
    ├── groundTruth/A_normals.exr    # Ground truth normals (camera coordinates) and albedo per pose
    ├── groundTruth/A_albedo.exr
    ├── groundTruth/spheres.json     # Calibration spheres of every view, as a shape file (usable as sphere file)
    ├── cameraInit.sfm               # SfMData from aliceVision_cameraInit (one pose per "ps_" folder)
    ├── cameraPoses.sfm              # Same SfMData with an arbitrary rotation per pose
    └── lights.json                  # Ground truth lights per view, in the LightingCalibration format

Usage (aliceVision_cameraInit must be available):
    rez env python-3.11 pyoiio aliceVision-develop -- python tests/makeSyntheticDataset.py <outputFolder> [options]
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import OpenImageIO as oiio

# Calibration sphere per pose: center relative to the width and height, radius relative to the height
SPHERES = {"A": (0.8, 0.3, 0.17), "B": (0.18, 0.72, 0.15)}
# Gaussian bumps of the object per pose: center relative to the width and height, standard deviation and
# height relative to the height
BUMPS = {"A": [(0.35, 0.55, 0.12, 0.25), (0.5, 0.4, 0.08, 0.15)], "B": [(0.6, 0.45, 0.15, 0.3)]}
# Mask of pose A: ellipse (center x, center y, semi-axis x, semi-axis y) relative to the image size
MASK_ELLIPSE = (0.42, 0.5, 0.3, 0.38)
# Height gradient of the background plane
PLANE_GRADIENT = (-0.2, 0.1)
SPHERE_ALBEDO = 0.8
# Blinn-Phong specular highlight of the calibration sphere
SPHERE_SPECULAR = 0.6
SPHERE_SHININESS = 200


def makeLights(nbLights):
    """ Directional lights around the viewing direction (AliceVision convention: z towards the camera is negative). """
    directions, intensities = [], []
    for i in range(nbLights):
        theta, phi = np.radians(25 + 4 * i * 8 / nbLights), 2 * np.pi * i / nbLights
        directions.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)])
        intensities.append([1.0 + 0.05 * i, 0.9, 1.1 - 0.03 * i])
    return np.array(directions, np.float32), np.array(intensities, np.float32)


def makeScene(width, height, sphere, bumps, seed):
    """
    Return the ground truth (H, W, 3) normals and albedo of a tilted plane with Gaussian bumps and a calibration sphere,
    the (H, W, 1) mask of the sphere and the (H, W, 3) viewing directions (orthographic camera).
    The normals of a surface of height h (towards the camera) are proportional to (-dh/dx, -dh/dy, -1).
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradientX = np.full((height, width), PLANE_GRADIENT[0], np.float32)
    gradientY = np.full((height, width), PLANE_GRADIENT[1], np.float32)
    for bx, by, sigma, amplitude in bumps:
        bx, by, sigma, amplitude = bx * width, by * height, sigma * height, amplitude * height
        bump = amplitude * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma * sigma))
        gradientX -= (xx - bx) / (sigma * sigma) * bump
        gradientY -= (yy - by) / (sigma * sigma) * bump
    normals = np.stack([-gradientX, -gradientY, -np.ones_like(gradientX)], axis=2)

    squareSize = max(2, width // 16)
    checker = ((xx // squareSize + yy // squareSize) % 2)[..., None] > 0
    albedo = np.where(checker, [0.7, 0.5, 0.3], [0.3, 0.6, 0.8]).astype(np.float32)
    albedo *= 0.8 + 0.2 * np.random.default_rng(seed).random((height, width, 1)).astype(np.float32)

    # Calibration sphere, in front of the plane
    views = np.zeros((height, width, 3), np.float32)
    views[..., 2] = -1
    cx, cy, radius = sphere
    nx, ny = (xx - cx) / radius, (yy - cy) / radius
    d2 = nx ** 2 + ny ** 2
    inside = (d2 < 1)[..., None]
    sphereNormals = np.stack([nx, ny, -np.sqrt(np.clip(1 - d2, 0, 1))], axis=2)
    normals = np.where(inside, sphereNormals, normals)
    albedo = np.where(inside, np.float32(SPHERE_ALBEDO), albedo)

    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    return normals.astype(np.float32), albedo.astype(np.float32), inside, views.astype(np.float32)


def writeExr(path, pixels):
    oiio.ImageBuf(np.ascontiguousarray(pixels, dtype=np.float32)).write(str(path))


def runCommand(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}")


def createSyntheticDataset(root, width=320, height=240, nbLights=8, noise=0.002, seed=0):
    """
    Create the dataset in "root" (see the module documentation for the layout).

    Returns:
        SimpleNamespace with the paths, the ground truth normals and calibration spheres (cx, cy, radius in pixels)
        per pose name ("A", "B"), the pose ids per pose name, and the light directions and intensities.
    """
    if not shutil.which("aliceVision_cameraInit"):
        raise RuntimeError("aliceVision_cameraInit is not available.")

    root = Path(root).absolute()
    rng = np.random.default_rng(seed)
    directions, intensities = makeLights(nbLights)
    groundTruth, spheres = {}, {}
    (root / "groundTruth").mkdir(parents=True, exist_ok=True)

    for name, (cx, cy, radius) in SPHERES.items():
        folder = root / "images" / f"ps_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        spheres[name] = (cx * width, cy * height, radius * height)
        normals, albedo, sphereMask, views = makeScene(width, height, spheres[name], BUMPS[name], ord(name))
        ambient = 0.04 * albedo
        for i in range(nbLights):
            shading = np.clip(normals @ directions[i], 0, None)[..., None]
            # Specular highlight on the sphere (Blinn-Phong, with the viewing direction of each pixel)
            halfVectors = directions[i] + views
            halfVectors /= np.linalg.norm(halfVectors, axis=2, keepdims=True)
            cosines = np.clip(np.sum(normals * halfVectors, axis=2, keepdims=True), 0, None)
            specular = SPHERE_SPECULAR * cosines ** SPHERE_SHININESS * sphereMask
            image = (albedo * shading + specular) * intensities[i] + ambient + rng.normal(0, noise, albedo.shape)
            writeExr(folder / f"img_{i:02d}.exr", image)
        writeExr(folder / "img_ambient.exr", ambient)
        writeExr(root / "groundTruth" / f"{name}_normals.exr", normals)
        writeExr(root / "groundTruth" / f"{name}_albedo.exr", albedo)
        groundTruth[name] = normals

    (root / "masks").mkdir(exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    ex, ey, ax, ay = MASK_ELLIPSE
    mask = ((((xx - ex * width) / (ax * width)) ** 2 + ((yy - ey * height) / (ay * height)) ** 2) < 1).astype(np.uint8) * 255
    oiio.ImageBuf(np.ascontiguousarray(mask[..., None])).write(str(root / "masks" / "A.png"))

    sfmPath = root / "cameraInit.sfm"
    runCommand(["aliceVision_cameraInit", "--imageFolder", str(root / "images"), "--defaultFieldOfView", "45",
                "--allowSingleView", "1", "--output", str(sfmPath), "--verboseLevel", "error"])
    with open(sfmPath) as f:
        sfm = json.load(f)

    # Lights per view, in the LightingCalibration format, and calibration spheres per view, as a shape file
    lights, poseIds, sphereObservations = [], {}, {}
    for view in sfm["views"]:
        stem = os.path.splitext(os.path.basename(view["path"]))[0]
        name = os.path.basename(os.path.dirname(view["path"]))[3:]
        poseIds[name] = int(view["poseId"])
        cx, cy, radius = spheres[name]
        sphereObservations[view["viewId"]] = {"center": {"x": cx, "y": cy}, "radius": radius}
        if "ambient" in stem:
            continue
        i = int(stem.split("_")[1])
        lights.append({"viewId": view["viewId"], "type": "directional",
                       "direction": [str(float(v)) for v in directions[i]],
                       "intensity": [str(float(v)) for v in intensities[i]]})
    lightsPath = root / "lights.json"
    with open(lightsPath, "w") as f:
        json.dump({"lights": lights}, f, indent=4)
    spheresPath = root / "groundTruth" / "spheres.json"
    with open(spheresPath, "w") as f:
        json.dump({"shapes": [{"name": "Calibration Spheres", "type": "Circle", "properties": {"color": "green"},
                               "observations": sphereObservations}]}, f, indent=4)

    # Same SfMData with an arbitrary rotation per pose
    sfm["poses"] = []
    for i, poseId in enumerate(sorted({v["poseId"] for v in sfm["views"]})):
        rotation, _ = np.linalg.qr(np.random.default_rng(i).normal(size=(3, 3)))
        rotation *= np.sign(np.linalg.det(rotation))
        sfm["poses"].append({"poseId": poseId, "pose": {"transform": {"rotation": [str(v) for v in rotation.flatten()],
                                                                      "center": ["0", "0", "0"]}, "locked": "0"}})
    posedSfmPath = root / "cameraPoses.sfm"
    with open(posedSfmPath, "w") as f:
        json.dump(sfm, f, indent=4)

    return SimpleNamespace(root=root, width=width, height=height, sfmPath=str(sfmPath), posedSfmPath=str(posedSfmPath),
                           lightsPath=str(lightsPath), spheresPath=str(spheresPath), maskFolder=str(root / "masks"),
                           groundTruth=groundTruth, spheres=spheres, poseIds=poseIds,
                           directions=directions, intensities=intensities)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="Output folder (created if needed).")
    parser.add_argument("--width", type=int, default=320, help="Image width.")
    parser.add_argument("--height", type=int, default=240, help="Image height.")
    parser.add_argument("--lights", type=int, default=8, help="Number of lights (images) per pose.")
    parser.add_argument("--noise", type=float, default=0.002, help="Standard deviation of the Gaussian noise.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the noise.")
    args = parser.parse_args()

    if args.lights < 3:
        parser.error("At least 3 lights are needed.")
    if os.path.isdir(args.output) and os.listdir(args.output):
        parser.error(f"The output folder '{args.output}' is not empty.")

    dataset = createSyntheticDataset(args.output, args.width, args.height, args.lights, args.noise, args.seed)
    print(f"Synthetic dataset created in '{dataset.root}':")
    print(f"  {len(SPHERES)} poses ({', '.join(f'ps_{n}: pose id {p}' for n, p in dataset.poseIds.items())}), "
          f"{args.lights} lights + ambient, {args.width}x{args.height}")
    print(f"  SfMData:        {dataset.sfmPath}")
    print(f"  SfMData+poses:  {dataset.posedSfmPath}")
    print(f"  Lights:         {dataset.lightsPath}")
    print(f"  Masks:          {dataset.maskFolder}")
    print(f"  Ground truth:   {dataset.root / 'groundTruth'} (normals, albedo, spheres.json)")


if __name__ == "__main__":
    main()
