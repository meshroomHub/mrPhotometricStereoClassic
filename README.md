# mrPhotometricStereo

Meshroom plugin providing Python implementations of AliceVision photometric stereo nodes, with the same inputs and
outputs, so they are drop-in replacements in the photometric stereo pipeline
(`CameraInit → SphereDetection → LightingCalibration → PhotometricStereo`):

- **SphereDetectionPy**: detection of the calibration spheres, running the AliceVision detection network with
  [ONNX Runtime](https://onnxruntime.ai), or manually defined spheres.
- **LightingCalibrationPy**: light directions and intensities from the calibration spheres (numpy), reading only the
  area of the spheres from the images.
- **PhotometricStereoHalide**: normal and albedo maps, with the per-pixel computations in [Halide](https://halide-lang.org).

The `photometricStereoHalide` pipeline template uses the 3 Python nodes.

## Layout

```
repo/
├── meshroom/
│   ├── photometricStereo/PhotometricStereoHalide.py   # Meshroom nodes
│   ├── photometricStereo/SphereDetectionPy.py
│   ├── photometricStereo/LightingCalibrationPy.py
│   └── photometricStereoHalide.mgt                    # Pipeline template using the Python nodes
├── photometricStereoHalide/                           # Implementation (importable from the plugin root)
│   ├── halidePipelines.py                             # Halide pipelines (per-pixel computations)
│   ├── photometricStereo.py                           # Per-pose orchestration
│   ├── sphereDetection.py                             # Calibration sphere detection (ONNX Runtime)
│   ├── lightingCalibration.py                         # Lighting calibration from the spheres
│   └── dataIO.py                                      # SfMData, light files, image read/write (pyalicevision)
├── tests/
└── pyproject.toml
```

The plugin is a rez plugin: `../package.py` registers it in `MESHROOM_REZ_PLUGINS` and defines the nodes' execution
environment in `MRPHOTOMETRICSTEREO_SUBREQUIRES` (`aliceVision`, `pyoiio`, `pyhalide`, `pyonnxruntime`). The plugin
root is added to the `PYTHONPATH` of the execution environment, which makes `photometricStereoHalide` importable.

## SphereDetectionPy

- **Automatic detection** (`autoDetect`): each image (except the ambient ones) is read in sRGB like AliceVision and
  given to the Mask R-CNN network (`ALICEVISION_SPHERE_DETECTION_MODEL`). The best box with a score above `minScore`
  gives the sphere: radius `min(width, height) / 2`, center at the top-left corner of the box plus the radius, as in
  AliceVision. Only the boxes and scores are requested, so the mask branch of the network is not computed.
- **Manual spheres**: from the keyable `sphereShape` circle (set per view in the 2D viewer) or from a shape file
  (`sphereFile`), optionally completed with the last known sphere for the views without one (`fillMissingSpheres`).

The output is the same shape file as AliceVision (`detection.json`), with JSON numbers instead of strings.

**GPU**: `useGpu` is off by default. The CUDA provider of the `pyonnxruntime-1.30.0` rez package fails to load
(`libonnxruntime_providers_cuda.so: undefined symbol onnxruntime::GetEnvironmentVar`, even in an environment with only
`pyonnxruntime`), and ONNX Runtime silently falls back to the CPU: the node logs a warning when this happens.
On CPU, the detection takes ~0.8s per image at 320x240, like the C++ node.

## LightingCalibrationPy

One light per image with a sphere (ambient images excluded), written in the same JSON file as AliceVision
(`lights.json`, lights identified by `viewId`, or by `lightId` with `saveAsModel`):

- **brightestPoint** (shiny spheres): the sphere is cut out of the image and smoothed with a triangle kernel; the light
  is the mirror reflection of the viewing direction (orthographic camera) on the normal of the brightest point.
  Same algorithm as AliceVision. Intensity 1.
- **whiteSphere** (white matte spheres): Lambertian least squares fit `normal . L = I` on the sphere pixels within 95%
  of the radius whose value is between 0.2 and 0.8 (neither in shadow nor saturated). Direction `L / |L|`,
  intensity `|L|`.
- **SH**: the whiteSphere fit for the directional part, then the 6 other terms of the AliceVision second order
  spherical harmonics fitted on the residual. The photometric stereo only uses the directional part.

**Sphere model**:

- Orthographic (default): the sphere image is the detected circle, the normals come from the pixel position relative
  to its center, and the viewing direction is `(0, 0, -1)` everywhere.
- Perspective (`ellipticEstimation`): the image of a sphere seen by a pinhole camera is an ellipse, whose minor axis is
  the detected circle. For a sphere of angular radius `α` whose center direction makes an angle `θ` with the optical
  axis, the minor semi-axis is `b = f.sin(α) / sqrt(cos²θ - sin²α)` and the ellipse center is at the distance
  `ρ = f.sin(θ).cos(θ) / (cos²θ - sin²α)` from the principal point. Inverting: `tan(θ) = ρ.f / (b² + f²)` and
  `sin(α) = cos(θ).b / sqrt(b² + f²)`, which gives the 3D center of a unit sphere from the circle and the pinhole
  intrinsics of the view (focal, principal point = image center + offset). The normals and viewing directions come
  from the camera rays intersecting the sphere, for all the methods (the mirror reflection of `brightestPoint` uses
  the viewing direction of the brightest pixel). Only pinhole intrinsics are supported (distortion is ignored).

  On the synthetic dataset seen with a 90° (resp. 45°) field of view, with the ground truth circles, the lights are
  within 0.9 to 1.6 degrees (median) with the elliptic estimation, and 20 to 22 (resp. 12) degrees without it. It must
  only be used on images seen in perspective: on the orthographic synthetic dataset, it gives 11 to 14 degrees of error.
  The circles must be centered on the ellipses: the sphere detection network, which boxes the sphere image and
  inscribes the circle in the top-left corner of the box, does not give these circles on elongated ellipses (and it
  misses most of the spheres of the 90° synthetic dataset).

The images are read as luminance (`0.2126 R + 0.7152 G + 0.0722 B`, like AliceVision's grayscale reading), and only
the sphere area is read through OIIO, except for RAW images. The node also writes `<imageStem>_<method>.png` images of
a sphere lit by each estimated light, like AliceVision (images with the same file name in different folders share it).

On 16 images of 2000x1500: 0.6 to 0.8s against 1.0 to 1.2s for the C++ node, half of it being image reading.

## PhotometricStereoHalide

For each pose (group of views sharing the same pose id, i.e. images in the same `ps_*` folder):

- **Default**: per-pixel least squares `L.m = I`. As `m = pinv(L).I` is linear in the images, the solution is
  accumulated one image at a time (`AccumulatePipeline`): memory does not depend on the number of lights.
  Normals are the normalized luminance solution, albedo the norm of each color channel's solution.
- **Robust** (`isRobust`): L1 fitting of the luminance with ADMM, each iteration being two Halide pipelines
  (`RobustSolvePipeline`, `RobustUpdatePipeline`). The penalty grows by 10% per iteration, and the iterations stop when
  the relative primal residual `|L.M - I - E| / |I|` is below 1e-5 (checked in numpy every 10 iterations): ~110
  iterations on the synthetic dataset, where it halves the normal error on the pixels in shadow for some lights
  (5.2 to 3.0 degrees) without changing the others. The albedo is the median, over the lit images, of the
  image/shading ratios. All the images of a pose are kept in memory.
- Post-processing in Halide: camera to world rotation of the normals and false-color PNG encoding.

The pipelines are JIT-compiled once per process (~1s), then vectorized (AVX2) and multi-threaded.

On 12 images of 4000x3000 (16 cores), the node takes 4.1s against 10.7s for the C++ node: ~0.6s of Halide compute,
~0.7s of JIT compilation, the rest being image reading and writing.

## Tests

```
rez env python-3.11 pyhalide pyoiio pyonnxruntime aliceVision-develop pytest -- python -m pytest tests
```

- `test_halidePipelines.py`: each Halide pipeline against a numpy reference on random data (odd widths to cover
  the vectorization tails of the in-place updates).
- `test_dataIO.py`: light file formats (JSON per view or as a model, SH, `.lp`, folder) and grouping of the views by pose.
- `test_photometricStereo.py`: end-to-end on the synthetic dataset: accuracy against the ground truth normals,
  outputs, masks, world normals, robustness to shadows, errors, and parity with `aliceVision_photometricStereo`.
- `test_sphereDetection.py`: automatic detection against the ground truth spheres, and parity of the automatic, manual
  and sphere file modes with `aliceVision_sphereDetection`. The automatic detection runs once for the session
  (~15s on CPU).
- `test_lightingCalibration.py`: partial image reading against AliceVision's, each method against the ground truth
  lights and intensities, `saveAsModel`, skipped views, parity of `brightestPoint` (lights and visualizations) with
  `aliceVision_lightingCalibration`, the perspective sphere model (geometry against rendered spheres, every method
  with and without `ellipticEstimation` on a 90° perspective dataset, pinhole intrinsics), and the full chain of Python
  nodes (detected spheres → lighting calibration → photometric stereo) against the ground truth lights and normals.
- `test_meshroomNode.py`: node descriptors, resources, keyable `sphereShape` values and template consistency,
  skipped without Meshroom. To run it too:
  `rez env meshroom-develop pyhalide pyoiio pyonnxruntime pytest -- python -m pytest tests`.

The tests comparing with AliceVision are skipped if the binaries are not available.

### Synthetic dataset

The dataset used by the tests can be created on disk to try the nodes or the whole template (in Meshroom: import the
`images` folder, or use `cameraInit.sfm`, `lights.json` or `groundTruth/spheres.json` directly as node inputs):

```
rez env python-3.11 pyoiio aliceVision-develop -- python tests/makeSyntheticDataset.py <outputFolder> [--width 320] [--height 240] [--lights 8] [--noise 0.002] [--seed 0] [--perspective] [--fieldOfView 45]
```

With `--perspective`, the calibration spheres are rendered with the pinhole camera of CameraInit (horizontal field of
view `--fieldOfView`): their images are ellipses, and `groundTruth/spheres.json` contains the circles on their minor
axes, measured on the rendered silhouettes. The spheres are shrunk if needed to stay inside the image.

Each pose (`images/ps_A`, `images/ps_B`) is a Lambertian tilted plane with Gaussian bumps and a glossy white
calibration sphere, rendered as `--lights` EXR images plus an ambient image. The dataset also contains a mask for pose A
(`masks/A.png`), the ground truth normals, albedo and spheres (`groundTruth/`, the spheres as a shape file usable as
`sphereFile`), `cameraInit.sfm`, `cameraPoses.sfm` (with an arbitrary rotation per pose, to test the world normals)
and the ground truth `lights.json`.

With the default 320x240 images, the whole template runs in ~25s with `meshroom_batch`, and the fully automatic
normals are within 3 to 5 degrees (median) of the ground truth. The error comes from the detected spheres: the
network only boxes the lit part of the sphere on grazing lights, which shifts the detected sphere (lights within 8
degrees, median). With the ground truth spheres, the lights are within 1.5 degrees with all the methods (the
specular highlight of the sphere slightly biases the Lambertian fits).

## Differences with the AliceVision C++ nodes

### SphereDetectionPy

Checked on the synthetic dataset: the automatic detections match the C++ ones (max difference 2e-5 on the centers,
radii and scores), as well as the manual and sphere file modes. Differences:

- The values are written as JSON numbers (the C++ node writes strings); both are read by Meshroom and AliceVision.
- When no sphere is detected, `observations` is an empty object (`""` in C++).
- With `fillMissingSpheres` and manual spheres, the last known sphere is the one of the last view id in numeric order
  (lexicographic order of the view id strings in C++).
- The circles of the views whose radius is not set (default -1) are ignored.

### PhotometricStereoHalide

Checked on the synthetic dataset: camera normals and albedo match to float32 rounding (max difference < 5e-7), the
PNGs differ by at most 1 level on a few pixels, and the output SfMData files are identical, in default,
mask + ambient removal and downscale modes. The differences come from fixes:

- **World normals**: the C++ `applyRotation` rotates in place, reusing already rotated components, which produces
  non-unit, wrongly oriented world normals. Here the rotation is applied correctly.
- **Robust mode**: the C++ estimation stops after 11 iterations, as its convergence test (relative change of the
  solution) is met while ADMM has not moved away from the least squares solution yet: its result is the least squares
  one. Besides, it starts from an uninitialized matrix: depending on the run, some pixels end up with wrong normals.
  Here ADMM starts from 0, uses an increasing penalty and stops on the primal residual (see above). The C++ albedo
  median also uses all the images, including unlit ones (division by negative or null shading); here only the lit
  ones are used.
- **Ambient image**: in C++, the ambient image of a pose is reused by the next poses without one. Here it is per pose.
- **Mask output**: `<POSE_ID>_mask.png` is the full-resolution mask of the processed pixels (C++ writes a 1x1 image
  when no mask is given).
- Poses with fewer than 3 images are skipped with a warning (the system is underdetermined).
- `maskPath` can be a mask file, as stated in the node description (the C++ node only supports a folder).
- The unused `SHOrder` parameter is not exposed: like in C++, the lighting is always reduced to directional.

### LightingCalibrationPy

Checked on the synthetic datasets (320x240 with detected spheres, 2000x1500 with the ground truth spheres):
`brightestPoint` gives the same lights as the C++ node (max difference 2e-7), in the same order, and the same
visualizations (at most 1 level of difference). Differences:

- **whiteSphere**: the C++ node solves the least squares system on the full matrix, whose rows beyond the valid
  pixels are never written: its lights are wrong (35 to 75 degrees of error) or NaN, and change between runs. Here
  only the valid pixels are used: 1.2 degrees of error with the ground truth spheres.
- **whiteSphere/SH**: the normals are computed from the pixel position relative to the sphere center (relative to
  the corner of the integer patch in C++), and the SH residual removes the scaled directional part (the unit
  direction in C++). The C++ SH directions are correct and match these ones.
- The views sharing the same acquisition date are all calibrated (C++ keeps only one of them).
- Views whose sphere has fewer usable pixels than unknowns are skipped with a warning.
- `ellipticEstimation` works (see above). The C++ implementation crashes, as the mask used to fit the ellipse is never
  drawn; its intended model (3D sphere from the ellipse and the intrinsics, normals from the camera rays) is the one
  implemented here, with a closed form instead of an ellipse fitting. Unlike C++, it also supports the SH method.
- Views with a sphere smaller than a pixel are skipped with a warning (C++ reads an empty patch).
- The values are written as JSON numbers (the C++ node writes strings).
