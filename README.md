# mrPhotometricStereo

Meshroom plugin providing **PhotometricStereoHalide**, a Python/[Halide](https://halide-lang.org) implementation of
AliceVision's `PhotometricStereo` node. It has the same inputs and outputs, so it is a drop-in replacement in the
photometric stereo pipeline (`CameraInit → SphereDetection → LightingCalibration → PhotometricStereo`).

## Layout

```
repo/
├── meshroom/
│   ├── photometricStereo/PhotometricStereoHalide.py   # Meshroom node
│   └── photometricStereoHalide.mgt                    # Pipeline template using the Halide node
├── photometricStereoHalide/                           # Implementation (importable from the plugin root)
│   ├── halidePipelines.py                             # Halide pipelines (per-pixel computations)
│   ├── photometricStereo.py                           # Per-pose orchestration
│   └── dataIO.py                                      # SfMData, light files, image read/write (pyalicevision)
└── pyproject.toml
```

The plugin is a rez plugin: `../package.py` registers it in `MESHROOM_REZ_PLUGINS` and defines the node's execution
environment in `MRPHOTOMETRICSTEREO_SUBREQUIRES` (`aliceVision`, `pyoiio`, `pyhalide`). The plugin root is added to the
`PYTHONPATH` of the execution environment, which makes `photometricStereoHalide` importable.

## Algorithm

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
rez env python-3.11 pyhalide pyoiio aliceVision-develop pytest -- python -m pytest tests
```

- `test_halidePipelines.py`: each Halide pipeline against a numpy reference on random data (odd widths to cover
  the vectorization tails of the in-place updates).
- `test_dataIO.py`: light file formats (JSON per view or as a model, SH, `.lp`, folder) and grouping of the views by pose.
- `test_photometricStereo.py`: end-to-end on a synthetic dataset (2 poses × 8 lights + ambient, generated with
  `aliceVision_cameraInit`): accuracy against the ground truth normals, outputs, masks, world normals, robustness to
  shadows, errors, and parity with `aliceVision_photometricStereo` (skipped if the binaries are not available).
- `test_meshroomNode.py`: node descriptor and template consistency, skipped without Meshroom. To run it too:
  `rez env meshroom-develop pyhalide pyoiio pytest -- python -m pytest tests`.

### Synthetic dataset

The dataset used by the tests can be created on disk to try the node manually (in Meshroom: import the `images`
folder, or use `cameraInit.sfm` and `lights.json` directly as inputs of the node):

```
rez env python-3.11 pyoiio aliceVision-develop -- python tests/makeSyntheticDataset.py <outputFolder> [--width 160] [--height 120] [--lights 8] [--noise 0.002] [--seed 0]
```

It contains 2 poses (`images/ps_A`, `images/ps_B`) of `--lights` EXR images plus an ambient image, a mask for pose A
(`masks/A.png`), the ground truth normals and albedo (`groundTruth/`), `cameraInit.sfm`, `cameraPoses.sfm` (with an
arbitrary rotation per pose, to test the world normals) and `lights.json`.

## Differences with the AliceVision C++ node

Checked on a synthetic dataset (2 poses × 8 lights + ambient): camera normals and albedo match to float32 rounding
(max difference < 5e-7), the PNGs differ by at most 1 level on a few pixels, and the output SfMData files are identical,
in default, mask + ambient removal and downscale modes. The differences come from fixes:

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
