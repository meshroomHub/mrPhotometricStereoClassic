"""
Halide pipelines for the photometric stereo.

All the images are planar float32 numpy arrays of shape (channels, height, width), which map to
Halide buffers indexed as (x, y, channel).

The pipelines are JIT-compiled on first use and then reused: every per-image or per-pose value
(light, intensity, rotation...) is a runtime Param, so nothing is recompiled between calls.
"""

import halide as hl
import numpy as np

# Luminance weights (Rec. 709) used to estimate the normals from the RGB images
LUMINANCE = (0.2126, 0.7152, 0.0722)

# Width of the SIMD vectors (8 x float32 for AVX2)
VECTOR_SIZE = 8

_x = hl.Var("x")
_y = hl.Var("y")
_c = hl.Var("c")
_n = hl.Var("n")


def _toBuffer(array: np.ndarray) -> hl.Buffer:
    """ Wrap a C-contiguous numpy array into a Halide buffer, without any copy. """
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Halide buffers require C-contiguous numpy arrays.")
    return hl.Buffer(array)


def _vectorizeX(func, *innerVars):
    """
    Default schedule for a per-pixel Func: the given innermost pure variables (e.g. the channel) are
    unrolled, x is vectorized and rows are processed in parallel.
    GuardWithIf is required for the pipelines that update a buffer in place: the default ShiftInwards
    strategy would evaluate some pixels twice on the last vector of each row.
    """
    xo, xi = hl.Var("xo"), hl.Var("xi")
    func.split(_x, xo, xi, VECTOR_SIZE, hl.TailStrategy.GuardWithIf)
    func.reorder(xi, *innerVars, xo, _y)
    func.vectorize(xi)
    for v in innerVars:
        func.unroll(v)
    func.parallel(_y)


class AccumulatePipeline:
    """
    Add the contribution of one image to the per-pixel least squares solution.

    The least squares solution of L.m = I is m = pinv(L).I = sum_i pinv(L)[:, i] * I_i: it is linear
    in the images, so it can be accumulated one image at a time without keeping the whole image stack
    in memory.

    The accumulator has 9 planes: plane 3 * c + j stores the j-th component of the solution for the
    color channel c. The luminance solution (used for the normals) is the luminance of these 3 solutions.
    """

    def __init__(self):
        self.image = hl.ImageParam(hl.Float(32), 3, "image")
        self.ambient = hl.ImageParam(hl.Float(32), 3, "ambient")
        self.accIn = hl.ImageParam(hl.Float(32), 3, "accIn")
        self.ambientWeight = hl.Param(hl.Float(32), "ambientWeight", 0.0)
        self.invIntensity = [hl.Param(hl.Float(32), f"invIntensity{c}", 1.0) for c in range(3)]
        self.pinvColumn = [hl.Param(hl.Float(32), f"pinvColumn{j}", 0.0) for j in range(3)]

        k = hl.Var("k")
        # Image without ambient light, divided by the light intensity
        value = [(self.image[_x, _y, c] - self.ambientWeight * self.ambient[_x, _y, c]) * self.invIntensity[c] for c in range(3)]
        contribution = [self.pinvColumn[j] * value[c] for c in range(3) for j in range(3)]

        self.accOut = hl.Func("accumulate")
        self.accOut[_x, _y, k] = self.accIn[_x, _y, k] + hl.mux(k, contribution)
        self.accOut.bound(k, 0, 9)
        _vectorizeX(self.accOut, k)
        self.accOut.compile_jit()

    def __call__(self, acc, image, ambient, intensity, pinvColumn):
        """
        Update "acc" in place (safe as each pixel only reads itself).

        Args:
            acc: (9, H, W) accumulator.
            image: (3, H, W) image.
            ambient: (3, H, W) ambient image to subtract, or None.
            intensity: the 3 light intensities (one per channel).
            pinvColumn: the 3 values of the pseudo-inverse of the light matrix for this image.
        """
        accBuffer = _toBuffer(acc)
        self.accIn.set(accBuffer)
        self.image.set(_toBuffer(image))
        # Without ambient, the image is bound as ambient with a null weight to keep the bounds valid
        self.ambient.set(_toBuffer(ambient if ambient is not None else image))
        self.ambientWeight.set(1.0 if ambient is not None else 0.0)
        for c in range(3):
            self.invIntensity[c].set(float(1.0 / intensity[c]))
            self.pinvColumn[c].set(float(pinvColumn[c]))
        self.accOut.realize(accBuffer)


class FinalizePipeline:
    """
    Compute the normals and the (non-normalized) albedo from the accumulated least squares solutions.
      - normal = normalized luminance solution
      - albedo[c] = norm of the solution of the color channel c
    Pixels outside of the mask are set to 0.
    """

    def __init__(self):
        self.acc = hl.ImageParam(hl.Float(32), 3, "acc")
        self.mask = hl.ImageParam(hl.Float(32), 2, "mask")

        m = [sum(hl.f32(LUMINANCE[ch]) * self.acc[_x, _y, 3 * ch + j] for ch in range(3)) for j in range(3)]
        norm2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2]
        inMask = hl.f64(self.mask[_x, _y]) > hl.f64(0.7)
        invNorm = hl.select(inMask & (norm2 > 0.0), 1.0 / hl.sqrt(norm2), 0.0)
        albedo = [hl.select(inMask, hl.sqrt(sum(self.acc[_x, _y, 3 * ch + j] ** 2 for j in range(3))), 0.0) for ch in range(3)]

        self.result = hl.Func("finalize")
        self.result[_x, _y, _c] = hl.Tuple([hl.mux(_c, m) * invNorm, hl.mux(_c, albedo)])
        self.result.bound(_c, 0, 3)
        _vectorizeX(self.result, _c)
        self.result.compile_jit()

    def __call__(self, acc, mask, normals, albedo):
        """ Fill the (3, H, W) "normals" and "albedo" arrays from the (9, H, W) accumulator and the (H, W) mask. """
        self.acc.set(_toBuffer(acc))
        self.mask.set(_toBuffer(mask))
        self.result.realize([_toBuffer(normals), _toBuffer(albedo)])


class NormalizePipeline:
    """ Normalize a (3, H, W) vector field (null vectors and pixels outside of the mask are set to 0). """

    def __init__(self):
        self.vectors = hl.ImageParam(hl.Float(32), 3, "vectors")
        self.mask = hl.ImageParam(hl.Float(32), 2, "mask")

        v = [self.vectors[_x, _y, j] for j in range(3)]
        norm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        invNorm = hl.select((hl.f64(self.mask[_x, _y]) > hl.f64(0.7)) & (norm2 > 0.0), 1.0 / hl.sqrt(norm2), 0.0)

        self.result = hl.Func("normalize")
        self.result[_x, _y, _c] = self.vectors[_x, _y, _c] * invNorm
        self.result.bound(_c, 0, 3)
        _vectorizeX(self.result, _c)
        self.result.compile_jit()

    def __call__(self, vectors, mask, out):
        self.vectors.set(_toBuffer(vectors))
        self.mask.set(_toBuffer(mask))
        self.result.realize(_toBuffer(out))


class RotatePipeline:
    """ Apply a 3x3 rotation to a (3, H, W) normal map. """

    def __init__(self):
        self.normals = hl.ImageParam(hl.Float(32), 3, "normals")
        self.rotation = [[hl.Param(hl.Float(32), f"r{i}{j}", 0.0) for j in range(3)] for i in range(3)]

        rotated = [sum(self.rotation[i][j] * self.normals[_x, _y, j] for j in range(3)) for i in range(3)]
        self.result = hl.Func("rotate")
        self.result[_x, _y, _c] = hl.mux(_c, rotated)
        self.result.bound(_c, 0, 3)
        _vectorizeX(self.result, _c)
        self.result.compile_jit()

    def __call__(self, normals, rotation, out):
        self.normals.set(_toBuffer(normals))
        for i in range(3):
            for j in range(3):
                self.rotation[i][j].set(float(rotation[i, j]))
        self.result.realize(_toBuffer(out))


class NormalsToPngPipeline:
    """
    Encode a (3, H, W) normal map in false colors as a (3, H, W) uint8 image.
    Reproduces the encoding of AliceVision's convertNormalMap2png, including the wrap-around of the
    negative values on the y and z components (computed in double precision, like the C++ code).
    """

    def __init__(self):
        self.normals = hl.ImageParam(hl.Float(32), 3, "normals")

        n = [hl.cast(hl.Float(64), self.normals[_x, _y, j]) for j in range(3)]
        isNull = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) == 0.0
        r = hl.cast(hl.Int(32), hl.floor(255.0 * (n[0] + 1.0) / 2.0))
        g = -hl.cast(hl.Int(32), hl.ceil(255.0 * (n[1] + 1.0) / 2.0))
        b = -hl.cast(hl.Int(32), hl.ceil(255.0 * n[2]))

        self.result = hl.Func("normalsToPng")
        self.result[_x, _y, _c] = hl.select(isNull, hl.cast(hl.UInt(8), 0), hl.cast(hl.UInt(8), hl.mux(_c, [r, g, b])))
        self.result.bound(_c, 0, 3)
        _vectorizeX(self.result, _c)
        self.result.compile_jit()

    def __call__(self, normals, out):
        self.normals.set(_toBuffer(normals))
        self.result.realize(_toBuffer(out))


class RobustSolvePipeline:
    """
    Least squares solution of the robust (ADMM) algorithm:
        M = pinv(L) . (I + E - W / mu)
    with I the (N, H, W) luminance images, E the (N, H, W) sparse errors and W the (N, H, W) Lagrange multipliers.
    """

    def __init__(self):
        self.gray = hl.ImageParam(hl.Float(32), 3, "gray")
        self.errors = hl.ImageParam(hl.Float(32), 3, "errors")
        self.multipliers = hl.ImageParam(hl.Float(32), 3, "multipliers")
        self.pinv = hl.ImageParam(hl.Float(32), 2, "pinv")  # (n, j)
        self.mu = hl.Param(hl.Float(32), "mu", 0.1)

        target = hl.Func("target")
        target[_x, _y, _n] = self.gray[_x, _y, _n] + self.errors[_x, _y, _n] - self.multipliers[_x, _y, _n] / self.mu

        r = hl.RDom([(0, self.gray.dim(2).extent())], "r")
        self.result = hl.Func("robustSolve")
        self.result[_x, _y, _c] = 0.0
        self.result[_x, _y, _c] += self.pinv[r.x, _c] * target[_x, _y, r.x]
        self.result.bound(_c, 0, 3)
        _vectorizeX(self.result, _c)
        # Loop over the images outside of the vectorized pixels: each target value is used for the 3 components
        xo, xi = hl.Var("xo"), hl.Var("xi")
        update = self.result.update(0)
        update.split(_x, xo, xi, VECTOR_SIZE, hl.TailStrategy.GuardWithIf)
        update.reorder(xi, _c, xo, r.x, _y)
        update.vectorize(xi).unroll(_c).parallel(_y)
        self.result.compile_jit()

    def __call__(self, gray, errors, multipliers, pinv, mu, out):
        self.gray.set(_toBuffer(gray))
        self.errors.set(_toBuffer(errors))
        self.multipliers.set(_toBuffer(multipliers))
        self.pinv.set(_toBuffer(pinv))
        self.mu.set(float(mu))
        self.result.realize(_toBuffer(out))


class RobustUpdatePipeline:
    """
    Error and Lagrange multiplier updates of the robust (ADMM) algorithm:
        R = L.M - I
        E = shrink(R + W / mu, 1 / mu)
        W = W + mu * (R - E)
    E and W are updated in place (each pixel only reads itself).
    """

    def __init__(self):
        self.gray = hl.ImageParam(hl.Float(32), 3, "gray")
        self.multipliersIn = hl.ImageParam(hl.Float(32), 3, "multipliersIn")
        self.solution = hl.ImageParam(hl.Float(32), 3, "solution")
        self.lights = hl.ImageParam(hl.Float(32), 2, "lights")  # (j, n)
        self.mu = hl.Param(hl.Float(32), "mu", 0.1)

        residual = sum(self.lights[j, _n] * self.solution[_x, _y, j] for j in range(3)) - self.gray[_x, _y, _n]
        w = self.multipliersIn[_x, _y, _n]
        t = residual + w / self.mu
        # Soft thresholding
        e = hl.select(t > 0.0, 1.0, -1.0) * hl.max(hl.abs(t) - 1.0 / self.mu, 0.0)

        self.result = hl.Func("robustUpdate")
        self.result[_x, _y, _n] = hl.Tuple([e, w + self.mu * (residual - e)])
        xo, xi = hl.Var("xo"), hl.Var("xi")
        self.result.split(_x, xo, xi, VECTOR_SIZE, hl.TailStrategy.GuardWithIf)
        self.result.vectorize(xi).parallel(_y)
        self.result.compile_jit()

    def __call__(self, gray, multipliers, solution, lights, mu, errors):
        """ Update the (N, H, W) "errors" and "multipliers" arrays in place. """
        multipliersBuffer = _toBuffer(multipliers)
        self.gray.set(_toBuffer(gray))
        self.multipliersIn.set(multipliersBuffer)
        self.solution.set(_toBuffer(solution))
        self.lights.set(_toBuffer(lights))
        self.mu.set(float(mu))
        self.result.realize([_toBuffer(errors), multipliersBuffer])


_pipelines = {}


def getPipeline(pipelineClass):
    """ Return the JIT-compiled instance of a pipeline class, compiling it on first use. """
    if pipelineClass not in _pipelines:
        _pipelines[pipelineClass] = pipelineClass()
    return _pipelines[pipelineClass]
