__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class LightingCalibrationPy(desc.Node):
    """
Evaluate the lighting in a scene using spheres placed in the scene.
Can also be used to calibrate a lighting dome (RTI type).

Python implementation of the AliceVision LightingCalibration node (same inputs and outputs), in which the
'whiteSphere' and 'SH' methods work. Only the area of the sphere is read from the images when the format allows it.
"""

    category = "Photometric Stereo"

    inputs = [
        desc.File(
            name="inputPath",
            label="Input SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="inputDetection",
            label="Sphere Detection File",
            description="Input JSON file containing sphere centers and radiuses.",
            value="",
        ),
        desc.BoolParam(
            name="saveAsModel",
            label="Save As Model",
            description="Check if this calibration file will be used with other datasets.",
            value=False,
        ),
        desc.BoolParam(
            name="ellipticEstimation",
            label="Use elliptic estimation",
            description="Consider the right projection of the sphere. Fit the circle tool on the small axe of the ellipse.\n"
                        "The sphere is recovered in 3D from the pinhole camera intrinsics: the normals and viewing\n"
                        "directions take the perspective into account.",
            value=False,
        ),
        desc.ChoiceParam(
            name="method",
            label="Calibration Method",
            description="Method used for light calibration.\n"
                        "Use 'brightestPoint' for shiny spheres and 'whiteSphere' for white matte spheres.\n"
                        "Spherical Harmonic lighting can be estimated using 'SH' method.",
            values=["brightestPoint", "whiteSphere", "SH"],
            value="brightestPoint",
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="outputFile",
            label="Light File",
            description="Light information will be written here.",
            value="{nodeCacheFolder}/lights.json",
        ),
        desc.File(
            name="lightingEstimationVisualization",
            label="Estimated Lighting Visualization",
            description="Estimated Lighting Visualization.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>_{methodValue}.png",
        ),
    ]

    def processChunk(self, chunk):
        # Imported here: these dependencies are only available in the node's execution environment
        from photometricStereoHalide.lightingCalibration import lightingCalibration

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            node = chunk.node
            lightingCalibration(node.inputPath.value, node.inputDetection.value, node.outputFile.value,
                                method=node.method.value, saveAsModel=node.saveAsModel.value,
                                visualizationFolder=os.path.dirname(node.outputFile.value),
                                ellipticEstimation=node.ellipticEstimation.value, logger=chunk.logger)
        finally:
            chunk.logManager.end()
