__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class SphereDetectionPy(desc.Node):
    """
Detect spheres in pictures. These spheres will be used for lighting calibration.
Spheres can be automatically detected or manually defined in the interface.

Python implementation of the AliceVision SphereDetection node (same inputs and outputs), running the detection
network with ONNX Runtime, on GPU if available.
"""

    category = "Photometric Stereo"
    gpu = lambda node: desc.Level.NORMAL if node.attribute("autoDetect").value and node.attribute("useGpu").value else desc.Level.NONE

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="modelPath",
            label="Detection Network",
            description="Deep learning network for automatic calibration sphere detection.",
            value="${ALICEVISION_SPHERE_DETECTION_MODEL}",
        ),
        desc.BoolParam(
            name="autoDetect",
            label="Automatic Sphere Detection",
            description="Automatic detection of calibration spheres.",
            value=False,
        ),
        desc.Circle(
            name="sphereShape",
            label="Sphere Shape",
            description="The shape of the calibration sphere for every image.",
            enabled=lambda node: not node.autoDetect.value,
            keyable=True,
            keyType="viewId",
        ),
        desc.File(
            name="sphereFile",
            label="Sphere Shape File",
            description="An input JSON file containing the shapes for every image. If provided, "
                        "the shapes provided with \"Sphere Shape\" will be ignored.",
            semantic="shapeFile",
            value="",
            enabled=lambda node: not node.autoDetect.value,
        ),
        desc.BoolParam(
            name="fillMissingSpheres",
            label="Fill Missing Spheres",
            description="Checked if a sphere position is to be written as detected although it "
                        "was not provided. In that case, the position of the last known sphere "
                        "will be used.",
            value=False,
            enabled=lambda node: not node.autoDetect.value,
        ),
        desc.FloatParam(
            name="minScore",
            label="Minimum Score",
            description="Minimum score for the detection.",
            value=0.0,
            range=(0.0, 50.0, 0.01),
            enabled=lambda node: node.autoDetect.value,
            advanced=True,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Run the detection network on GPU if available.\n"
                        "Off by default: the CUDA provider of the pyonnxruntime-1.30.0 package fails to load.",
            value=False,
            invalidate=False,
            enabled=lambda node: node.autoDetect.value,
            advanced=True,
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
            name="output",
            label="Output Path",
            description="Sphere detection information will be written here.",
            semantic="shapeFile",
            value="{nodeCacheFolder}/detection.json",
        )
    ]

    def processChunk(self, chunk):
        # Imported here: these dependencies are only available in the node's execution environment
        from photometricStereoHalide import sphereDetection

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            node = chunk.node
            if node.autoDetect.value:
                sphereDetection.autoDetectSpheres(node.input.value, node.modelPath.value, node.output.value,
                                                  minScore=node.minScore.value, useGpu=node.useGpu.value,
                                                  logger=chunk.logger)
            elif node.sphereFile.value:
                sphereDetection.writeSpheresFromFile(node.input.value, node.sphereFile.value, node.output.value,
                                                     node.fillMissingSpheres.value, logger=chunk.logger)
            else:
                sphereDetection.writeManualSpheres(node.input.value, node.sphereShape.geometry.getValueAsDict(),
                                                   node.output.value, node.fillMissingSpheres.value, logger=chunk.logger)
        finally:
            chunk.logManager.end()
