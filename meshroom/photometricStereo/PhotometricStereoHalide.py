__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class PhotometricStereoHalide(desc.Node):
    """
Reconstruction using Photometric Stereo, implemented in Python with Halide.
A normal map is evaluated from several photographs taken from the same point of view, but under different lighting conditions.
The lighting conditions are assumed to be known.

Drop-in replacement of the AliceVision PhotometricStereo node (same inputs and outputs).
The per-pixel computations are JIT-compiled Halide pipelines (vectorized and multi-threaded), and the images are
streamed one at a time in the default (non-robust) mode, so the memory usage does not depend on the number of lights.
"""

    size = desc.DynamicNodeSize("inputPath")
    category = "Photometric Stereo"
    cpu = desc.Level.INTENSIVE
    # The robust algorithm keeps all the images in memory
    ram = lambda node: desc.Level.INTENSIVE if node.attribute("isRobust").value else desc.Level.NORMAL

    inputs = [
        desc.File(
            name="inputPath",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="pathToJSONLightFile",
            label="Light File",
            description="Path to the lighting information: a JSON file (from LightingCalibration), a .lp file,\n"
                        "or a folder containing light_directions.txt and light_intensities.txt.",
            value="defaultJSON.txt",
        ),
        desc.File(
            name="maskPath",
            label="Mask Folder Path",
            description="Path to a folder containing masks or to a mask directly.\n"
                        "In a folder, the mask of the images in 'ps_<name>/' is '<name>.png'.",
            value="",
        ),
        desc.BoolParam(
            name="removeAmbient",
            label="Remove Ambient Light",
            description="True if the ambient light is to be removed on the PS images, false otherwise.\n"
                        "The ambient image is the image of the pose containing 'ambient' in its filename.",
            value=False,
            advanced=True,
        ),
        desc.BoolParam(
            name="isRobust",
            label="Use Robust Algorithm",
            description="True to use the robust algorithm (L1 fitting, robust to shadows and highlights), false otherwise.\n"
                        "All the images of a pose are kept in memory.",
            value=False,
            advanced=True,
        ),
        desc.IntParam(
            name="downscale",
            label="Downscale Factor",
            description="Downscale factor for faster results.",
            value=1,
            range=(1, 10, 1),
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
            name="outputPath",
            label="Output Folder",
            description="Path to the output folder.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputSfmDataAlbedo",
            label="SfMData Albedo",
            description="Output SfMData file containing the albedo information.",
            value="{nodeCacheFolder}/albedoMaps.sfm",
        ),
        desc.File(
            name="outputSfmDataNormal",
            label="SfMData Normal",
            description="Output SfMData file containing the normal maps information.",
            value="{nodeCacheFolder}/normalMaps.sfm",
        ),
        desc.File(
            name="outputSfmDataNormalPNG",
            label="SfMData Normal PNG",
            description="Output SfMData file containing the normal maps information.",
            value="{nodeCacheFolder}/normalMapsPNG.sfm",
        ),
        # These attributes are only here to describe more accurately the output of the node
        # by specifying that it generates sequences of images (see in Viewer2D.qml how they can be used)
        desc.File(
            name="normals",
            label="Normal Maps Camera",
            description="Generated normal maps in the camera coordinate system.",
            semantic="image",
            value="{nodeCacheFolder}/<POSE_ID>_normals.exr",
        ),
        desc.File(
            name="normalsPNG",
            label="Normal Maps Camera (in false colors)",
            description="Generated normal maps in the camera coordinate system (in false colors).",
            semantic="image",
            value="{nodeCacheFolder}/<POSE_ID>_normals.png",
        ),
        desc.File(
            name="normalsWorld",
            label="Normal Maps World",
            description="Generated normal maps in the world coordinate system.",
            semantic="image",
            value="{nodeCacheFolder}/<POSE_ID>_normals_w.exr",
        ),
        desc.File(
            name="albedo",
            label="Albedo Maps",
            description="Generated albedo maps.",
            semantic="image",
            value="{nodeCacheFolder}/<POSE_ID>_albedo.png",
        ),
    ]

    def processChunk(self, chunk):
        # Imported here: these dependencies are only available in the node's execution environment
        from photometricStereoHalide.photometricStereo import photometricStereo, PhotometricStereoParameters

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            params = PhotometricStereoParameters(
                removeAmbient=chunk.node.removeAmbient.value,
                isRobust=chunk.node.isRobust.value,
                downscale=chunk.node.downscale.value,
            )
            photometricStereo(
                sfmPath=chunk.node.inputPath.value,
                lightPath=chunk.node.pathToJSONLightFile.value,
                maskPath=chunk.node.maskPath.value,
                outputFolder=chunk.node.outputPath.value,
                params=params,
                logger=chunk.logger,
            )
        finally:
            chunk.logManager.end()
