"""
Tests of the Meshroom node descriptors and pipeline template (skipped when Meshroom is not available).
"""

import importlib.util
import json
import os
import re
import sys

import pytest

pytest.importorskip("meshroom.core")

from meshroom.core import desc, pluginManager  # noqa: E402
from meshroom.core.graph import Graph  # noqa: E402
from meshroom.core.plugins.base import NodeDescProvider  # noqa: E402

from photometricStereoHalide import sphereDetection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_FOLDER = os.path.join(REPO, "meshroom", "photometricStereo")
TEMPLATE_FILE = os.path.join(REPO, "meshroom", "photometricStereoHalide.mgt")
NODE_TYPES = ["PhotometricStereoHalide", "SphereDetectionPy", "LightingCalibrationPy"]


@pytest.fixture(scope="module")
def nodeDescs():
    """ Load the node descriptors and register them in Meshroom. """
    nodeDescs = {}
    for nodeType in NODE_TYPES:
        spec = importlib.util.spec_from_file_location(nodeType, os.path.join(NODES_FOLDER, nodeType + ".py"))
        module = importlib.util.module_from_spec(spec)
        # Registered module: required by inspect.getfile() in NodeDescProvider
        sys.modules[nodeType] = module
        spec.loader.exec_module(module)
        nodeDescs[nodeType] = getattr(module, nodeType)

    registered = []
    for nodeType, nodeDesc in nodeDescs.items():
        if not pluginManager.isNodeDescRegistered(nodeType):
            pluginManager._nodeDescProviders[nodeType] = NodeDescProvider(nodeDesc)
            registered.append(nodeType)
    yield nodeDescs
    for nodeType in registered:
        del pluginManager._nodeDescProviders[nodeType]


def test_photometricStereoDescriptor(nodeDescs):
    nodeDesc = nodeDescs["PhotometricStereoHalide"]
    assert issubclass(nodeDesc, desc.Node)
    inputs = {a.name for a in nodeDesc.inputs}
    outputs = {a.name for a in nodeDesc.outputs}
    assert {"inputPath", "pathToJSONLightFile", "maskPath", "removeAmbient", "isRobust", "downscale"} <= inputs
    assert {"outputPath", "outputSfmDataAlbedo", "outputSfmDataNormal", "outputSfmDataNormalPNG",
            "normals", "normalsPNG", "normalsWorld", "albedo"} <= outputs


def test_sphereDetectionDescriptor(nodeDescs):
    nodeDesc = nodeDescs["SphereDetectionPy"]
    assert issubclass(nodeDesc, desc.Node)
    inputs = {a.name for a in nodeDesc.inputs}
    assert {"input", "modelPath", "autoDetect", "sphereShape", "sphereFile", "fillMissingSpheres", "minScore"} <= inputs
    assert {a.name for a in nodeDesc.outputs} == {"output"}


def test_sphereDetectionResources(nodeDescs):
    """ A GPU is only requested for the automatic detection on GPU. """
    node = Graph("test").addNewNode("SphereDetectionPy")
    assert node.gpu == desc.Level.NONE
    node.autoDetect.value = True
    assert node.gpu == desc.Level.NONE
    node.useGpu.value = True
    assert node.gpu == desc.Level.NORMAL


def test_sphereShapeKeyableValues(nodeDescs, dataset, tmp_path):
    """ Manual mode from the keyable "sphereShape" attribute, as set in the 2D viewer. """
    node = Graph("test").addNewNode("SphereDetectionPy", input=dataset.sfmPath)
    with open(dataset.spheresPath) as f:
        groundTruth = json.load(f)["shapes"][0]["observations"]
    viewIds = sorted(groundTruth, key=int)[:3]
    for viewId in viewIds:
        node.sphereShape.geometry.setObservation(viewId, groundTruth[viewId])

    outputPath = tmp_path / "detection.json"
    sphereDetection.writeManualSpheres(dataset.sfmPath, node.sphereShape.geometry.getValueAsDict(), str(outputPath),
                                       fillMissingSpheres=False)
    with open(outputPath) as f:
        result = json.load(f)["shapes"][0]["observations"]
    assert set(result) == set(viewIds)
    for viewId in viewIds:
        assert result[viewId]["center"] == pytest.approx(groundTruth[viewId]["center"])
        assert result[viewId]["radius"] == pytest.approx(groundTruth[viewId]["radius"])


def test_template(nodeDescs):
    nodeDesc = nodeDescs["PhotometricStereoHalide"]
    with open(TEMPLATE_FILE) as f:
        template = json.load(f)
    graph = template["graph"]
    node = graph["PhotometricStereoHalide_1"]
    assert node["nodeType"] == "PhotometricStereoHalide"
    assert node["version"] == "1.0"

    # Every attribute referenced in the template exists on the node
    attributes = {a.name for a in nodeDesc.inputs + nodeDesc.outputs}
    assert set(node["inputs"]) <= attributes
    references = re.findall(r"\{PhotometricStereoHalide_1\.(\w+)\}", json.dumps(graph))
    assert references and set(references) <= attributes

    # The template node description is the current one
    templateInputs = template["nodeDescriptions"]["PhotometricStereoHalide"]["1.0"]["inputs"]
    assert set(templateInputs) == {a.name for a in nodeDesc.inputs}

def test_lightingCalibrationDescriptor(nodeDescs):
    nodeDesc = nodeDescs["LightingCalibrationPy"]
    assert issubclass(nodeDesc, desc.Node)
    assert {a.name for a in nodeDesc.inputs} >= {"inputPath", "inputDetection", "saveAsModel", "ellipticEstimation", "method"}
    assert {a.name for a in nodeDesc.outputs} == {"outputFile", "lightingEstimationVisualization"}
    method = next(a for a in nodeDesc.inputs if a.name == "method")
    assert method.values == ["brightestPoint", "whiteSphere", "SH"]
