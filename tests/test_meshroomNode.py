"""
Tests of the Meshroom node descriptor and pipeline template (skipped when Meshroom is not available).
"""

import importlib.util
import json
import os
import re

import pytest

pytest.importorskip("meshroom.core")

from meshroom.core import desc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_FILE = os.path.join(REPO, "meshroom", "photometricStereo", "PhotometricStereoHalide.py")
TEMPLATE_FILE = os.path.join(REPO, "meshroom", "photometricStereoHalide.mgt")


@pytest.fixture(scope="module")
def nodeDesc():
    spec = importlib.util.spec_from_file_location("PhotometricStereoHalide", NODE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhotometricStereoHalide


def test_descriptor(nodeDesc):
    assert issubclass(nodeDesc, desc.Node)
    inputs = {a.name for a in nodeDesc.inputs}
    outputs = {a.name for a in nodeDesc.outputs}
    assert {"inputPath", "pathToJSONLightFile", "maskPath", "removeAmbient", "isRobust", "downscale"} <= inputs
    assert {"outputPath", "outputSfmDataAlbedo", "outputSfmDataNormal", "outputSfmDataNormalPNG",
            "normals", "normalsPNG", "normalsWorld", "albedo"} <= outputs


def test_template(nodeDesc):
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
