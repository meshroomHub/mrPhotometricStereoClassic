#!/bin/bash
# Run the test suite in the AliceVision Docker image (alicevision/alicevision), which provides the AliceVision
# command line tools, the pyalicevision and OpenImageIO Python bindings (built for Python 3.11 and NumPy 1.26) and a
# Python 3.11 runtime. Halide, ONNX Runtime and pytest are installed in a virtual environment, and Meshroom is cloned
# for the node tests (unless MESHROOM_ROOT is given).
#
# Usage, from the repository root:
#   docker run --rm -v "$PWD:/workspace" -w /workspace alicevision/alicevision:<tag> bash ci/runTests.sh
#
# Environment:
#   MESHROOM_ROOT                      Meshroom repository used by the node tests (default: clone of MESHROOM_REF).
#   MESHROOM_REF                       Meshroom branch or tag to clone (default: develop).
#   ALICEVISION_SPHERE_DETECTION_MODEL Sphere detection model (ONNX): the detection tests are skipped without it.
set -euo pipefail

AV_BUNDLE=${AV_BUNDLE:-/opt/AliceVision_bundle}
MESHROOM_REF=${MESHROOM_REF:-develop}
WORK=$(mktemp -d)

# Same NumPy as the one used to build the AliceVision bindings
/opt/python/bin/python3.11 -m venv "${WORK}/venv"
"${WORK}/venv/bin/python" -m pip install --quiet --upgrade pip
"${WORK}/venv/bin/python" -m pip install --quiet "numpy==1.26.4" "halide==21.0.0" "onnxruntime==1.30.0" pytest psutil pyseq

if [ -z "${MESHROOM_ROOT:-}" ]; then
    MESHROOM_ROOT="${WORK}/meshroom"
    git clone --quiet --depth 1 --branch "${MESHROOM_REF}" https://github.com/alicevision/Meshroom.git "${MESHROOM_ROOT}"
fi

export PATH="${AV_BUNDLE}/bin:${PATH}"
export PYTHONPATH="${AV_BUNDLE}/lib/python:${AV_BUNDLE}/lib/python3.11/site-packages:${MESHROOM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ALICEVISION_ROOT="${AV_BUNDLE}"
export ALICEVISION_SENSOR_DB="${AV_BUNDLE}/share/aliceVision/cameraSensors.db"

"${WORK}/venv/bin/python" -c "import halide, onnxruntime, OpenImageIO, pyalicevision, meshroom.core"
"${WORK}/venv/bin/python" -m pytest tests -p no:cacheprovider -rs "$@"
