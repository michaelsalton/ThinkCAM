#!/bin/bash
# Launcher for EVS capture script on OpenSUSE Leap 16
# Sets LD_LIBRARY_PATH to the extracted ArenaSDK libraries and activates the venv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SDK=${ARENA_SDK:-$REPO_ROOT/ArenaSDK/ArenaSDK_Linux_x64}

if [ ! -d "$SDK" ]; then
    echo "ERROR: ArenaSDK not found at $SDK"
    echo "Download it from https://thinklucid.com/downloads-hub/"
    echo "Extract it and either place it at ~/ArenaSDK_Linux_x64"
    echo "or set ARENA_SDK=/path/to/ArenaSDK_Linux_x64"
    exit 1
fi

export LD_LIBRARY_PATH=\
$SDK/lib64:\
$SDK/GenICam/library/lib/Linux64_x64:\
$SDK/Metavision/lib:\
$SDK/ffmpeg

# GenTL producer for LUCID cameras (skips the need for sudo Arena_SDK.conf -cti)
export GENICAM_GENTL64_PATH=$SDK/lib64${GENICAM_GENTL64_PATH:+:$GENICAM_GENTL64_PATH}

# The offline probe (D / Run Probe) shells out to an analysis interpreter that
# carries numpy/h5py/cv2 plus the colmap binary, not the camera venv's
# arena_api + PySide6. thinkcam/constants.py defaults to ~/envs/phase1, which
# does not exist on every machine; the camera venv now carries those packages
# itself, so fall back to it rather than leaving the panel disabled. An
# explicit THINKCAM_PROBE_PYTHON in the environment always wins.
if [ -z "$THINKCAM_PROBE_PYTHON" ]; then
    for cand in ~/envs/phase1/bin/python ~/envs/default/bin/python; do
        if [ -x "$cand" ]; then export THINKCAM_PROBE_PYTHON="$cand"; break; fi
    done
fi

source ~/envs/default/bin/activate

# thinkcam and pipeline are packages under the repo root, not this directory.
cd "$REPO_ROOT"

exec python3 -m thinkcam.main "$@"
