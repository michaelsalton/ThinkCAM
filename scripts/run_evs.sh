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

source ~/envs/default/bin/activate

# thinkcam and pipeline are packages under the repo root, not this directory.
cd "$REPO_ROOT"

exec python3 -m thinkcam.main "$@"
