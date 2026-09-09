#!/bin/bash
# Run any Arena SDK Python example (or arbitrary python script) with the
# ArenaSDK env wired up. Usage:
#   ./scripts/run_example.sh ArenaSDK/ArenaSDK_Linux_x64/Examples/Python/py_evs_acquisition.py
#   ./scripts/run_example.sh Arena_examples/Python/py_evs_xytp_frame_heatmap.py
# Script paths are resolved against your cwd (run it from the repo root), or
# against the SDK's own Examples/Python if you pass a bare filename.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SDK=${ARENA_SDK:-$REPO_ROOT/ArenaSDK/ArenaSDK_Linux_x64}

if [ ! -d "$SDK" ]; then
    echo "ERROR: ArenaSDK not found at $SDK"
    echo "Set ARENA_SDK=/path/to/ArenaSDK_Linux_x64 or place it at $REPO_ROOT/ArenaSDK/ArenaSDK_Linux_x64"
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [args...]"
    echo ""
    echo "Available EVS examples:"
    ls "$SDK/Examples/Python/" 2>/dev/null | grep -i evs | sed 's/^/  /'
    exit 1
fi

export LD_LIBRARY_PATH=\
$SDK/lib64:\
$SDK/GenICam/library/lib/Linux64_x64:\
$SDK/Metavision/lib:\
$SDK/ffmpeg

export GENICAM_GENTL64_PATH=$SDK/lib64${GENICAM_GENTL64_PATH:+:$GENICAM_GENTL64_PATH}

source ~/envs/default/bin/activate

# Resolve a bare filename (e.g. "py_evs_save_ply.py") against the SDK's
# Examples/Python directory so the user doesn't have to type the full path.
SCRIPT="$1"
shift
if [ ! -f "$SCRIPT" ] && [ -f "$SDK/Examples/Python/$SCRIPT" ]; then
    SCRIPT="$SDK/Examples/Python/$SCRIPT"
fi

exec python3 "$SCRIPT" "$@"
