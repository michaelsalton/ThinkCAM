import os

import cv2

CAMERA_IP = "169.254.80.199"
NUM_BUFFERS = 10
IMAGE_TIMEOUT_MS = 2000
# Rate at which decoded frames are rendered and pushed to the GUI. The camera
# delivers event buffers far faster than this (~1.7k/s); emitting one Qt signal
# per buffer floods the GUI thread's queue and makes the display lag further and
# further behind real time. We consume every buffer (raw recording stays
# lossless) but accumulate events and only render/emit at this rate.
DISPLAY_FPS = 30.0
ERC_RATE_LIMIT_MEV = 10.0

BIAS_THRESHOLD_POS = 10
BIAS_THRESHOLD_NEG = 10
BIAS_REFRACTORY = 10
BURST_FILTER_ENABLE = True

DENOISE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# EVS stream output format. "XYTPFrame" delivers the raw asynchronous event
# stream as 4x float32 (x, y, t, p) per fired pixel — the lossless input the
# raw recorder taps. ("CDFrame" was the old preview-only rasterized mode.)
EVS_OUTPUT_FORMAT = "XYTPFrame"

# Raw event recorder
RAW_RECORD_DIR = "recordings"
# Bounded handoff queue between the acquisition QThread and the writer thread,
# measured in event *batches* (one per camera buffer). If it ever saturates the
# dropped-events counter rises (it must stay 0 at normal rates). The writer
# coalesces all queued batches into one HDF5 append, so this mainly needs to be
# deep enough to absorb a motion burst while the writer catches up; at ~1.7k
# buffers/s, 4096 is ~2.4s of headroom.
RAW_QUEUE_MAXSIZE = 4096
# HDF5 chunk length along the event axis.
RAW_HDF5_CHUNK = 1_000_000
# HDF5 compression for the event datasets. LZF caps the writer at ~8 Mev/s on
# this machine — below the camera's 10-MEV ErcRateLimit, so a fast scene
# overflows the queue and drops events. None lifts the write ceiling to
# ~50 Mev/s (well above any rate the sensor can emit) at the cost of larger
# files (~13 bytes/event vs ~8.4 compressed). Lossless capture wins; set to
# "lzf" or "gzip" only for long takes where disk space matters more than rate.
RAW_HDF5_COMPRESSION = None

# --------------------------------------------------------------------------
# Capture probe (plans/01_CaptureDashboard.md)
# --------------------------------------------------------------------------
# The pipeline window is PINNED to the best configuration on record --
# work/phase1b/tracked_mc25k, 25 k events per frame with motion compensation on,
# which produced 12 fragments at mean track length 3.07-4.37. This is a
# correction to Phase1b-Results.md, which used 200 k and judged compensation not
# worth 1.7x the cost: that held for the SIFT path, not the LK one. Changing
# these numbers makes a probe incomparable with every earlier probe, so change
# them only together with a new reference run.
PROBE_EVENTS_PER_FRAME = 25_000
PROBE_STRIDE = 25_000            # no overlap, as that run used
PROBE_MOTION_COMP = True
# track_frames' own default is 10. tracked_mc25k was run at 12, and the
# difference is not cosmetic: with bit-identical LK tracks (81,823 keypoints
# either way), 10 gives 7 fragments with the largest covering 24 frames / 0.63 s,
# and 12 gives 12 fragments with the largest covering 37 frames / 1.04 s -- union
# coverage 55% vs 84%. So it is pinned here too, or a probe is not comparable
# with the one result on record. It is also the first hard evidence for
# plans/01_CaptureDashboard.md §9's suspicion that --max-gap tuning moves
# fragmentation before the sensor is the limit.
PROBE_MAX_GAP = 12

# Events per lit pixel needs TWO windows. Measured on the take every wiki number
# came from: 1.07 ev/lit-px at the 25 k pipeline window, 1.21 at 200 k, 2.27 at
# 2.0 s. A 76x longer window buys 2x the density, because a moving camera paints
# a wider swath rather than building up the same pixels. A >= 5 gate at the
# pipeline window is therefore unreachable by construction; it hangs on the
# reference window instead.
PROBE_REFERENCE_S = 2.0
PROBE_EV_PER_LIT_TARGET = 5.0    # Phase1b-Results.md §9.1, vs the REFERENCE window

PROBE_UPDATE_HZ = 2.0
PROBE_MOTION_SEARCH_EVENTS = 30_000
PROBE_WORK_DIR = "work/probe"

# The GUI venv carries arena_api + PySide6; the analysis tools need numpy/h5py/
# cv2 and the `colmap` binary, which live in a separate environment. The GUI
# shells the full probe out to this interpreter rather than importing it.
PROBE_PYTHON = os.environ.get(
    "THINKCAM_PROBE_PYTHON", os.path.expanduser("~/envs/phase1/bin/python")
)
