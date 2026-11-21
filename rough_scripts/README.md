# Rough scripts — Kinova Gen3

Throwaway scripts for poking at the Kinova Gen3 arm and its depth camera.
Not a library, not tested beyond "it ran once" — these exist to confirm the
hardware and the APIs behave the way we expect before the real planner is built.

Default robot IP is `192.168.2.9`; credentials are the stock `admin` / `admin`.

## Dependencies

Python 3.10. The arm scripts and the depth script need different packages, and
they don't have to live in the same environment.

### For `kortex_hello.py` and `move_arm.py`

`kortex-api` is **not on PyPI** — `pip install kortex-api` will not work. The
wheel has to be downloaded from Kinova's Artifactory server first:

- **Gen3 API 2.8.0 (what these scripts use):**
  <https://artifactory.kinovaapps.com/ui/native/generic-local-public/kortex/API/2.8.0/>
- Python wheels, all versions:
  <https://artifactory.kinovaapps.com/ui/repos/tree/General/pypi-local-build/Kortex/API>
- Download links for every arm model and version are listed in the README of
  the official repo: <https://github.com/Kinovarobotics/kortex>

No login is needed. Open the folder, pick the `.whl` in the tree view on the
left, and hit Download on the right. The file these scripts were written
against is `kortex_api-2.8.0.post5-py3-none-any.whl`. Then install from the
downloaded path:

```bash
python3 -m pip install ~/Downloads/kortex_api-2.8.0.post5-py3-none-any.whl
```

It pulls in its own pinned dependencies: `protobuf>=3.5.1,<=3.20` and
`deprecated==1.2.7`. The protobuf ceiling is strict, so this is worth keeping
in its own virtual environment — a newer protobuf elsewhere in the environment
will break the generated message classes.

### For `depth_cam.py`

Python packages:

```bash
pip install numpy opencv-python
```

Plus PyGObject and the GStreamer runtime, which come from the system package
manager rather than pip (on Ubuntu/Debian):

```bash
sudo apt install python3-gi python3-gi-cairo \
                 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
                 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

`gstreamer1.0-plugins-good` is the one that provides `rtspsrc` and
`rtpgstdepay`; `-plugins-base` provides `videoconvert` and `appsink`. If the
pipeline fails to build, check the elements are present:

```bash
gst-inspect-1.0 rtspsrc rtpgstdepay videoconvert appsink
```

Note that `python3-gi` is only importable from the Python it was installed
for — a virtual environment needs `--system-site-packages` (or `vext`) to see
it.

## Scripts

### `kortex_hello.py` — connection smoke test

Opens a session, prints the measured joint angles and the tool pose, closes.
Run this first; if it fails, nothing else will work.

```bash
python3 kortex_hello.py [IP]
```

### `move_arm.py` — relative joint move

Moves the arm by a *relative* offset on each joint, in degrees. Reads the
current angles, adds the deltas, and sends one `reach_joint_angles` action.

```bash
python3 move_arm.py 0 0 0 1 0 2              # +1 deg on joint 3, +2 deg on joint 5
python3 move_arm.py 0 0 0 1 0 2 --dry-run    # print current -> target, don't move
python3 move_arm.py 5 0 0 0 0 0 --speed 10   # slower
```

- One angle per joint, so six for a 6-DOF arm. The script checks
  `GetActuatorCount()` and refuses a mismatch.
- Joint indices are **0-based** (`joint_identifier` 0–5).
- Blocks until the robot reports `ACTION_END`, so the session isn't torn down
  mid-motion. Times out after 30 s and calls `StopAction()`.
- On a 6-DOF Gen3 the even-numbered actuators are range-limited rather than
  continuous — a delta that wraps past a limit comes back as an aborted action,
  not a clamped one.
- `--dry-run` still connects, because "relative" needs the current angles.
  Use it to sanity-check sign convention and joint ordering before moving.

### `depth_cam.py` — depth stream viewer

Pulls the arm's depth RTSP stream (`rtsp://<ip>/depth`) through GStreamer,
prints the median depth of a 10x10 patch at the image centre, and shows a
JET-colormapped view. `q` quits.

```bash
python3 depth_cam.py
```

Still rough: the IP is hardcoded, and `FIXED_MAX = 5000` is a guessed
normalisation range. The script prints the negotiated caps on the first frame
so you can see the real format and units — check that before trusting the
numbers. Zero means "no depth", not "zero distance", so it's masked out of the
median.

## Known rough edges

- IPs and credentials are hardcoded in `kortex_hello.py` and `depth_cam.py`;
  only `move_arm.py` takes `--ip`.
- No error recovery anywhere. If the arm is in a fault state, clear it from the
  web app at `http://192.168.2.9` first.
- Nothing here checks for collisions or joint limits ahead of time — the robot
  is the only thing saying no. Keep the e-stop in reach.

