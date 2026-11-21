# Rough scripts — Kinova Gen3

Throwaway scripts for poking at the Kinova Gen3 arm and its depth camera.
Not a library, not tested beyond "it ran once" — these exist to confirm the
hardware and the APIs behave the way we expect before the real planner is built.

Default robot IP is `192.168.2.9`; credentials are the stock `admin` / `admin`.

## Which Python to use

Two different interpreters, because the dependencies don't live together:

| Script | Interpreter | Needs |
| --- | --- | --- |
| `kortex_hello.py`, `move_arm.py` | `~/kortex_env/bin/python` (venv) | `kortex-api` 2.8.0 |
| `depth_cam.py` | system `python3` | `numpy`, `opencv-python`, `PyGObject` + GStreamer |

The venv has `kortex-api` but no numpy/cv2/gi; the system Python has
numpy/cv2/gi but no `kortex-api`. Don't mix them up — the import error is the
first thing that goes wrong.

```bash
source ~/kortex_env/bin/activate     # for the two arm scripts
```

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
