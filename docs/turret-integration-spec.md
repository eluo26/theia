# Turret integration spec

Connect the Arduino turret to the live object search. The server sweeps the camera through real pan angles, and then, only after a confident `found` result, aims the laser at the object and turns it on.

This spec is written for whoever implements it on the computer that has the Arduino plugged in, including a coding assistant with no earlier context. Each step says **why** it exists, so you can make sensible calls when the hardware doesn't match the assumptions.

---

## 1. Where things stand

**The search pipeline already works end to end, apart from the hardware.**

- [server/app.py](../server/app.py): `POST /api/search` starts a background thread running `run_object_search`, and the UI polls `GET /api/search/<id>`.
- [server/scan.py](../server/scan.py): `capture_scan` takes 7 JPEGs from the DroidCam stream 0.5 s apart. Every frame is tagged `pan=0, tilt=0` because nothing moves the camera yet. Scans are saved to `server/data/scans/<stamp>/` (`frame_NNN.jpg` + `manifest.json`) and can be replayed with `scan_dir`.
- [vision/aim.py](../vision/aim.py) `search_frames`: sends the frames to Grok and returns `fire_laser`, `aim {azimuth_deg, elevation_deg}`, the result box and the chosen frame. `fire_laser` is true only when `status == "found"`.
- [vision/geometry.py](../vision/geometry.py): converts a box center to turret angles using each frame's `pan`/`tilt` and the configured field of view.
- [server/object_search.py](../server/object_search.py) `run_object_search`: sets the UI stage to `on_target` or `failed`, and then stops. **Nothing talks to the motors or the laser yet.**

**Why this matters:** the vision side already expects angle-tagged frames and already produces the aim angles. The only missing piece is hardware. Because every frame currently claims `pan=0`, the azimuth the vision code returns today is wrong for any photo not taken straight ahead. Real pan tags fix that.

### Angle convention (fixed; the vision code depends on it)

- Azimuth / pan: 0 = pan home, **positive = right**, degrees.
- Elevation / tilt: 0 = horizontal, **positive = up**, degrees.

Everything above the turret driver uses this convention. Servo degrees exist **only inside `turret.py` and the firmware**.

**Why:** if servo-to-vision conversions are spread around the code, a sign error ends up in three places and aims the laser the wrong way. Keeping them in one place means one constant to fix at the bench.

---

## 2. What the current Arduino sketch tells us

Person B shared a demo sketch. Relevant facts:

| Item | Sketch | Meaning |
|---|---|---|
| Pan servo | Large servo, pin 9, home `write(0)`, sweeps 0→180 in 30° steps | Pan home = servo 0. The 7 positions match the plan. **Whether increasing servo angle turns left or right is unknown.** |
| Tilt servo | Micro servo, pin 10, `0` = up, `45` = down | Only aims the laser. A larger servo angle points **lower**, the opposite of the vision sign. **Which servo angle is horizontal is unknown.** Usable range looks like about 45°. |
| Laser power | Not in the sketch | CLAUDE.md says a transistor switches it. **We need the pin number.** |
| Serial | None | Python has no way to talk to it. |
| Behaviour | Loops forever from power-on | Must be replaced. Otherwise the turret moves by itself while the server tries to control it. |

**Consequence:** we can't wrap an existing protocol, because there isn't one. We write a small command-driven sketch (section 3) based on Person B's pins and angles, and Person B reviews it.

### Questions for Person B (answer these before or during the bench session)

1. Which pin switches the laser transistor, and is it active-high?
2. Is the phone mounted on the rotator (so it pans with it)? Does the phone tilt at all?
3. Which Arduino board is it (Uno, Nano, etc.)?
4. Is it OK to replace the demo loop with the command-driven sketch below?

**Why question 2 is critical:** `geometry.py` assumes the camera turns with the pan axis. If the phone rides on the rotator but doesn't tilt, frames are tagged with the real pan and `tilt = 0`. If the phone is on a fixed stand, pan-tagging frames is wrong and the design has to change. Stop and rethink before continuing.

---

## 3. Firmware: `firmware/theia_turret/theia_turret.ino`

(The Arduino IDE requires the folder name to match the `.ino` name.)

### Protocol

Plain-text ASCII lines, terminated by `\n`, at **115200 baud**. The Arduino sends one reply line per command.

| Python sends | Arduino replies | Notes |
|---|---|---|
| (nothing; board just booted) | `READY` | Sent once at the end of `setup()` |
| `PING` | `OK` | Liveness check |
| `HOME` | `OK <pan> <tilt>` | Pan to home, tilt to level; replies once settled |
| `GOTO <pan> <tilt>` | `OK <pan> <tilt>` or `ERR <reason>` | Integer **servo** degrees. Replies **after** the move time has passed |
| `LASER 1` / `LASER 0` | `OK` | Laser power on / off |
| anything else | `ERR unknown` | |

**Why text lines:** you can test it by typing into the Serial Monitor at the bench with no Python involved, and the Python parser is trivial.

**Why reply only after the move:** `goto` then waits until the servo has actually arrived, so Python knows exactly when to take a photo instead of guessing with a sleep. Hobby servos give no position feedback, so the firmware estimates the move time from the distance travelled.

### Required firmware behaviour

1. **Boot:** laser pin `OUTPUT` and `LOW` **first**, then attach the servos, go to home, wait, and print `READY`.
   - **Why:** the laser must never be on during a reset. The board resets every time Python opens the serial port.
2. **Soft limits:** reject a `GOTO` outside pan `0–180` or tilt `0–45` (adjust after the bench session) with `ERR out_of_range`. Don't clamp.
   - **Why:** clamping silently aims somewhere other than the target. An error lets Python refuse to fire.
3. **Move time:** `wait_ms = max(|Δpan| * PAN_MS_PER_DEG, |Δtilt| * TILT_MS_PER_DEG) + MARGIN_MS`. Start with `PAN_MS_PER_DEG = 8`, `TILT_MS_PER_DEG = 4`, `MARGIN_MS = 100`, and tune at the bench.
   - **Why:** the demo's fixed 166 ms is tight for 30° and far too short for returning from 180 to 0.
4. **Laser hard timeout:** switch the laser off automatically `LASER_MAX_ON_MS = 20000` after the last `LASER 1`.
   - **Why:** if the Python server crashes, or Flask's debug reloader restarts it while the laser is on, nothing in Python runs to turn it off. The firmware is the only layer that can guarantee it. This is a device aimed at a room with people in it.
5. **No automatic movement.** `loop()` only reads and handles commands and runs the laser timeout.
6. Use a non-blocking line reader (collect characters into a buffer until `\n`) and ignore `\r`.
   - **Why:** Windows serial monitors often send `\r\n`.

---

## 4. Python driver: `server/turret.py`

Add `pyserial` to [requirements.txt](../requirements.txt).

### Public interface (the rest of the server uses only this)

```python
class TurretError(RuntimeError): ...

class Turret:                         # implemented by SerialTurret and FakeTurret
    def home(self) -> None: ...
    def goto(self, pan_deg: float, tilt_deg: float) -> tuple[float, float]: ...
        # vision degrees in; blocks until settled; returns the pose actually commanded
    def laser_on(self) -> None: ...
    def laser_off(self) -> None: ...
    def current_pose(self) -> tuple[float, float]: ...   # last commanded, vision degrees
    def can_reach(self, pan_deg: float, tilt_deg: float) -> bool: ...
    def close(self) -> None: ...

def get_turret() -> Turret | None:    # process-wide singleton, like scan.get_camera()
```

**Why a small interface:** the pipeline person can build and test everything against `FakeTurret` while the hardware person works on the serial details. The two meet only at this interface.

### Servo ↔ vision conversion (the only place it happens)

```python
PAN_HOME_SERVO   = 0     # from the sketch
PAN_SIGN         = +1    # TODO bench: +1 if increasing servo angle turns right, else -1
TILT_LEVEL_SERVO = 0     # TODO bench: servo angle at which the laser is horizontal
TILT_SIGN        = -1    # sketch: larger servo angle = lower

servo_pan  = PAN_HOME_SERVO + PAN_SIGN * pan_deg
servo_tilt = TILT_LEVEL_SERVO + TILT_SIGN * tilt_deg
```

Round to integers before sending. `can_reach` converts and checks against the firmware limits.

**Why constants with TODOs:** these values can only be measured with the hardware. Writing them as named constants turns the bench session into editing four numbers, not rewriting code.

### `SerialTurret` requirements

- **Configuration from the environment:**
  - `TURRET` = `serial` (default) or `none`
  - `ARDUINO_PORT` (for example `COM5`)
  - `ARDUINO_BAUD` (default `115200`)
  - Add all three to [.env.example](../.env.example), commented.
- **Open the port lazily**, on first use and not at import.
  - **Why:** `app.run(debug=True)` starts two processes (the reloader and the worker), and Windows lets only one process open a COM port. Opening at import makes the second process fail.
- **After opening, wait up to 5 s for `READY`**, discard any other startup noise, then send `HOME`.
  - **Why:** opening the port resets the board, and commands sent during the bootloader are lost silently.
- **One `threading.Lock` around every command/reply exchange.** Read timeout: 3 s for `GOTO`/`HOME` (a 180° move with margin takes about 1.6 s) and 1 s for other commands.
  - On `ERR …`, timeout or `SerialException`, raise `TurretError` with a readable message. For example, "Arduino not found on COM5. Is it plugged in, and is the Serial Monitor closed?"
- **Clear errors for common failures:** a missing or empty `ARDUINO_PORT` with `TURRET=serial` raises `TurretError("ARDUINO_PORT is not set…")`. Don't quietly fall back to fake mode.
  - **Why:** a quiet fallback would tag every frame `pan=0` and make aims silently wrong, which is exactly the bug this work removes.
- **Register `atexit` → `laser_off()`**, best effort, swallowing errors.
  - **Why:** a normal shutdown should also leave the laser off. This backs up the firmware timeout.

### `FakeTurret`

- Records every call in `self.calls` (for example `[("home",), ("goto", 30.0, 0.0), ("laser_on",)]`).
- Tracks the pose and laser state, uses the same `can_reach` limits, and can be told to raise on the next call.
- Used in all tests.

`TURRET=none` makes `get_turret()` return `None`. See section 5 for what that means.

---

## 5. Scan: change `capture_scan` in `server/scan.py`

**New behaviour when a turret is available:**

```
laser_off()
home()
for pan in SCAN_PANS:                  # (0, 30, 60, 90, 120, 150, 180) in vision degrees
    if should_continue is not None and not should_continue(): raise ScanCancelled
    pose = turret.goto(pan, 0)
    time.sleep(SETTLE_S)                # default 0.4, env SEARCH_SETTLE_S
    frame = camera.read(newer_than=time.monotonic())
    tag the frame with pan = pose[0], tilt = CAMERA_TILT (0)
```

**Notes and reasons:**

- **Signature:** `capture_scan(camera=None, turret=None, on_frame=None, should_continue=None, …)`. Tests pass a `FakeTurret` and a `FakeCamera`.
- **`SETTLE_S` plus `newer_than`:** DroidCam lags a few hundred milliseconds. A frame that arrives right after the move finishes may have been captured while the camera was still moving, which gives a blurry photo at the wrong angle. Tune `SETTLE_S` at the bench.
- **Frame tilt is the camera's tilt, not the laser servo's.** If the phone doesn't tilt, the frame's tilt is always 0.
- **`SCAN_PANS` assumes pan home is one end of the travel and the sweep goes right.** If the bench shows the servo turns left as its angle increases, `PAN_SIGN = -1` and the sweep becomes `0, -30, … -180` in vision degrees. `SCAN_PANS` must then be derived from the servo range, not hard-coded. Compute it as `vision_pan(servo)` for servo angles 0, 30, …, 180.
- **`scan_photo_count()`** returns `len(SCAN_PANS)` when a turret is in use. `app.py` shows `total_photos` from this, and the UI already expects 7.
- **`should_continue`:** see section 7. It lets a replaced search stop moving the turret.
- **`TURRET=none`:** keep today's timed handheld capture with `pan=0` and log a warning once.
  - **Why:** the UI and vision work can continue on laptops without the Arduino. It's opt-in, so it can't happen by accident.
- **Nothing changes in `save_scan`, `load_saved_scan` or replay.** Saved `manifest.json` files will now contain real pans, which is what vision consumes. Replay (`scan_dir`) never moves the turret.

---

## 6. Firing: change `run_object_search` in `server/object_search.py`

After `search_frames` returns:

```
found = fire_laser is True and status == "found" and aim has numeric azimuth_deg and elevation_deg
if not found:                   -> existing "failed" path. No goto, no laser.
elif turret is None:            -> existing on_target path (TURRET=none or replay without hardware)
elif not turret.can_reach(az, el):
                                -> failed, failed_at="aiming",
                                   error="Found it, but it's outside the turret's range."
else:
    turret.goto(az, el); turret.laser_on(); start the auto-off timer; existing on_target update
```

**Rules and reasons:**

- **Never move or fire on `ambiguous` or `not_found`.** Pointing at a guess tells the patient something false with confidence. CLAUDE.md requires this.
- **Fail when the aim is out of reach; don't clamp.** The tilt servo covers only about 45°, and objects at the edge of the frame can fall outside the sweep. Clamping would light up the wrong spot.
  - Check with the UI owner that `failed_at: "aiming"` renders sensibly. If not, use `"searching"` with that error text.
- **Auto-off timer:** `threading.Timer(LASER_ON_SECONDS, laser_off)`, default 15 s (env `LASER_ON_SECONDS`). Cancel any previous timer first. This is shorter than the firmware's 20 s so that Python normally turns the laser off before the firmware has to.
- **Any exception after `laser_on`** means best-effort `laser_off()`, then fail the search.
- **`TurretError` during the scan** means `stage="failed", failed_at="scanning"`, with the error message shown to the user. Treat it like the existing `CameraError` handling.
- **Replay with a real turret connected:** don't aim for now. Replay is for debugging the vision code, and a replayed scan may not match where the turret is now pointing. It's easy to allow later.

---

## 7. Concurrency and safety in `server/app.py`

**The problem:** `start_search` replaces `_search`, but the old worker thread keeps running. `_update` returning `False` only takes effect where the code checks it. Without changes, two threads can send turret commands at the same time.

**Required:**

1. **Stop a replaced search.** Pass `should_continue=lambda: update(search_id)` into `capture_scan`. With no changes, `update` returns `True` only while this search is still the current one. Check it before every move. Also check it before aiming, so a replaced search never fires.
2. **Laser off first.** At the start of every `run_object_search`, call `turret.laser_off()`.
   - **Why:** a new search means the patient has moved on, and the old dot is now misleading.
3. **One search uses the turret at a time.** Add a module-level `turret_session = threading.Lock()` in `turret.py`. `run_object_search` holds it from the laser-off through aiming. The per-command lock in `SerialTurret` is not enough: it stops two commands from being mixed up, but it doesn't stop two sweeps from interleaving.

---

## 8. Tests (`server/test_search.py` and a new `server/test_turret.py`)

All tests run without hardware.

| Test | Asserts |
|---|---|
| Scan with `FakeTurret` + `FakeCamera` | 7 frames tagged pan `0,30,…,180`, tilt `0`; `laser_off` and `home` happen before the first `goto` |
| Scan stops when replaced | `should_continue` flips to False after 2 frames → no further `goto`, search not marked `on_target` |
| Found + reachable | `goto(az, el)` then `laser_on`, stage `on_target` |
| `not_found` and `ambiguous` | no `goto` after the scan, no `laser_on` |
| Found but out of reach | no `laser_on`, stage `failed` |
| `TurretError` during scan | stage `failed`, `failed_at="scanning"`, readable message |
| Auto-off | with `LASER_ON_SECONDS` patched to about 0.05, `laser_off` is called |
| `SerialTurret` against a fake serial object | waits for `READY`, parses `OK <pan> <tilt>`, `ERR` → `TurretError`, silence → timeout `TurretError`, conversion signs correct |
| Missing `ARDUINO_PORT` | clear `TurretError`, no fallback to pan 0 |

- **Replace** `test_capture_scan_returns_jpegs_with_zero_angles`. It checks the behaviour this work removes.
- **Keep** the replay tests. Replay must still never touch the camera or the turret.

Run `python -m unittest discover -s server -p 'test_*.py'` from the repo root.

---

## 9. Bench procedure (with the hardware)

Do these in order. Each step catches a class of mistake before it can aim a laser.

1. **Power:** Arduino over USB; servos on their own battery with a **common ground** to the Arduino; laser behind its transistor.
   - **Why:** servos drawing current through USB cause brownouts and random resets. Without a common ground the servo signal has no reference, and the servos jitter or don't move.
2. **Flash the firmware and use the Serial Monitor** (115200, newline). Type `PING`, `HOME`, `GOTO 90 0`, `GOTO 90 45`, `LASER 1`, wait 20 s and check the laser turns itself off, then `GOTO 999 0` and expect `ERR`.
   - **Why:** this proves the firmware works on its own before Python is involved.
3. **Measure the constants:**
   - Send `GOTO 30 0`. Did the turret turn right or left from home? That sets `PAN_SIGN`.
   - Step the tilt servo until the laser is level. That sets `TILT_LEVEL_SERVO`.
   - Time a 180° move by eye to check `PAN_MS_PER_DEG`.
   - **Close the Serial Monitor** afterwards. It holds the COM port.
4. **Find the COM port** (Device Manager → Ports) and put `ARDUINO_PORT` and `TURRET=serial` in `.env`.
5. **Scan with the laser disabled** (temporarily set `LASER_ON_SECONDS=0`, or have `laser_on` do nothing behind a flag). Run one live search, open the newest `server/data/scans/*/manifest.json`, and check that the pans are `0, 30, …, 180` and that the photos actually face those directions.
   - **Why:** this checks the angles and the photo sharpness with no laser risk.
6. **Aim with the laser disabled:** search for an object you can see. Check that the reported `azimuth_deg` / `elevation_deg` are roughly where the object is, and that the turret physically points at it.
7. **Enable the laser** and test one object. Then test a query that should give `not_found` and confirm the laser stays off.

---

## 10. Acceptance criteria

- [ ] Live search moves the turret through 7 positions and saves `manifest.json` with real pans.
- [ ] `found` → the turret points at the object and the laser turns on; it goes off by itself within 15 s.
- [ ] `ambiguous`, `not_found`, out of reach, or any error → the laser never turns on.
- [ ] Starting a new search turns the laser off and stops an unfinished sweep.
- [ ] Unplugged Arduino with `TURRET=serial` → the UI shows a clear error, not a scan tagged pan 0.
- [ ] `TURRET=none` still runs the handheld flow.
- [ ] All server tests pass; `npm run build` and `npm run lint` still pass (no UI change expected).
- [ ] CLAUDE.md updated: the search is real, and the turret, env variables and firmware location are documented.

## 11. Out of scope

- The laser dot correction loop ([server/laser.py](../server/laser.py) `evaluate_pair`). This needs pixel-to-motor calibration first.
- Calibrating the camera-to-laser offset. The laser and camera aren't at the same point, so expect a small, distance-dependent miss.
- Camera intrinsics. `config.yaml` assumes `hfov_deg: 70` for 4032×3024 photos, and the DroidCam video stream may differ, so expect a few degrees of aim error. Note it; don't fix it here.
- UI changes, and anything from the warehouse/staff flows.
