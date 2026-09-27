# Turret hardware handoff (one page)

Use this to add the pan/tilt turret and laser to the software-only Theia search (commit `b34f9f8`, or the `eluo26/theia` repo). That base already scans with DroidCam, sends angle-tagged frames to Grok, and returns an aim angle. It never moves anything: every frame is tagged `pan=0`. A working implementation exists in this checkout. Port it rather than redesigning it. The longer design rationale is in [turret-integration-spec.md](turret-integration-spec.md).

## Physical setup

- **Arduino UNO** over USB (COM4 on the bench laptop), 115200 baud. The servos run on a separate battery. A brown-out resets the board mid-command.
- **Pan:** large servo on **D9**. **Tilt:** micro servo on **D10**. **Laser:** **D8**, active-high, switched by a transistor. `firmware/laser_pin_finder` identified the pin.
- **Camera:** an iPhone running DroidCam at `http://172.20.10.1:4747/video`, mounted on the pan base. It **pans but does not tilt**, so photos are always tagged tilt 0. The laser rides the tilt servo.

## Firmware: `firmware/theia_turret/theia_turret.ino`

Text protocol with one line in and one reply line out. Servo degrees only.

| Command | Reply |
|---|---|
| (boot) | `READY` after homing and a worst-case sweep delay (~1.5 s) |
| `PING` | `OK` |
| `HOME` | `OK <pan> <tilt>` |
| `GOTO <pan> <tilt>` | `OK <pan> <tilt>`, or `ERR out_of_range` / `ERR bad_args` |
| `LASER 1` / `LASER 0` | `OK` |

Limits: pan 0–180, tilt 0–45. `GOTO` blocks for an estimated travel time (8 ms/° pan, 4 ms/° tilt, +100 ms) because hobby servos give no feedback. Safety rules: **the laser turns off on every move and after 20 s**, and it is driven low before `Serial.begin`.

## Python driver: `server/turret.py` (the only place that knows servo degrees)

- **Vision convention:** pan 0 is home and positive is right; tilt 0 is level and positive is up.
- **Bench constants:** `PAN_HOME_SERVO=0`, `PAN_SIGN=-1` (a larger servo angle turns left, so the sweep runs vision pan 0 to −180), `TILT_LEVEL_SERVO=0`, `TILT_SIGN=-1` (45 = down). **Re-measure these if the mount has shifted.** With the server stopped, open the IDE Serial Monitor (115200, Newline), send `HOME` and `GOTO` by hand, edit these constants, then close the monitor. They are the only place calibration lives.
- **`SCAN_PANS` = servo 0, 30, …, 180**, which gives 7 photos. `servo_pan` also tries ±360 because azimuths near the far end come back wrapped.
- **Reach is decided by pan only.** Tilt is clamped into 0–45, so an object slightly above level (for example el +3.9°) still gets panned to, with the laser held level. Refusing the whole aim over tilt was a real bug.
- **`SerialTurret`** opens the port lazily. Opening resets the UNO, so the driver waits for `READY`, then sends `HOME`. Any timeout, `ERR` or unexpected `READY` closes the port and raises `TurretError` with a readable message. `FakeTurret` is for tests.
- **Environment:** `TURRET=serial|none` (`none` keeps the old handheld pan-0 scan), `ARDUINO_PORT`, `ARDUINO_BAUD`, `SEARCH_SETTLE_S` (default 0.4), `LASER_ON_SECONDS` (default 10; 0 aims without lighting the laser). Add `pyserial` to `requirements.txt`.

## Wiring it into the software (`server/scan.py`, `server/object_search.py`)

1. `run_object_search` gets the turret only for live camera searches. **Replaying a saved scan never moves hardware.** It holds `turret_session` (a lock), so only one search drives the turret at a time.
2. At the start of a search, cancel any pending laser-off timer and turn the laser off. Then `_turret_scan`: check that the camera works, `HOME`, and for each `SCAN_PAN` do `goto`, wait `SEARCH_SETTLE_S`, and **take only a frame that arrived after the settle**, because DroidCam lags. Tag the frame with the pan the turret reported. Before each move, check `should_continue` and raise `ScanCancelled` if a newer search replaced this one.
3. Vision runs unchanged on the tagged frames.
4. Pan when the aim has finite angles and the status is `found`, or `ambiguous` (the best candidate: `best_guess=true` in the search state, and the UI says "My best guess"). Both light the laser. `not_found` never moves the turret. Then `can_reach`, stage `pointing`, `goto(az, el)`, re-check that the search is still current, `laser_on`, and `schedule_laser_off(LASER_ON_SECONDS)`. On any error, turn the laser off and fail with `failed_at="pointing"`.
5. `atexit` turns the laser off if the port is open.

## Operational traps we hit

- **Only one program can hold COM4.** Arduino IDE uploads fail with "Access is denied" while Flask holds the port (it keeps it open after the first search) or while the Serial Monitor is open. Stop the server before uploading, and close the monitor before searching.
- **Don't edit Python files while a search is running under `flask debug`.** The auto-reloader killed a process mid-serial-I/O, which left a process that could not be killed. It still held port 5000 and COM4 until the USB cable was replugged or the PC rebooted.
- **Latency is the vision API, not the hardware.** Use the fast model (`grok-4.3`, with no `VISION_MODEL` override). grok-4.6 took about 54 s against about 15 s. With `image_detail: high` and `api_max_edge: 1536`, small objects are found; at 768/low, grok-4.3 missed an eraser.
- **The laser dot correction loop (`server/laser.py`) and pixel-to-motor calibration are not integrated yet.**

## Tests to carry over

`server/test_turret.py` covers conversions, the serial protocol against a fake board, the pan-only reach check and laser timers. `server/test_search.py` covers the sweep tagging, cancellation, the laser only on `found`, and out-of-range pan. Run them with `python -m unittest discover -s server -p 'test_*.py'`.
