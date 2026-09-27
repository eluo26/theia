"""Pan/tilt turret driver for firmware/theia_turret.

Everything outside this module uses vision angles: pan 0 at home, positive
right; tilt 0 horizontal, positive up. Servo degrees exist only here and in
the firmware.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
import time

try:
    import serial
except ImportError:  # Tests and TURRET=none run without pyserial.
    serial = None

logger = logging.getLogger("turret")

# Measured at the bench on 2026-09-27.
PAN_HOME_SERVO = 0
PAN_SIGN = -1          # increasing servo angle turns left
TILT_LEVEL_SERVO = 0   # laser is level at servo 0
TILT_SIGN = -1         # larger servo angle points lower

# Must match the firmware limits.
PAN_SERVO_MIN, PAN_SERVO_MAX = 0, 180
TILT_SERVO_MIN, TILT_SERVO_MAX = 0, 45

SCAN_STEP_SERVO = 30
READY_TIMEOUT_S = 5.0
MOVE_TIMEOUT_S = 3.0
COMMAND_TIMEOUT_S = 1.0
DEFAULT_LASER_ON_SECONDS = 10.0

# One search drives the turret at a time, from laser-off through aiming.
turret_session = threading.Lock()


class TurretError(RuntimeError):
    """The turret could not be reached or refused a command."""


def servo_pan(pan_deg):
    """Vision pan to an integer servo angle, or None if it is out of reach.

    Azimuths near the far end of the sweep can come back wrapped by 360
    (for example +175 instead of -185), so both equivalents are tried.
    """
    for candidate in (pan_deg, pan_deg - 360.0, pan_deg + 360.0):
        servo = int(round(PAN_HOME_SERVO + PAN_SIGN * candidate))
        if PAN_SERVO_MIN <= servo <= PAN_SERVO_MAX:
            return servo
    return None


def servo_tilt(tilt_deg):
    servo = int(round(TILT_LEVEL_SERVO + TILT_SIGN * tilt_deg))
    if TILT_SERVO_MIN <= servo <= TILT_SERVO_MAX:
        return servo
    return None


def clamped_servo_tilt(tilt_deg):
    """Nearest reachable tilt servo angle. Pan is what matters for now, so an
    elevation above level or below the tilt range still aims, at the limit."""
    servo = int(round(TILT_LEVEL_SERVO + TILT_SIGN * tilt_deg))
    return max(TILT_SERVO_MIN, min(TILT_SERVO_MAX, servo))


def vision_pan(servo):
    return float((servo - PAN_HOME_SERVO) * PAN_SIGN)


def vision_tilt(servo):
    return float((servo - TILT_LEVEL_SERVO) * TILT_SIGN)


SCAN_PANS = tuple(
    vision_pan(servo) for servo in range(PAN_SERVO_MIN, PAN_SERVO_MAX + 1, SCAN_STEP_SERVO)
)


def laser_on_seconds():
    """How long a found object stays lit. 0 or less aims without lighting the laser."""
    return float(os.environ.get("LASER_ON_SECONDS", DEFAULT_LASER_ON_SECONDS))


class Turret:
    """Interface shared by SerialTurret and FakeTurret."""

    def __init__(self):
        self._pose = (0.0, 0.0)

    def home(self):
        raise NotImplementedError

    def goto(self, pan_deg, tilt_deg):
        """Move in vision degrees, block until settled, return the pose commanded."""
        raise NotImplementedError

    def laser_on(self):
        raise NotImplementedError

    def laser_off(self):
        raise NotImplementedError

    def close(self):
        pass

    def current_pose(self):
        return self._pose

    def can_reach(self, pan_deg, tilt_deg):
        """Only pan limits reach. Tilt is clamped to its range in _servo_pose."""
        return servo_pan(pan_deg) is not None

    def _servo_pose(self, pan_deg, tilt_deg):
        pan = servo_pan(pan_deg)
        if pan is None:
            raise TurretError(f"Pan {pan_deg:.1f} is outside the turret's range.")
        return pan, clamped_servo_tilt(tilt_deg)


class SerialTurret(Turret):
    """Talks to the Arduino over USB serial. The port opens on first use."""

    def __init__(self, port, baud=115200, serial_factory=None, ready_timeout=READY_TIMEOUT_S):
        super().__init__()
        self.port = port
        self.baud = baud
        self._factory = serial_factory
        self._ready_timeout = ready_timeout
        self._serial = None
        self._buffer = b""
        self._lock = threading.Lock()

    def home(self):
        with self._lock:
            self._ensure_open()
            self._pose = self._parse_pose(self._command("HOME", MOVE_TIMEOUT_S))
        return self._pose

    def goto(self, pan_deg, tilt_deg):
        pan, tilt = self._servo_pose(pan_deg, tilt_deg)
        with self._lock:
            self._ensure_open()
            self._pose = self._parse_pose(self._command(f"GOTO {pan} {tilt}", MOVE_TIMEOUT_S))
        return self._pose

    def laser_on(self):
        with self._lock:
            self._ensure_open()
            self._command("LASER 1", COMMAND_TIMEOUT_S)

    def laser_off(self):
        with self._lock:
            self._ensure_open()
            self._command("LASER 0", COMMAND_TIMEOUT_S)

    def laser_off_if_open(self):
        """Best effort for shutdown. Never opens the port, which would reset the board anyway."""
        with self._lock:
            if self._serial is None:
                return
            try:
                self._command("LASER 0", COMMAND_TIMEOUT_S)
            except TurretError:
                pass

    def close(self):
        with self._lock:
            self._close()

    def _ensure_open(self):
        if self._serial is not None:
            return
        factory = self._factory
        if factory is None:
            if serial is None:
                raise TurretError("pyserial is not installed. Run pip install -r requirements.txt.")
            factory = serial.Serial
        try:
            self._serial = factory(self.port, self.baud, timeout=0.1)
        except OSError as exc:  # serial.SerialException is an OSError.
            raise TurretError(
                f"Arduino not found on {self.port}. Is it plugged in, and is the Serial Monitor closed?"
            ) from exc
        self._buffer = b""
        # Opening the port resets the board. Commands sent before READY are lost.
        deadline = time.monotonic() + self._ready_timeout
        while True:
            line = self._read_line(deadline)
            if line is None:
                self._close()
                raise TurretError(
                    f"The Arduino on {self.port} did not send READY. Is the turret firmware uploaded?"
                )
            if line == "READY":
                break
        self._pose = self._parse_pose(self._command("HOME", MOVE_TIMEOUT_S))

    def _command(self, line, timeout):
        try:
            self._serial.write((line + "\n").encode("ascii"))
            deadline = time.monotonic() + timeout
            while True:
                reply = self._read_line(deadline)
                if reply is None:
                    raise TurretError(f"The turret did not answer {line!r}.")
                if reply == "READY":
                    raise TurretError("The Arduino reset during a command. Check the servo battery.")
                if reply == "OK" or reply.startswith("OK "):
                    return reply
                if reply.startswith("ERR"):
                    raise TurretError(f"The turret refused {line!r}: {reply[3:].strip() or 'error'}.")
                # Anything else is noise; keep waiting for the reply.
        except TurretError:
            self._close()
            raise
        except OSError as exc:
            self._close()
            raise TurretError(f"Lost the Arduino on {self.port}: {exc}") from exc

    def _read_line(self, deadline):
        """Next non-empty line, or None at the deadline."""
        while True:
            if b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                text = raw.decode("ascii", errors="replace").strip()
                if text:
                    return text
                continue
            if time.monotonic() >= deadline:
                return None
            self._buffer += self._serial.readline()

    def _parse_pose(self, reply):
        parts = reply.split()
        try:
            pan, tilt = int(parts[1]), int(parts[2])
        except (IndexError, ValueError) as exc:
            raise TurretError(f"Unexpected turret reply {reply!r}.") from exc
        return vision_pan(pan), vision_tilt(tilt)

    def _close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except OSError:
                pass
        self._serial = None
        self._buffer = b""


class FakeTurret(Turret):
    """In-memory turret for tests. Set fail_next to an exception to raise it once."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.laser = False
        self.fail_next = None

    def _record(self, *call):
        self.calls.append(call)
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error

    def home(self):
        self._record("home")
        self._pose = (vision_pan(PAN_HOME_SERVO), vision_tilt(TILT_LEVEL_SERVO))
        return self._pose

    def goto(self, pan_deg, tilt_deg):
        self._record("goto", float(pan_deg), float(tilt_deg))
        pan, tilt = self._servo_pose(pan_deg, tilt_deg)
        self.laser = False  # the firmware turns the laser off on every move
        self._pose = (vision_pan(pan), vision_tilt(tilt))
        return self._pose

    def laser_on(self):
        self._record("laser_on")
        self.laser = True

    def laser_off(self):
        self._record("laser_off")
        self.laser = False


_turret = None
_turret_lock = threading.Lock()
_warned_none = False
_laser_timer = None
_laser_timer_lock = threading.Lock()


def get_turret():
    """The process-wide turret, or None when TURRET=none."""
    global _turret, _warned_none
    mode = os.environ.get("TURRET", "serial").strip().lower() or "serial"
    if mode == "none":
        if not _warned_none:
            logger.warning("TURRET=none: scans are handheld and every photo is tagged pan 0.")
            _warned_none = True
        return None
    if mode != "serial":
        raise TurretError(f"TURRET must be 'serial' or 'none', not {mode!r}.")
    with _turret_lock:
        if _turret is None:
            port = os.environ.get("ARDUINO_PORT", "").strip()
            if not port:
                raise TurretError(
                    "ARDUINO_PORT is not set. Add it to .env (for example COM4), or set TURRET=none."
                )
            baud = int(os.environ.get("ARDUINO_BAUD", "115200"))
            _turret = SerialTurret(port, baud)
            atexit.register(_turret.laser_off_if_open)
        return _turret


def schedule_laser_off(turret, seconds):
    """Turn the laser off after seconds, replacing any earlier timer."""
    global _laser_timer
    with _laser_timer_lock:
        if _laser_timer is not None:
            _laser_timer.cancel()
        _laser_timer = threading.Timer(seconds, _safe_laser_off, args=(turret,))
        _laser_timer.daemon = True
        _laser_timer.start()


def cancel_laser_timer():
    global _laser_timer
    with _laser_timer_lock:
        if _laser_timer is not None:
            _laser_timer.cancel()
            _laser_timer = None


def _safe_laser_off(turret):
    try:
        turret.laser_off()
        print("Laser off (timer)")
    except Exception:
        logger.exception("Could not turn the laser off")
