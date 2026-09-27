"""Run with python -m unittest discover -s server -p 'test_*.py'."""
import time
import unittest
from unittest.mock import patch

import turret as turret_module
from turret import (
    SCAN_PANS,
    FakeTurret,
    SerialTurret,
    TurretError,
    clamped_servo_tilt,
    get_turret,
    schedule_laser_off,
    servo_pan,
    servo_tilt,
)


class FakeSerial:
    """Answers like firmware/theia_turret, one reply per command."""

    def __init__(self, boot=(b"READY\n",), replies=None, silent=False):
        self.pending = bytearray(b"".join(boot))
        self.replies = replies or {}
        self.silent = silent
        self.written = []
        self.closed = False

    def write(self, data):
        line = data.decode("ascii").strip()
        self.written.append(line)
        if self.silent:
            return
        if line in self.replies:
            reply = self.replies[line]
        elif line == "HOME":
            reply = "OK 0 0"
        elif line.startswith("GOTO "):
            reply = "OK " + line[5:]
        else:
            reply = "OK"
        self.pending += (reply + "\r\n").encode("ascii")

    def readline(self):
        if not self.pending:
            time.sleep(0.001)
            return b""
        index = self.pending.find(b"\n")
        end = len(self.pending) if index < 0 else index + 1
        line = bytes(self.pending[:end])
        del self.pending[:end]
        return line

    def close(self):
        self.closed = True


def serial_turret(fake):
    return SerialTurret("COM9", serial_factory=lambda *_args, **_kw: fake, ready_timeout=0.2)


class ConversionTests(unittest.TestCase):
    def test_pan_sign_matches_bench(self):
        # Bench: increasing servo angle turns left, so vision pan is negative.
        self.assertEqual(servo_pan(0), 0)
        self.assertEqual(servo_pan(-30), 30)
        self.assertEqual(servo_pan(-180), 180)
        self.assertIsNone(servo_pan(10))

    def test_wrapped_azimuth_at_far_end_still_reaches(self):
        # atan2 reports -185 as +175; that is servo 185, still out of range.
        self.assertIsNone(servo_pan(175))
        # -178 reported as +182 is servo 178.
        self.assertEqual(servo_pan(182), 178)

    def test_tilt_only_points_down(self):
        self.assertEqual(servo_tilt(0), 0)
        self.assertEqual(servo_tilt(-45), 45)
        self.assertIsNone(servo_tilt(5))
        self.assertIsNone(servo_tilt(-50))

    def test_clamped_tilt_stays_in_range(self):
        self.assertEqual(clamped_servo_tilt(3.9), 0)
        self.assertEqual(clamped_servo_tilt(-10), 10)
        self.assertEqual(clamped_servo_tilt(-50), 45)

    def test_scan_pans_follow_servo_range(self):
        self.assertEqual(SCAN_PANS, (0.0, -30.0, -60.0, -90.0, -120.0, -150.0, -180.0))


class SerialTurretTests(unittest.TestCase):
    def test_waits_for_ready_then_homes(self):
        fake = FakeSerial(boot=(b"garbage\n", b"READY\n"))
        driver = serial_turret(fake)
        self.assertEqual(driver.goto(-30, -10), (-30.0, -10.0))
        self.assertEqual(fake.written, ["HOME", "GOTO 30 10"])
        self.assertEqual(driver.current_pose(), (-30.0, -10.0))

    def test_err_reply_raises(self):
        fake = FakeSerial(replies={"LASER 1": "ERR broken"})
        driver = serial_turret(fake)
        with self.assertRaises(TurretError):
            driver.laser_on()
        self.assertTrue(fake.closed)

    def test_silence_times_out(self):
        driver = serial_turret(FakeSerial(boot=()))
        with self.assertRaisesRegex(TurretError, "READY"):
            driver.home()

    def test_out_of_range_is_refused_before_sending(self):
        fake = FakeSerial()
        driver = serial_turret(fake)
        with self.assertRaises(TurretError):
            driver.goto(20, 0)
        self.assertEqual(fake.written, [])

    def test_unreachable_tilt_still_pans(self):
        # A bottle at elevation +3.9 is above level; pan to it with tilt at the limit.
        fake = FakeSerial()
        driver = serial_turret(fake)
        self.assertTrue(driver.can_reach(-34.4, 3.9))
        self.assertEqual(driver.goto(-34.4, 3.9), (-34.0, 0.0))
        self.assertEqual(driver.goto(-60, -70), (-60.0, -45.0))
        self.assertEqual(fake.written, ["HOME", "GOTO 34 0", "GOTO 60 45"])

    def test_missing_port_is_a_clear_error(self):
        with patch.dict("os.environ", {"TURRET": "serial", "ARDUINO_PORT": ""}), patch.object(
            turret_module, "_turret", None
        ):
            with self.assertRaisesRegex(TurretError, "ARDUINO_PORT"):
                get_turret()

    def test_turret_none_disables_hardware(self):
        with patch.dict("os.environ", {"TURRET": "none"}):
            self.assertIsNone(get_turret())

    def test_open_failure_explains_serial_monitor(self):
        def refuse(*_args, **_kw):
            raise OSError("access denied")

        driver = SerialTurret("COM9", serial_factory=refuse)
        with self.assertRaisesRegex(TurretError, "Serial Monitor"):
            driver.home()


class LaserTimerTests(unittest.TestCase):
    def test_auto_off(self):
        fake = FakeTurret()
        fake.laser_on()
        schedule_laser_off(fake, 0.01)
        deadline = time.monotonic() + 1.0
        while fake.laser and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(fake.laser)
        self.assertEqual(fake.calls[-1], ("laser_off",))


if __name__ == "__main__":
    unittest.main()
