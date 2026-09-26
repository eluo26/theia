"""Read frames from the phone's DroidCam video stream."""
import os
import socket
import threading
import time
from urllib.parse import urlparse

import cv2

DEFAULT_STREAM_URL = "http://172.20.10.1:4747/video"


class CameraError(RuntimeError):
    pass


def check_reachable(url, timeout=3.0):
    """Fail fast if nothing is listening at the stream's host and port.

    OpenCV can take over a minute to give up on an unreachable host, so we
    try a plain TCP connection first.
    """
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as error:
        raise CameraError(
            f"Could not reach the camera at {host}:{port} ({error.strerror or error}). "
            "Is the phone on the hotspot and DroidCam open on screen?"
        ) from error


class LatestFrameReader:
    """Reads the stream on a background thread and keeps only the newest frame.

    OpenCV buffers network frames, so reading on demand can return a frame
    from seconds ago. Draining the stream continuously avoids that.
    """

    def __init__(self, url=DEFAULT_STREAM_URL):
        check_reachable(url)
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open the camera stream at {url}. Is DroidCam open on the phone, "
                "and is no other client (such as a browser tab) connected to it?"
            )
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_time = time.monotonic()

    def read(self, newer_than=0.0, timeout=5.0):
        """Return (frame, arrival_time) for a frame that arrived after `newer_than`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_time > newer_than:
                    return self._frame.copy(), self._frame_time
            time.sleep(0.005)
        raise CameraError(f"No new frame from the camera stream within {timeout} s")

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


class SharedCamera:
    """One connection to the phone, shared by the live preview and label scans.

    DroidCam may only allow one viewer at a time, so everything in the server
    reads from this single connection. It opens on first use and closes after
    `idle_seconds` without reads, freeing the phone for other programs such as
    the turret code. Set CAMERA_STREAM_URL to use a different stream address.
    """

    def __init__(self, url=None, idle_seconds=15.0):
        self.url = url or os.environ.get("CAMERA_STREAM_URL", DEFAULT_STREAM_URL)
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._reader = None
        self._last_used = 0.0
        threading.Thread(target=self._close_when_idle, daemon=True).start()

    def _get_reader(self):
        with self._lock:
            self._last_used = time.monotonic()
            if self._reader is None:
                self._reader = LatestFrameReader(self.url)
            return self._reader

    def _discard(self, reader):
        with self._lock:
            if self._reader is reader:
                self._reader = None
        reader.close()

    def read(self, newer_than=0.0, timeout=5.0):
        """Return (frame, arrival_time), connecting to the phone if needed."""
        reader = self._get_reader()
        try:
            return reader.read(newer_than=newer_than, timeout=timeout)
        except CameraError:
            self._discard(reader)
            raise

    def _close_when_idle(self):
        while True:
            time.sleep(2)
            with self._lock:
                reader = self._reader
                idle = reader is not None and time.monotonic() - self._last_used > self.idle_seconds
                if idle:
                    self._reader = None
            if idle:
                reader.close()
