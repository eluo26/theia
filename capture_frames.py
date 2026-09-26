"""Capture still frames from the iPhone's DroidCam stream and save them to disk.

Usage:
    python capture_frames.py
    python capture_frames.py --count 2 --interval 1.0 --out captures
"""
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

DEFAULT_URL = "http://172.20.10.1:4747/video"


class LatestFrameReader:
    """Reads the stream on a background thread and keeps only the newest frame.

    OpenCV buffers network frames, so reading on demand can return a frame
    from seconds ago. Draining the stream continuously avoids that.
    """

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open stream at {url}. Is DroidCam open on the phone, "
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
        raise TimeoutError(f"No new frame from the stream within {timeout} s")

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


def capture(url, count, interval, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reader = LatestFrameReader(url)
    try:
        reader.read(timeout=10.0)
        start = time.monotonic()
        previous_time = None
        for i in range(count):
            target = start + i * interval
            time.sleep(max(0.0, target - time.monotonic()))
            frame, frame_time = reader.read(newer_than=target)

            path = out_dir / f"{stamp}_{i}.jpg"
            cv2.imwrite(str(path), frame)
            height, width = frame.shape[:2]
            gap = "" if previous_time is None else f", {frame_time - previous_time:.3f} s after previous"
            print(f"Saved {path} ({width}x{height}{gap})")
            previous_time = frame_time
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="DroidCam stream URL")
    parser.add_argument("--count", type=int, default=2, help="number of pictures to take")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between pictures")
    parser.add_argument("--out", default="captures", help="folder to save pictures in")
    args = parser.parse_args()
    capture(args.url, args.count, args.interval, args.out)


if __name__ == "__main__":
    main()
