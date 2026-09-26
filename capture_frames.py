"""Capture still frames from the iPhone's DroidCam stream and save them to disk.

Usage:
    python capture_frames.py
    python capture_frames.py --count 2 --interval 1.0 --out captures
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from server.camera import DEFAULT_STREAM_URL, LatestFrameReader


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
    parser.add_argument("--url", default=DEFAULT_STREAM_URL, help="DroidCam stream URL")
    parser.add_argument("--count", type=int, default=2, help="number of pictures to take")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between pictures")
    parser.add_argument("--out", default="captures", help="folder to save pictures in")
    args = parser.parse_args()
    capture(args.url, args.count, args.interval, args.out)


if __name__ == "__main__":
    main()
