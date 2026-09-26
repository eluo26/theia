"""Log laser detection and a correction step from two still images."""
import argparse
import json
import sys

import cv2

from server.laser import DotNotFound, evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', required=True)
    parser.add_argument('--on', required=True)
    parser.add_argument('--target', type=float, nargs=2, required=True, metavar=('X', 'Y'),
                        help='Target center in full-resolution laser-on pixels (x right, y down).')
    parser.add_argument('--tolerance', type=float, default=5)
    parser.add_argument('--no-align', action='store_true', help='Use only for an unchanged, fixed camera.')
    args = parser.parse_args()
    off, on = cv2.imread(args.off), cv2.imread(args.on)
    try:
        if on is None or off is None:
            raise ValueError('Could not read both image files.')
        x, y = args.target
        if not (0 <= x < on.shape[1] and 0 <= y < on.shape[0]):
            raise ValueError('Target must lie inside the laser-on image.')
        result = evaluate_pair(off, on, args.target, args.tolerance, align=not args.no_align)
    except (ValueError, DotNotFound) as error:
        print(json.dumps({'status': 'no_correction', 'error': str(error)}, indent=2))
        return 1
    print(json.dumps({'image_size': {'width': on.shape[1], 'height': on.shape[0]},
                      'tolerance_px': args.tolerance, **result}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
