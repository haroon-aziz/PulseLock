import argparse
import sys
import cv2

sys.path.insert(0, "..")
from pulselock.auth import Authenticator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    authenticator = Authenticator(cap, duration_s=args.duration)
    result = authenticator.verify(args.user)
    cap.release()

    print(f"Access granted: {result['access_granted']}")
    print(f"Gesture match: {result['gesture_match']} (distance={result['gesture_distance']:.3f})")
    print(f"Liveness: {result['liveness']}")


if __name__ == "__main__":
    main()
