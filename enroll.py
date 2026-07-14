import argparse
import sys
import cv2

sys.path.insert(0, "..")
from pulselock.enrollment import Enroller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    enroller = Enroller(cap, duration_s=args.duration)
    record = enroller.run(args.user)
    cap.release()

    print(f"Enrolled user: {record['user_id']}")
    print(f"Gesture sequence: {record['gesture_sequence']}")
    print(f"Baseline HR: {record['baseline_hr_bpm']}")


if __name__ == "__main__":
    main()
