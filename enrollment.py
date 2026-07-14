import json
import os
import time

from .face_roi import FaceROIExtractor
from .gesture import HandGestureClassifier, collapse_sequence
from .rppg import RPPGLivenessScorer
from .utils import RingBuffer

DEFAULT_STORE = os.path.join(os.path.dirname(__file__), "..", "enrollments")


class Enroller:
    def __init__(self, capture, duration_s=6.0, fs=30.0, store_dir=DEFAULT_STORE):
        self.capture = capture
        self.duration_s = duration_s
        self.fs = fs
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

    def run(self, user_id):
        face_extractor = FaceROIExtractor()
        gesture_classifier = HandGestureClassifier()
        rgb_buffer = RingBuffer(int(self.duration_s * self.fs))
        gesture_labels = []

        start = time.time()
        while time.time() - start < self.duration_s:
            ok, frame = self.capture.read()
            if not ok:
                break
            roi_rgb = face_extractor.extract(frame)
            if roi_rgb is not None:
                rgb_buffer.push(roi_rgb)
            label = gesture_classifier.classify_frame(frame)
            gesture_labels.append(label)

        face_extractor.close()
        gesture_classifier.close()

        scorer = RPPGLivenessScorer(fs=self.fs)
        baseline = scorer.score(rgb_buffer.array()) if len(rgb_buffer) else {"hr_bpm": None}
        sequence = collapse_sequence(gesture_labels)

        record = {
            "user_id": user_id,
            "gesture_sequence": sequence,
            "baseline_hr_bpm": baseline.get("hr_bpm"),
        }

        path = os.path.join(self.store_dir, f"{user_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

        return record
