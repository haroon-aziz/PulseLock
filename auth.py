import json
import os
import time

from .face_roi import FaceROIExtractor
from .gesture import HandGestureClassifier, collapse_sequence
from .rppg import RPPGLivenessScorer
from .utils import RingBuffer, dtw_distance
from .enrollment import DEFAULT_STORE

DTW_THRESHOLD = 0.35


class Authenticator:
    def __init__(self, capture, duration_s=6.0, fs=30.0, store_dir=DEFAULT_STORE):
        self.capture = capture
        self.duration_s = duration_s
        self.fs = fs
        self.store_dir = store_dir

    def _load_enrollment(self, user_id):
        path = os.path.join(self.store_dir, f"{user_id}.json")
        with open(path, "r") as f:
            return json.load(f)

    def verify(self, user_id):
        record = self._load_enrollment(user_id)
        enrolled_sequence = record["gesture_sequence"]

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

        observed_sequence = collapse_sequence(gesture_labels)
        norm_dist = dtw_distance(enrolled_sequence, observed_sequence)
        gesture_match = norm_dist <= DTW_THRESHOLD

        scorer = RPPGLivenessScorer(fs=self.fs)
        liveness = scorer.score(rgb_buffer.array()) if len(rgb_buffer) else {"valid": False, "reason": "no_face"}

        access_granted = gesture_match and liveness["valid"]

        return {
            "user_id": user_id,
            "access_granted": access_granted,
            "gesture_match": gesture_match,
            "gesture_distance": norm_dist,
            "observed_sequence": observed_sequence,
            "liveness": liveness,
        }
