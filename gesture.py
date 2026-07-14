import numpy as np
import mediapipe as mp

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

GESTURES = ["fist", "palm", "peace", "point", "thumbs_up", "unknown"]


class HandGestureClassifier:
    def __init__(self, max_hands=1, min_det_conf=0.6, min_track_conf=0.6):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def _finger_states(self, landmarks):
        states = []
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            states.append(landmarks[tip].y < landmarks[pip].y)
        thumb_extended = landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x
        return [thumb_extended] + states

    def _classify(self, states):
        thumb, index, middle, ring, pinky = states
        extended_count = sum(states)

        if extended_count == 0:
            return "fist"
        if extended_count == 5:
            return "palm"
        if index and middle and not ring and not pinky and not thumb:
            return "peace"
        if index and not middle and not ring and not pinky and not thumb:
            return "point"
        if thumb and not index and not middle and not ring and not pinky:
            return "thumbs_up"
        return "unknown"

    def classify_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        landmarks = result.multi_hand_landmarks[0].landmark
        states = self._finger_states(landmarks)
        label = self._classify(states)
        return label

    def close(self):
        self.hands.close()


def collapse_sequence(labels):
    collapsed = []
    prev = None
    for label in labels:
        if label is None:
            continue
        if label != prev:
            collapsed.append(label)
        prev = label
    return [l for l in collapsed if l != "unknown"]
