import numpy as np
import mediapipe as mp

FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 21, 71, 68]
CHEEK_L_IDX = [50, 205, 206, 187]
CHEEK_R_IDX = [280, 425, 426, 411]


class FaceROIExtractor:
    def __init__(self, max_faces=1, min_det_conf=0.6, min_track_conf=0.6):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def _landmarks_to_px(self, landmarks, idxs, w, h):
        pts = []
        for i in idxs:
            lm = landmarks[i]
            pts.append((int(lm.x * w), int(lm.y * h)))
        return np.array(pts)

    def extract(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        regions = []
        for idxs in (FOREHEAD_IDX, CHEEK_L_IDX, CHEEK_R_IDX):
            pts = self._landmarks_to_px(landmarks, idxs, w, h)
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(frame_bgr[y0:y1, x0:x1])
        if not regions:
            return None
        means = [r.reshape(-1, 3).mean(axis=0) for r in regions if r.size > 0]
        if not means:
            return None
        avg_bgr = np.mean(means, axis=0)
        return avg_bgr[::-1]

    def close(self):
        self.mesh.close()
