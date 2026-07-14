import numpy as np
from .utils import bandpass_filter, signal_snr

MIN_HR = 40.0
MAX_HR = 180.0
MIN_SNR_DB = -5.0


def pos_algorithm(rgb_buffer, fs):
    rgb = rgb_buffer.astype(np.float64)
    mean_rgb = rgb.mean(axis=0)
    mean_rgb[mean_rgb == 0] = 1e-8
    normed = rgb / mean_rgb

    n = normed.shape[0]
    h = np.zeros(n)
    win_len = int(1.6 * fs)
    if win_len < 5:
        win_len = 5

    for start in range(0, n - win_len + 1):
        window = normed[start:start + win_len]
        c = window.T
        mean_c = c.mean(axis=1, keepdims=True)
        c = c / mean_c

        s1 = c[1] - c[2]
        s2 = c[1] + c[2] - 2 * c[0]

        std1 = s1.std() + 1e-8
        std2 = s2.std() + 1e-8
        p = s1 + (std1 / std2) * s2

        h[start:start + win_len] += p - p.mean()

    return h


class RPPGLivenessScorer:
    def __init__(self, fs=30.0):
        self.fs = fs

    def score(self, rgb_buffer):
        if len(rgb_buffer) < int(2 * self.fs):
            return {"valid": False, "reason": "insufficient_samples", "hr_bpm": None, "snr_db": None}

        pulse = pos_algorithm(np.array(rgb_buffer), self.fs)
        try:
            filtered = bandpass_filter(pulse, self.fs)
        except Exception:
            return {"valid": False, "reason": "filter_failed", "hr_bpm": None, "snr_db": None}

        hr_bpm, snr_db = signal_snr(filtered, self.fs)

        valid = (MIN_HR <= hr_bpm <= MAX_HR) and (snr_db >= MIN_SNR_DB)
        reason = "ok" if valid else "implausible_or_low_snr"

        return {"valid": valid, "reason": reason, "hr_bpm": float(hr_bpm), "snr_db": float(snr_db)}
