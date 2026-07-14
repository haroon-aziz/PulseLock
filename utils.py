import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque


class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buf = deque(maxlen=size)

    def push(self, item):
        self.buf.append(item)

    def full(self):
        return len(self.buf) == self.size

    def array(self):
        return np.array(self.buf)

    def clear(self):
        self.buf.clear()

    def __len__(self):
        return len(self.buf)


def bandpass_filter(signal, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def signal_snr(signal, fs, hr_band=(0.7, 4.0)):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2
    band_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    if not np.any(band_mask):
        return 0.0, 0.0
    band_power = spectrum[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]
    signal_power = band_power[peak_idx]
    noise_power = np.sum(band_power) - signal_power + 1e-8
    snr = 10 * np.log10(signal_power / noise_power)
    hr_bpm = peak_freq * 60.0
    return hr_bpm, snr


def dtw_distance(seq_a, seq_b):
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = 0.0 if seq_a[i - 1] == seq_b[j - 1] else 1.0
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / max(n, m)
