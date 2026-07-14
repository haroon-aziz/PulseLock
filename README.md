# PulseLock

Two-factor biometric authentication that fuses a **hand-gesture password** with **remote photoplethysmography (rPPG) liveness detection**. A gesture sequence alone can be replayed from a video; PulseLock also verifies that a live, physiologically plausible pulse signal is present in the face region during the same window, which a photo, screen replay, or static mask cannot produce.

## How it works

1. **Gesture password** — MediaPipe Hands tracks landmark geometry per frame and classifies a small gesture vocabulary (`fist`, `palm`, `peace`, `point`, `thumbs_up`). The frame-level stream is collapsed into a gesture sequence.
2. **rPPG liveness** — MediaPipe Face Mesh extracts forehead/cheek ROIs. The POS algorithm (Wang et al.) recovers a pulse waveform from subtle RGB color changes caused by blood volume pulse. A bandpass filter + FFT peak gives heart rate and SNR.
3. **Fusion** — Access is granted only if the observed gesture sequence matches the enrolled sequence (DTW distance below threshold) **and** the rPPG signal is within a physiologically plausible HR range with sufficient SNR.

## Structure

```
pulselock/
  face_roi.py     Face Mesh ROI extraction
  rppg.py         POS algorithm + liveness scoring
  gesture.py      Hand landmark gesture classification
  enrollment.py   Enrollment pipeline
  auth.py         Verification pipeline
  utils.py        Ring buffer, filtering, DTW
demo/
  enroll.py       CLI: python demo/enroll.py --user alice
  verify.py       CLI: python demo/verify.py --user alice
```

## Usage

```bash
pip install -r requirements.txt

python demo/enroll.py --user alice --duration 6
python demo/verify.py --user alice --duration 6
```

## Notes

- rPPG is sensitive to lighting and motion; keep the face steady and well-lit during both enrollment and verification.
- The gesture vocabulary and DTW threshold in `auth.py` are tunable per use case.
- This is a research/demo system, not a production access-control product — spoofing resistance has not been adversarially evaluated.

## License

MIT
