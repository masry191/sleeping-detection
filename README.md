# Sleep Detector AI

Real-time sleep/drowsiness detector with a Tkinter UI.

## Run
```
cd H:\sleep_detector
pip install -r requirements.txt
python sleep_detector.py
```

## How it works
- MediaPipe FaceMesh detects face & eye landmarks.
- Eye Aspect Ratio (EAR) computed; if EAR < 0.25 for 15+ frames -> "SLEEPING".
- 8-14 frames -> "Drowsy".
- Head tilted down also flags drowsy.
- Beeps alarm when sleeping; tracks sleep events & total sleep time.

## Notes
- Requires a webcam.
- Adjust EAR_THRESHOLD / CONSEC_FRAMES_* in the script if needed.
