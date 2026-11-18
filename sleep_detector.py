"""
Sleep Detector AI - Real-time sleep/drowsiness detector with UI.
Uses MediaPipe FaceMesh + Eye Aspect Ratio (EAR) + head pose.
All in one folder: H:\sleep_detector
"""
import os, sys, time, threading, math
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import mediapipe as mp

# ---------- Config ----------
EAR_THRESHOLD = 0.21          # below -> eyes closed (lowered; typical awake EAR ~0.28-0.35)
CONSEC_FRAMES_SLEEP = 20       # frames of closed eyes -> sleeping
CONSEC_FRAMES_DROWSY = 12     # more frames required to flag drowsy
HEAD_PITCH_DOWN = 30          # degrees, head tilted down -> drowsy/sleeping

# MediaPipe landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def ear(landmarks, idx):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in idx]
    A = np.linalg.norm(p[1]-p[5])
    B = np.linalg.norm(p[2]-p[4])
    C = np.linalg.norm(p[0]-p[3])
    return (A+B)/(2.0*C+1e-6)

# ---------- App ----------
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleep Detector AI")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2e")

        self.running = False
        self.cap = None
        self.thread = None
        self.lock = threading.Lock()

        self.status = "Idle"
        self.ear_val = 0.0
        self.closed_frames = 0
        self.sleep_count = 0
        self.total_sleep_time = 0.0
        self.sleep_start = None
        self.last_state = "Awake"

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.root, text="Sleep Detector AI", font=("Segoe UI",20,"bold"),
                         bg="#1e1e2e", fg="#cdd6f4")
        title.pack(pady=8)

        self.video_label = tk.Label(self.root, bg="#000")
        self.video_label.pack(padx=10, pady=5)

        info = tk.Frame(self.root, bg="#1e1e2e")
        info.pack(fill=tk.X, padx=10, pady=5)

        self.status_var = tk.StringVar(value="Status: Idle")
        self.ear_var = tk.StringVar(value="EAR: 0.00")
        self.closed_var = tk.StringVar(value="Closed frames: 0")
        self.sleeps_var = tk.StringVar(value="Sleep events: 0")
        self.time_var = tk.StringVar(value="Total sleep time: 0s")

        for v in (self.status_var, self.ear_var, self.closed_var, self.sleeps_var, self.time_var):
            tk.Label(info, textvariable=v, font=("Consolas",12), bg="#1e1e2e",
                     fg="#a6e3a1", anchor="w").pack(fill=tk.X)

        btns = tk.Frame(self.root, bg="#1e1e2e")
        btns.pack(pady=10)
        self.start_btn = ttk.Button(btns, text="Start Detection", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Reset Stats", command=self.reset).grid(row=0, column=2, padx=5)
        ttk.Button(btns, text="Quit", command=self.quit).grid(row=0, column=3, padx=5)

        self.log = tk.Text(self.root, height=6, bg="#313244", fg="#cdd6f4", font=("Consolas",9))
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def log_msg(self, m):
        self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {m}\n")
        self.log.see(tk.END)

    def start(self):
        if self.running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Could not open webcam")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.log_msg("Detection started")

    def stop(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap=None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Idle")
        self.log_msg("Detection stopped")

    def reset(self):
        self.sleep_count=0; self.total_sleep_time=0.0; self.closed_frames=0
        self.sleeps_var.set("Sleep events: 0"); self.time_var.set("Total sleep time: 0s")
        self.log_msg("Stats reset")

    def quit(self):
        self.stop(); self.root.destroy()

    def loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            state = "No face"
            earv = 0.0
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                left = ear(lm, LEFT_EYE); right = ear(lm, RIGHT_EYE)
                earv = (left+right)/2.0
                # head pitch estimate using nose vs chin
                nose = lm[1]; chin = lm[152]; forehead = lm[10]
                pitch = (chin.y - nose.y)*100  # crude
                # determine
                if earv < EAR_THRESHOLD:
                    self.closed_frames += 1
                else:
                    if self.closed_frames >= CONSEC_FRAMES_SLEEP and self.sleep_start is not None:
                        dur = time.time()-self.sleep_start
                        self.total_sleep_time += dur
                        self.time_var.set(f"Total sleep time: {self.total_sleep_time:.1f}s")
                        self.log_msg(f"Sleep ended ({dur:.1f}s)")
                        self.sleep_start = None
                    self.closed_frames = 0

                if self.closed_frames >= CONSEC_FRAMES_SLEEP:
                    state = "SLEEPING"
                    if self.sleep_start is None:
                        self.sleep_start = time.time()
                        self.sleep_count += 1
                        self.sleeps_var.set(f"Sleep events: {self.sleep_count}")
                        self.log_msg("Sleep detected!")
                elif self.closed_frames >= CONSEC_FRAMES_DROWSY or pitch > 12:
                    state = "Drowsy"
                else:
                    state = "Awake"

            self.ear_val = earv
            self.status = state
            self._update_ui(frame, state, earv)

            # alarm
            if state == "SLEEPING":
                # beep (non-blocking-ish)
                try:
                    import winsound
                    winsound.Beep(1000, 200)
                except Exception:
                    pass

            time.sleep(0.03)

    def _update_ui(self, frame, state, earv):
        color = {"Awake":(0,200,0),"Drowsy":(0,200,255),"SLEEPING":(0,0,255),"No face":(128,128,128)}.get(state,(255,255,255))
        cv2.putText(frame, state, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"EAR: {earv:.2f}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(frame, f"Closed frames: {self.closed_frames}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640,480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.status_var.set(f"Status: {state}")
        self.ear_var.set(f"EAR: {earv:.2f}")
        self.closed_var.set(f"Closed frames: {self.closed_frames}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
