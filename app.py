import cv2
import numpy as np
import librosa
import gradio as gr
import os

# ================= CONFIG =================
FRAME_SKIP = 4
RESIZE_W = 320
MIN_DURATION_SEC = 4

IRREG_T = 0.06
VERTICAL_MOUTH_T = 0.08
PVC_WIN = 20
MAX_LAG = 3

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= UTIL =================
def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6) if len(x) else x

def smooth(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode="same") if len(x) >= k else x

def video_duration(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps

# ================= AUDIO =================
def load_audio_onset(video):
    try:
        audio, sr = librosa.load(video, sr=16000, mono=True)
        onset = librosa.onset.onset_strength(y=audio, sr=sr)
        return smooth(normalize(onset))
    except:
        return np.array([])

# ================= PHASE 2 + 3 (INTEGRATED) =================
def extract_with_semantic_gate(video, onset):

    cap = cv2.VideoCapture(video)
    ret, first = cap.read()
    if not ret:
        return None, None, True

    ratio = RESIZE_W / first.shape[1]
    new_h = int(first.shape[0] * ratio)

    prev = cv2.resize(first, (RESIZE_W, new_h))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    mouth_motion = []
    frame_motion = []

    frame_id = 0
    face_box = None

    speech_frames = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP:
            continue

        frame = cv2.resize(frame, (RESIZE_W, new_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        processed_frames += 1

        # Global motion
        frame_motion.append(np.mean(cv2.absdiff(gray, prev_gray)))

        # Face detection occasionally
        if frame_id % 12 == 0 or face_box is None:
            faces = face_detector.detectMultiScale(gray, 1.1, 4)
            face_box = faces[0] if len(faces) == 1 else None

        if face_box is not None:
            x, y, w, h = face_box
            roi = gray[int(y+0.65*h):int(y+0.85*h),
                       int(x+0.30*w):int(x+0.70*w)]
            prev_roi = prev_gray[int(y+0.65*h):int(y+0.85*h),
                                 int(x+0.30*w):int(x+0.70*w)]
            if roi.size and prev_roi.size:
                val = np.mean(cv2.absdiff(roi, prev_roi))
                mouth_motion.append(val)
            else:
                mouth_motion.append(0)
        else:
            mouth_motion.append(0)

        # 🔴 Live Semantic Gate
        if processed_frames < len(onset):
            if onset[processed_frames] > np.percentile(onset, 60):
                speech_frames += 1

        if processed_frames > 10:
            speech_ratio = speech_frames / processed_frames
            vertical_score = np.mean(np.abs(mouth_motion))

            if speech_ratio > 0.25 and vertical_score < VERTICAL_MOUTH_T:
                cap.release()
                return None, None, True

        prev_gray = gray

    cap.release()

    return (
        smooth(normalize(np.array(mouth_motion))),
        smooth(normalize(np.array(frame_motion))),
        False
    )

# ================= PVC =================
def lag_corr(x, y):
    vals = []
    for lag in range(-MAX_LAG, MAX_LAG+1):
        if lag < 0:
            vals.append(np.corrcoef(x[:lag], y[-lag:])[0,1])
        elif lag > 0:
            vals.append(np.corrcoef(x[lag:], y[:-lag])[0,1])
        else:
            vals.append(np.corrcoef(x, y)[0,1])
    vals = [v for v in vals if not np.isnan(v)]
    return max(vals) if vals else 0

def pvc_check(mouth_motion, onset):
    if len(mouth_motion) < PVC_WIN or len(onset) < PVC_WIN:
        return False, 1.0

    L = min(len(mouth_motion), len(onset))
    mouth_motion = mouth_motion[:L]
    onset = onset[:L]

    scores = []
    for i in range(0, L-PVC_WIN, PVC_WIN):
        scores.append(lag_corr(
            mouth_motion[i:i+PVC_WIN],
            onset[i:i+PVC_WIN]
        ))

    pvc_score = np.mean(scores)
    pvc_T = pvc_score - 0.5*np.std(scores)

    return pvc_score < pvc_T, pvc_score

# ================= IRREGULAR =================
def irregular(frame_motion):
    if len(frame_motion) == 0:
        return False, 0.0
    ratio = np.mean(frame_motion > frame_motion.mean() + 2*frame_motion.std())
    return ratio > IRREG_T, ratio

# ================= MAIN =================
def analyze_video(video):

    if video is None:
        return "NO VIDEO", "", "", ""

    if video_duration(video) < MIN_DURATION_SEC:
        return "FALSE", "Low", "", "Video too short"

    onset = load_audio_onset(video)

    mouth_motion, frame_motion, non_human = extract_with_semantic_gate(video, onset)

    if non_human:
        return "FALSE", "High", "", "Non-human speaking entity detected"

    if mouth_motion is None:
        return "FALSE", "Low", "", "Feature extraction failed"

    pvc_flag, pvc_score = pvc_check(mouth_motion, onset)
    if pvc_flag:
        return "FALSE", "Medium", f"PVC score: {pvc_score:.3f}", \
               "Phonetic-visual inconsistency detected"

    ir_flag, ir_score = irregular(frame_motion)
    if ir_flag:
        return "FALSE", "Medium", f"Irregularity: {ir_score:.3f}", \
               "Visual motion anomaly detected"

    return "TRUE", "High", "All checks passed", \
           "Integrated detection successful"

# ================= UI =================
gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Textbox(label="Result (TRUE / FALSE)"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Metrics"),
        gr.Textbox(label="Explanation")
    ],
    title="🛡️ DeepShield – Integrated Early-Gate Model"
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))