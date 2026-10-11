"""
Face tools used by the app. You do NOT need to read or change this file.
"""
import os, random, urllib.request
import cv2
import numpy as np
import config

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_models():
    """Download the two face models (once). They are small (about 40 MB total)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                f.write(r.read())
        except Exception as e:
            raise RuntimeError(
                f"Could not download {name}. Reason: {e}. "
                f"FIX: open this link in your browser, download the file and save it inside the folder "
                f"{MODEL_DIR}  -> {url}")
        if os.path.getsize(path) < 100_000:
            os.remove(path)
            raise RuntimeError(f"{name} did not download correctly. Please download it manually.")


class FaceEngine:
    """Finds faces in a picture and turns each face into a number-fingerprint."""

    def __init__(self):
        ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
            "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"), "")

    def faces_and_features(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        out = []
        if faces is None:
            return out
        for f in faces:
            aligned = self.recognizer.alignCrop(img, f)
            feat = self.recognizer.feature(aligned)
            x, y, bw, bh = [int(v) for v in f[:4]]
            out.append({"box": (x, y, bw, bh), "feat": feat})
        return out

    def similarity(self, f1, f2):
        return float(self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE))


def shrink(img, max_w=None):
    max_w = max_w or config.MAX_FRAME_WIDTH
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (int(w * s), int(h * s)))


def read_frames(video_path, n):
    """Pick n frames at moments spread across the whole video (with a little randomness)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video looks empty: {video_path}")
    n = max(1, min(n, total))
    slot = total / n
    picks = [min(total - 1, int(i * slot + random.random() * slot)) for i in range(n)]
    frames = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(shrink(f))
    cap.release()
    if not frames:
        raise ValueError("Could not read any picture from the video.")
    return frames


def camera_problem(frame):
    """Returns None if the picture looks fine, or a short reason if not."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < config.DARK_LIMIT:
        return "too dark (lens covered?)"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < config.BLUR_LIMIT:
        return "blurred or smeared"
    return None


def parse_enrol_name(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    pid = parts[0].upper()
    name = parts[1] if len(parts) > 1 else pid
    return pid, name


def load_enrolled(engine, folder, learned_dir=None):
    """Reads photos in the 'enrol' folder. File name = ID_Name.jpg  e.g. T001_Ravi.jpg
    Also loads extra 'learned' samples (made when a trainer confirms a face)."""
    people, problems = {}, []
    exts = (".jpg", ".jpeg", ".png")
    if not os.path.isdir(folder):
        return people, ["enrol folder not found"]
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(exts):
            continue
        pid, name = parse_enrol_name(fn)
        img = cv2.imread(os.path.join(folder, fn))
        if img is None:
            problems.append(f"{fn}: cannot read picture")
            continue
        faces = engine.faces_and_features(shrink(img))
        if not faces:
            problems.append(f"{fn}: no face found - use a clearer, front-facing photo")
            continue
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        entry = people.setdefault(pid, {"name": name, "feats": [], "learned": 0})
        entry["feats"].append(faces[0]["feat"])
    if learned_dir and os.path.isdir(learned_dir):
        for fn in sorted(os.listdir(learned_dir)):
            if fn.endswith(".npy"):
                pid = fn.split("__")[0].upper()
                if pid in people:
                    people[pid]["feats"].append(np.load(os.path.join(learned_dir, fn)))
                    people[pid]["learned"] += 1
    return people, problems


def find_duplicates(engine, people):
    """Same face enrolled under two different IDs = possible fake trainee."""
    ids = list(people)
    dups = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            best = max(engine.similarity(a, b)
                       for a in people[ids[i]]["feats"] for b in people[ids[j]]["feats"])
            if best >= config.DUPLICATE_THRESHOLD:
                dups.append({"id_1": ids[i], "name_1": people[ids[i]]["name"],
                             "id_2": ids[j], "name_2": people[ids[j]]["name"],
                             "similarity": round(best, 2)})
    return dups
