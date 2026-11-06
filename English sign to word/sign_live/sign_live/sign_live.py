#!/usr/bin/env python3
"""
sign_live.py - sign a word in front of your webcam and see which word it is.

Two ways to use it
------------------
1) Ready-made ASL model (no training, 250 American Sign Language words):
       python sign_live.py asl
       python sign_live.py words            # list the 250 words it knows

2) Your own signs (any sign language, e.g. BdSL) - most accurate for YOU:
       python sign_live.py record hello water thanks --samples 15
       python sign_live.py train
       python sign_live.py custom

How a sign is captured (auto mode, default)
-------------------------------------------
  Keep your hands OUT of the camera view (in your lap) -> raise them and sign
  -> drop them out of view again. Recording starts when a hand appears and
  stops ~0.4 s after both hands disappear. Use --manual to start/stop with
  SPACE instead (useful if your hands are always visible).

Useful options
--------------
  --camera 1          use another webcam
  --complexity 0      faster MediaPipe model for slow laptops (slightly less accurate)
  --mirror            feed the mirrored image to the model (try if ASL results are poor)
  --video clip.mp4    run on a video file instead of the webcam
  --whole             with --video: treat the whole video as ONE sign
  --no-display        don't open a window (for --video)

Keys in the window: q = quit, SPACE = start/stop (manual mode),
                    u = undo last sample (record mode), s = skip word (record mode)

Requirements: Python 3.9-3.12, mediapipe==0.10.14 (see requirements.txt).
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASL_MODEL = os.path.join(HERE, "asl_model.tflite")
ASL_LABELS = os.path.join(HERE, "asl_labels.json")
MY_DIR = os.path.join(HERE, "my_signs")
MY_DATA = os.path.join(MY_DIR, "data")
MY_MODEL = os.path.join(MY_DIR, "model.pkl")

# Landmark layout used by the Kaggle "asl-signs" data (543 points per frame):
#   0-467 face | 468-488 left hand | 489-521 pose | 522-542 right hand
FACE, LH, POSE, RH = slice(0, 468), slice(468, 489), slice(489, 522), slice(522, 543)
N_LANDMARKS = 543


# --------------------------------------------------------------------------- #
# MediaPipe
# --------------------------------------------------------------------------- #
def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError:
        sys.exit("MediaPipe is not installed. Run:  pip install -r requirements.txt")
    if not hasattr(mp, "solutions"):
        sys.exit(
            f"Your MediaPipe ({mp.__version__}) is too new: it removed the Holistic API this\n"
            "script needs. Use Python 3.9-3.12 and run:  pip install mediapipe==0.10.14"
        )
    return mp


class Landmarker:
    """Runs MediaPipe Holistic on a frame and returns a (543, 3) array (NaN = not detected)."""

    def __init__(self, complexity=1):
        self.mp = load_mediapipe()
        self.holistic = self.mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=complexity,
            refine_face_landmarks=False,  # 468 face points, as in the training data
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, bgr):
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.holistic.process(rgb)
        lm = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
        for part, sl in ((res.face_landmarks, FACE), (res.left_hand_landmarks, LH),
                         (res.pose_landmarks, POSE), (res.right_hand_landmarks, RH)):
            if part is not None:
                pts = np.array([[p.x, p.y, p.z] for p in part.landmark], dtype=np.float32)
                lm[sl] = pts[: sl.stop - sl.start]
        hands = res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
        return lm, hands, res

    def draw(self, bgr, res):
        d, h = self.mp.solutions.drawing_utils, self.mp.solutions.holistic
        d.draw_landmarks(bgr, res.pose_landmarks, h.POSE_CONNECTIONS)
        d.draw_landmarks(bgr, res.left_hand_landmarks, h.HAND_CONNECTIONS)
        d.draw_landmarks(bgr, res.right_hand_landmarks, h.HAND_CONNECTIONS)


# --------------------------------------------------------------------------- #
# Cutting the stream into one-sign clips
# --------------------------------------------------------------------------- #
class Segmenter:
    """
    auto mode : start when a hand is visible for `start` frames, stop after
                `gap` frames with no hands. Trailing empty frames are trimmed.
    manual    : toggle() starts/stops recording.
    """

    def __init__(self, manual=False, start=3, gap=12, min_len=8, max_len=150, preroll=4):
        self.manual, self.start, self.gap_limit = manual, start, gap
        self.min_len, self.max_len = min_len, max_len
        self.pre = deque(maxlen=preroll)
        self._reset()

    def _reset(self, wait_release=False):
        self.state = "release" if wait_release else "idle"
        self.buf, self.gap, self.streak = [], 0, 0
        self.pre.clear()

    def _finish(self, trim):
        seq = self.buf[: len(self.buf) - trim] if trim else self.buf
        hands_left = self.state == "rec" and self.gap == 0
        self._reset(wait_release=hands_left and not self.manual)
        return np.stack(seq) if len(seq) >= self.min_len else None

    def toggle(self):
        if not self.manual:
            return None
        if self.state == "rec":
            return self._finish(0)
        self.state, self.buf = "rec", []
        return None

    def update(self, lm, hands):
        if self.manual:
            if self.state == "rec":
                self.buf.append(lm)
                if len(self.buf) >= self.max_len:
                    return self._finish(0)
            return None

        if self.state == "release":  # wait until hands leave after a forced stop
            self.streak = self.streak + 1 if not hands else 0
            if self.streak >= 3:
                self._reset()
            return None

        if self.state == "idle":
            self.pre.append(lm)
            self.streak = self.streak + 1 if hands else 0
            if self.streak >= self.start:
                self.state, self.buf, self.gap = "rec", list(self.pre), 0
            return None

        # recording
        self.buf.append(lm)
        self.gap = 0 if hands else self.gap + 1
        if self.gap >= self.gap_limit:
            return self._finish(self.gap)
        if len(self.buf) >= self.max_len:
            return self._finish(0)
        return None

    def flush(self):
        if self.state == "rec" and self.buf:
            return self._finish(self.gap)
        return None


# --------------------------------------------------------------------------- #
# Pretrained ASL model (6th place, Google ISLR Kaggle competition, MIT licence)
# --------------------------------------------------------------------------- #
def load_tflite(path):
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            m = __import__(mod, fromlist=["Interpreter"])
            return m.Interpreter(model_path=path)
        except ImportError:
            continue
    sys.exit("No TFLite runtime found. Run:  pip install ai-edge-litert")


class ASLModel:
    def __init__(self):
        if not os.path.exists(ASL_MODEL):
            sys.exit(f"Missing {ASL_MODEL}. Keep asl_model.tflite next to this script.")
        self.run = load_tflite(ASL_MODEL).get_signature_runner("serving_default")
        idx = json.load(open(ASL_LABELS))
        self.labels = [w for w, _ in sorted(idx.items(), key=lambda kv: kv[1])]

    def predict(self, seq):
        """seq: (T, 543, 3) with NaN for missing points -> probabilities (250,)"""
        return np.asarray(self.run(inputs=seq.astype(np.float32))["outputs"]).reshape(-1)


# --------------------------------------------------------------------------- #
# Your own signs: features + small classifier
# --------------------------------------------------------------------------- #
N_STEPS = 24  # every clip is resampled to this many time steps


def _interp_missing(a, present):
    """Fill frames where a hand was briefly lost by linear interpolation."""
    idx = np.where(present)[0]
    if len(idx) == 0:
        return np.zeros_like(a), present
    out = a.copy()
    t = np.arange(len(a))
    inside = (t >= idx[0]) & (t <= idx[-1])
    flat = a[idx].reshape(len(idx), -1)
    filled = np.stack([np.interp(t[inside], idx, flat[:, j]) for j in range(flat.shape[1])], 1)
    out[inside] = filled.reshape(-1, *a.shape[1:])
    out[~inside] = 0.0
    return out, inside


def _resample(x, n=N_STEPS):
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    flat = x.reshape(len(x), -1)
    res = np.stack([np.interp(t_new, t_old, flat[:, j]) for j in range(flat.shape[1])], 1)
    return res.reshape(n, *x.shape[1:])


def featurize(seq):
    """(T, 543, 3) -> 1-D feature vector, invariant to where you sit and how big you appear."""
    seq = np.asarray(seq, dtype=np.float32)
    lh, rh, pose = seq[:, LH], seq[:, RH], seq[:, POSE]
    pl, pr = ~np.isnan(lh[:, 0, 0]), ~np.isnan(rh[:, 0, 0])
    keep = pl | pr
    if keep.sum() >= 2:
        lh, rh, pose, pl, pr = lh[keep], rh[keep], pose[keep], pl[keep], pr[keep]

    # body reference: shoulder centre and shoulder width (pose 11, 12)
    sh = pose[:, [11, 12], :2]
    if np.isnan(sh).all():
        center, scale = np.array([0.5, 0.5], np.float32), 0.3
    else:
        center = np.nanmean((sh[:, 0] + sh[:, 1]) / 2, axis=0)
        scale = float(np.nanmean(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))) or 0.3

    feats = []
    for h, p in ((lh, pl), (rh, pr)):
        h, p = _interp_missing(np.nan_to_num(h), p)
        rel = h - h[:, :1]                                        # hand shape, wrist-centred
        size = np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=-1)  # wrist -> middle knuckle
        size = np.where(size > 1e-6, size, 1.0)[:, None, None]
        shape = (rel / size).reshape(len(h), -1)
        pos = (h[:, 0, :2] - center) / scale                       # where the hand is
        pos[~p] = 0.0
        shape[~p] = 0.0
        feats += [shape, pos, p[:, None].astype(np.float32)]
    arms = (np.nan_to_num(pose[:, [13, 14, 15, 16], :2], nan=0.0) - center) / scale
    feats.append(arms.reshape(len(pose), -1))

    f = np.concatenate(feats, axis=1)  # (T, 142)
    f = _resample(f) if len(f) > 1 else np.repeat(f, N_STEPS, 0)
    motion = np.diff(f, axis=0)
    return np.concatenate([f.ravel(), f.mean(0), f.std(0), np.abs(motion).mean(0)]).astype(np.float32)


def augment(seq, rng):
    """Small random changes so the model doesn't overfit to one exact performance."""
    s = seq.copy()
    T = len(s)
    a, b = int(rng.uniform(0, 0.1) * T), int(rng.uniform(0, 0.1) * T)  # trim start/end
    s = s[a: T - b] if T - a - b >= 4 else s
    ang = np.deg2rad(rng.uniform(-8, 8))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    c = np.array([0.5, 0.5], np.float32)
    xy = (s[..., :2] - c) @ rot.T * rng.uniform(0.9, 1.1) + c + rng.uniform(-0.03, 0.03, 2)
    s[..., :2] = xy
    s += rng.normal(0, 0.002, s.shape).astype(np.float32)
    return s


def load_recordings():
    X, y = [], []
    if not os.path.isdir(MY_DATA):
        return X, y
    for word in sorted(os.listdir(MY_DATA)):
        d = os.path.join(MY_DATA, word)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".npy"):
                X.append(np.load(os.path.join(d, f)))
                y.append(word)
    return X, y


def _fit(seqs, labels, n_aug, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    F, Y = [], []
    for s, l in zip(seqs, labels):
        F.append(featurize(s)); Y.append(l)
        for _ in range(n_aug):
            F.append(featurize(augment(s, rng))); Y.append(l)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    clf.fit(np.stack(F), Y)
    return clf


def train_custom(n_aug=5):
    from sklearn.model_selection import StratifiedKFold

    seqs, labels = load_recordings()
    words = sorted(set(labels))
    if len(words) < 2:
        sys.exit("Record at least 2 different words first:  python sign_live.py record word1 word2")
    counts = {w: labels.count(w) for w in words}
    print("Samples per word:", counts)
    k = min(5, min(counts.values()))
    if k >= 2:
        print(f"Checking accuracy with {k}-fold cross-validation (on your own held-out samples)...")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        correct, confusions = 0, {}
        for tr, te in skf.split(np.zeros(len(labels)), labels):
            clf = _fit([seqs[i] for i in tr], [labels[i] for i in tr], n_aug)
            pred = clf.predict(np.stack([featurize(seqs[i]) for i in te]))
            for i, p in zip(te, pred):
                correct += p == labels[i]
                if p != labels[i]:
                    confusions[(labels[i], p)] = confusions.get((labels[i], p), 0) + 1
        print(f"  Estimated accuracy: {correct / len(labels):.1%}")
        for (t, p), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  '{t}' mistaken for '{p}': {n}x")
    else:
        print("(Need >= 2 samples per word to estimate accuracy.)")
    clf = _fit(seqs, labels, n_aug)
    os.makedirs(MY_DIR, exist_ok=True)
    with open(MY_MODEL, "wb") as f:
        pickle.dump({"clf": clf, "words": words, "n_steps": N_STEPS}, f)
    print(f"Saved model -> {MY_MODEL}\nRun:  python sign_live.py custom")


class CustomModel:
    def __init__(self):
        if not os.path.exists(MY_MODEL):
            sys.exit("No custom model yet. Run 'record' then 'train' first.")
        with open(MY_MODEL, "rb") as f:
            m = pickle.load(f)
        self.clf = m["clf"]
        self.labels = list(self.clf.classes_)

    def predict(self, seq):
        return self.clf.predict_proba(featurize(seq)[None])[0]


# --------------------------------------------------------------------------- #
# Video / webcam loop
# --------------------------------------------------------------------------- #
def open_source(args):
    import cv2

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open {'video ' + args.video if args.video else f'camera {args.camera}'}."
                 + ("" if args.video else " Try --camera 1."))
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    import cv2

    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def frames(args, lmk):
    """Yields (display_image, landmarks, hands_visible)."""
    import cv2

    cap = open_source(args)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            inp = cv2.flip(frame, 1) if args.mirror else frame
            lm, hands, res = lmk.process(inp)
            show = None
            if not args.no_display:
                lmk.draw(inp, res)
                show = inp if args.mirror else cv2.flip(inp, 1)  # selfie view
            yield show, lm, hands
    finally:
        cap.release()


def top_k(probs, labels, k=3):
    order = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in order]


def run_recognizer(model, args):
    import cv2

    lmk = Landmarker(args.complexity)
    seg = Segmenter(manual=args.manual)
    last, last_t, fps_t, fps = None, 0.0, time.time(), 0.0
    all_frames = []

    def report(seq):
        nonlocal last, last_t
        res = top_k(model.predict(seq), model.labels)
        last, last_t = res, time.time()
        sure = res[0][1] >= args.threshold
        msg = "  |  ".join(f"{w} {p:.0%}" for w, p in res)
        print(("=> " if sure else "?? ") + msg + f"   ({len(seq)} frames)")

    for img, lm, hands in frames(args, lmk):
        if args.whole:
            all_frames.append(lm)
            continue
        seq = seg.update(lm, hands)
        if seq is not None:
            report(seq)
        if img is None:
            continue
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
        fps_t = now
        state = {"idle": "waiting - raise hands to sign" if not args.manual else "SPACE to start",
                 "rec": f"RECORDING ({len(seg.buf)} frames)",
                 "release": "drop your hands"}[seg.state]
        put(img, state, 30, (0, 0, 255) if seg.state == "rec" else (255, 255, 255))
        put(img, f"{fps:.0f} fps", img.shape[0] - 15, scale=0.5)
        if last and now - last_t < 4:
            w, p = last[0]
            sure = p >= args.threshold
            put(img, (w if sure else f"{w}?") + f"  {p:.0%}", 75,
                (0, 220, 0) if sure else (0, 165, 255), 1.3)
            put(img, "   ".join(f"{w} {p:.0%}" for w, p in last[1:]), 110, scale=0.6)
        cv2.imshow("sign_live  (q = quit)", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            seq = seg.toggle()
            if seq is not None:
                report(seq)

    if args.whole and all_frames:
        seq = np.stack(all_frames)
        has_hand = ~np.isnan(seq[:, LH, 0]).all(1) | ~np.isnan(seq[:, RH, 0]).all(1)
        if has_hand.sum() < 2:
            print("No hands detected in this video.")
        else:
            report(seq)
    else:
        seq = seg.flush()
        if seq is not None:
            report(seq)
    cv2.destroyAllWindows()


def run_record(words, args):
    import cv2

    if args.no_display:
        sys.exit("record needs the camera window.")
    for w in words:
        if not w.isascii():
            print(f"Note: '{w}' isn't plain ASCII - it will save fine but the window can't draw it. "
                  "Consider a Latin name, e.g. 'pani' for পানি.")
    lmk = Landmarker(args.complexity)
    seg = Segmenter(manual=args.manual)
    wi = 0
    saved = []
    msg, msg_t = "", 0.0
    gen = frames(args, lmk)
    while wi < len(words):
        word = words[wi]
        d = os.path.join(MY_DATA, word)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".npy")])
        if have >= args.samples:
            print(f"'{word}': {have} samples - done.")
            wi += 1
            continue
        try:
            img, lm, hands = next(gen)
        except StopIteration:
            break
        seq = seg.update(lm, hands)

        def save(seq):
            nonlocal msg, msg_t
            path = os.path.join(d, f"{int(time.time() * 1000)}.npy")
            np.save(path, seq.astype(np.float32))
            saved.append(path)
            msg, msg_t = f"saved {word} #{have + 1} ({len(seq)} frames)", time.time()
            print(msg)

        if seq is not None:
            save(seq)
        put(img, f"Sign: {word}   [{have}/{args.samples}]   word {wi + 1}/{len(words)}", 30,
            (0, 255, 255), 0.8)
        state = {"idle": "raise hands & sign, then drop hands" if not args.manual else "SPACE to start",
                 "rec": f"RECORDING ({len(seg.buf)} frames)", "release": "drop your hands"}[seg.state]
        put(img, state, 65, (0, 0, 255) if seg.state == "rec" else (255, 255, 255))
        if time.time() - msg_t < 2:
            put(img, msg, 100, (0, 220, 0))
        put(img, "u = undo last   s = skip word   q = quit", img.shape[0] - 15, scale=0.5)
        cv2.imshow("sign_live record", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            wi += 1
        if key == ord("u") and saved:
            os.remove(saved.pop())
            msg, msg_t = "removed last sample", time.time()
            print(msg)
        if key == ord(" "):
            seq = seg.toggle()
            if seq is not None:
                save(seq)
    cv2.destroyAllWindows()
    print("Next:  python sign_live.py train")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["asl", "words", "record", "train", "custom"])
    ap.add_argument("words", nargs="*", help="words to record (record mode)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="video file instead of webcam")
    ap.add_argument("--whole", action="store_true", help="with --video: whole video = one sign")
    ap.add_argument("--manual", action="store_true", help="SPACE starts/stops a sign")
    ap.add_argument("--mirror", action="store_true", help="mirror the image before recognition")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--threshold", type=float, default=0.4, help="below this, show 'not sure'")
    ap.add_argument("--samples", type=int, default=15, help="recordings per word (record mode)")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    if args.mode == "words":
        labels = [w for w, _ in sorted(json.load(open(ASL_LABELS)).items(), key=lambda kv: kv[1])]
        print(f"{len(labels)} ASL words:\n" + ", ".join(labels))
    elif args.mode == "asl":
        run_recognizer(ASLModel(), args)
    elif args.mode == "record":
        if not args.words:
            sys.exit("Give the words to record, e.g.:  python sign_live.py record ami pani dhonnobad")
        run_record(args.words, args)
    elif args.mode == "train":
        train_custom()
    elif args.mode == "custom":
        run_recognizer(CustomModel(), args)


if __name__ == "__main__":
    main()
