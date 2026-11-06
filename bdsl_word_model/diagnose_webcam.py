"""
diagnose_webcam.py - runs the live-side checks for sign_to_word_webcam.py
using YOUR model (word_gru.pt + config.json) and YOUR pipeline code.

Put this file next to sign_to_word_webcam.py and run:
    python diagnose_webcam.py W20          # the word you will sign (W-id or Bangla, e.g. মা)

Then, in the window:  SPACE = start, sign the word ONCE, SPACE = stop  (exactly like
HOLD mode). Repeat 5-10 times. Press Q to finish and print the summary.

Each recorded sign is classified 5 ways:
  A  live      exactly what sign_to_word_webcam.py does now
  B  realtime  A, but MediaPipe gets real millisecond timestamps          (check #4)
  C  mirror    A, but the frame is mirrored before MediaPipe              (check #1)
  D  trim      A, cut to the frames where a hand is visible               (check #2)
  E  mirror+trim                                                          (#1 + #2)
and it prints shoulder orientation and missing-shoulder frames            (#1, #5).
Landmarks are saved to diagnose_out/ so they can be analysed later.
"""
import os, sys, time, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sign_to_word_webcam.py")
src = open(SRC, encoding="utf-8").read()
cut = src.index("# ---- main loop ----")
G = {"__file__": SRC, "__name__": "webcam_pipeline"}
exec(compile(src[:cut], SRC, "exec"), G)   # your exact models + feature code, no main loop

import torch
mp, mpvision, mptasks = G["mp"], G["mpvision"], G["mptasks"]
CLASSES, ID2BN = G["CLASSES"], G["ID2BN"]
POSE_N, N_PTS, L_SH, R_SH = G["POSE_N"], G["N_PTS"], G["L_SH"], G["R_SH"]
BN2ID = {v: k for k, v in ID2BN.items()}

if len(sys.argv) < 2:
    sys.exit("Usage: python diagnose_webcam.py <word you will sign, e.g. W20 or মা>")
target = sys.argv[1]
target = target if target in CLASSES else BN2ID.get(target)
if target not in CLASSES:
    sys.exit(f"'{sys.argv[1]}' is not one of the model's words:\n" +
             ", ".join(f"{c}={ID2BN.get(c, c)}" for c in CLASSES))
VIDEO = sys.argv[2] if len(sys.argv) > 2 else None    # optional: test on a video file
print(f"Target word: {target} = {ID2BN.get(target, target)}")


def new_landmarkers():
    """Fresh landmarkers with the same settings as sign_to_word_webcam.py."""
    pose = mpvision.PoseLandmarker.create_from_options(mpvision.PoseLandmarkerOptions(
        base_options=mptasks.BaseOptions(model_asset_path=G["_get"](G["POSE_URL"], "pose_landmarker.task")),
        running_mode=mpvision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))
    hands = mpvision.HandLandmarker.create_from_options(mpvision.HandLandmarkerOptions(
        base_options=mptasks.BaseOptions(model_asset_path=G["_get"](G["HAND_URL"], "hand_landmarker.task")),
        running_mode=mpvision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
    return pose, hands


def extract(frames, times, mirror=False, real_ts=False):
    pose, hands = new_landmarkers()
    out, t0 = [], times[0]
    for i, (f, t) in enumerate(zip(frames, times)):
        if mirror:
            f = cv2.flip(f, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        ts = int((t - t0) * 1000) + 1 if real_ts else i + 1
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out.append(G["res_to_75"](pose.detect_for_video(img, ts), hands.detect_for_video(img, ts)))
    pose.close(); hands.close()
    return np.array(out, np.float32)


def trim(seq):
    hand = ~np.all(seq[:, POSE_N:N_PTS] == 0, axis=(1, 2))
    if hand.sum() < 6:
        return None
    i = np.where(hand)[0]
    return seq[i[0]: i[-1] + 1]


def probs(seq):
    s = G["resample_T"](G["normalize_seq"](seq))
    x = torch.from_numpy(G["to_features"](s)[None]).to(G["DEV"])
    with torch.no_grad():
        return torch.softmax(G["model"](x), 1)[0].cpu().numpy()


def stats(seq):
    sh_ok = ~(np.all(seq[:, L_SH] == 0, 1) | np.all(seq[:, R_SH] == 0, 1))
    orient = float(np.sign(seq[sh_ok, R_SH, 0] - seq[sh_ok, L_SH, 0]).mean()) if sh_ok.any() else float("nan")
    hand = ~np.all(seq[:, POSE_N:N_PTS] == 0, axis=(1, 2))
    return orient, 1 - sh_ok.mean(), hand.mean()


VARIANTS = ["A live", "B realtime", "C mirror", "D trim", "E mirror+trim"]
results = {v: [] for v in VARIANTS}
os.makedirs(os.path.join(HERE, "diagnose_out"), exist_ok=True)


def analyse(frames, times, live_seq, trial):
    seqs = {"A live": live_seq,
            "B realtime": extract(frames, times, real_ts=True),
            "C mirror": extract(frames, times, mirror=True)}
    seqs["D trim"] = trim(live_seq)
    seqs["E mirror+trim"] = trim(seqs["C mirror"])
    fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
    print(f"\n--- trial {trial}: {len(frames)} frames, {fps:.1f} fps")
    for v in VARIANTS:
        s = seqs[v]
        if s is None or len(s) < 6:
            print(f"  {v:14s} too few frames with hands - skipped"); continue
        p = probs(s)
        order = np.argsort(-p)
        rank = int(np.where(order == CLASSES.index(target))[0][0]) + 1
        results[v].append(rank)
        o, miss, hand = stats(s)
        top = ", ".join(f"{CLASSES[i]}({ID2BN.get(CLASSES[i], '')}) {p[i]:.0%}" for i in order[:3])
        print(f"  {v:14s} rank of target={rank:2d} | top3: {top}")
        print(f"  {'':14s} frames={len(s):3d}  orientation={o:+.2f}  no-shoulder={miss:.0%}  hand-visible={hand:.0%}")
    np.savez_compressed(os.path.join(HERE, "diagnose_out", f"trial{trial}_{target}.npz"),
                        times=np.array(times), **{v.split()[0]: s for v, s in seqs.items() if s is not None})


cap = cv2.VideoCapture(VIDEO if VIDEO else 0)
if not cap.isOpened():
    sys.exit("Could not open the webcam/video.")
lpose, lhands = new_landmarkers()          # continuously running, like the live script
ts, recording, trial = 0, False, 0
frames, times, live = [], [], []
print("SPACE = start/stop one sign | Q = finish")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts += 1
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pts = G["res_to_75"](lpose.detect_for_video(img, ts), lhands.detect_for_video(img, ts))
    if VIDEO:                                   # video file = one sign
        frames.append(frame); times.append(now); live.append(pts)
        continue
    if recording:
        frames.append(frame); times.append(now); live.append(pts)
    disp = cv2.flip(frame, 1)
    msg = f"REC {len(frames)}" if recording else f"SPACE to sign '{target}'  (trials: {trial})  Q = done"
    cv2.putText(disp, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (255, 255, 255), 2)
    cv2.imshow("diagnose", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord(" "):
        if not recording:
            frames, times, live, recording = [], [], [], True
        else:
            recording = False
            if len(frames) >= 6:
                trial += 1
                print("analysing ...")
                analyse(frames, times, np.array(live, np.float32), trial)
cap.release(); cv2.destroyAllWindows()
if VIDEO and len(frames) >= 6:
    trial += 1
    analyse(frames, times, np.array(live, np.float32), trial)

print("\n================ SUMMARY (target %s = %s) ================" % (target, ID2BN.get(target, target)))
for v in VARIANTS:
    r = results[v]
    if r:
        print(f"  {v:14s} correct(top-1) {sum(x == 1 for x in r)}/{len(r)}   in top-3 {sum(x <= 3 for x in r)}/{len(r)}   mean rank {np.mean(r):.1f}")
print("Copy everything above and send it back.")
