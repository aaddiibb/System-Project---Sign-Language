# sign_live: webcam sign → word

## 1. Install (once)

You need **Python 3.9 – 3.12**. MediaPipe 0.10.14 doesn't support 3.13+. Check with `python --version`.

```bash
cd sign_live
python -m venv venv
# Windows:  venv\Scripts\activate      macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## 2a. Ready-made ASL model (no training)

```bash
python sign_live.py words     # the 250 ASL words it knows
python sign_live.py asl       # open webcam and sign
```

How to sign:
- Sit so your resting hands are **out of the camera view**.
- Raise your hands, sign **one** word, then drop your hands out of view.
- The prediction appears on screen with its top-3 alternatives. Orange "word?" means the model isn't sure.

These are **ASL** signs (toddler vocabulary: dog, cat, apple, mom, …). Look each one up in an ASL dictionary before you try it.

## 2b. Your own signs (BdSL or anything), most accurate for you

```bash
python sign_live.py record ami pani dhonnobad bhalo --samples 15
python sign_live.py train      # takes seconds; prints estimated accuracy
python sign_live.py custom     # live recognition
```

Tips:
- Record each word 15–20 times. Vary your position slightly and record under different lighting.
- In record mode: `u` = undo the last sample, `s` = skip a word, `q` = quit.
- Use Latin names for words (e.g. `pani`). The window can't draw Bangla text.
- You can add more words later. Run `record` again, then `train` again.

## Options

| Option | What it does |
|---|---|
| `--manual` | Press SPACE to start/stop each sign (if your hands are always in view) |
| `--camera 1` | Use a different webcam |
| `--complexity 0` | Faster on slow laptops (downloads a small model the first time; needs internet) |
| `--mirror` | Try this if ASL predictions are consistently wrong |
| `--video clip.mp4 --whole --no-display` | Predict one sign from a video file |
| `--threshold 0.4` | Confidence below which the word is shown with "?" |

## Credits

`asl_model.tflite` is the 6th-place model from the Google Isolated Sign Language Recognition Kaggle competition by Theo Viel & Christof Henkel (MIT licence, see `MODEL_LICENSE_MIT.txt`), trained on 94,477 clips from 21 Deaf signers.
