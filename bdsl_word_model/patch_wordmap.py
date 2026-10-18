"""
Patch BdSLW60 config.json with the W-id -> Bangla word map.
Run ONCE in the folder that holds config.json (next to word_gru.pt), then
re-run sign_to_word_webcam.py — predictions will show Bangla instead of W-ids.

    python patch_wordmap.py

Source: BdSLW60.pdf (the dataset's official 60-word list).
"""
import os, json

WORDMAP = {
    # --- family / people (W1-W20) ---
    "W1": "বাবা", "W2": "আত্মীয়", "W3": "ভাই", "W4": "বোন", "W5": "বউ",
    "W6": "চাচা", "W7": "চাচী", "W8": "দাদা/নানা", "W9": "দাদী/নানী",
    "W10": "দায়িত্ব", "W11": "দেবর", "W12": "দুলাভাই", "W19": "কন্যা", "W20": "মা",
    # --- food (W37-W50) ---
    "W37": "আম", "W38": "আলু", "W39": "আনারস", "W40": "আঙ্গুর", "W41": "আপেল",
    "W42": "বিস্কুট", "W43": "বরই", "W44": "কেক", "W45": "চা", "W46": "চাল",
    "W47": "চিনি", "W48": "চিপস", "W49": "চকলেট", "W50": "ডাল",
    # --- household items (W91-W112) ---
    "W91": "বোতাম", "W92": "টুপি", "W93": "চাদর", "W94": "চিরুনি", "W95": "চশমা",
    "W96": "চুড়ি", "W97": "ক্লিপ", "W98": "ক্রিম", "W99": "তথ্য", "W100": "দেনাদার",
    "W111": "জমজ", "W112": "জুতা",
    # --- household / electronics (W211-W220) ---
    "W211": "টুথপেস্ট", "W212": "টিশার্ট", "W213": "টিউবলাইট", "W214": "টিভি",
    "W215": "এসি", "W216": "অ্যাপার্টমেন্ট", "W217": "অডিও ক্যাসেট", "W218": "আয়না",
    "W219": "বালতি", "W220": "বালু",
    # --- health (W351-W360) ---
    "W351": "এইডস", "W352": "বাত", "W353": "ব্যান্ডেজ", "W354": "ক্যাপসুল",
    "W355": "চিকিৎসা", "W356": "চোখ ওঠা", "W357": "ডেঙ্গু", "W358": "ডক্টর",
    "W359": "দংশন", "W360": "দুর্বল",
}

HERE = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(HERE, "config.json")
cfg = json.load(open(cfg_path, encoding="utf-8"))

classes = cfg.get("classes", [])
cfg["id2bn"] = {c: WORDMAP.get(c, c) for c in classes}

missing = [c for c in classes if c not in WORDMAP]
json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"Patched {cfg_path}")
print(f"  mapped {len(classes) - len(missing)}/{len(classes)} classes to Bangla")
if missing:
    print("  NOT in word list (left as ID):", missing)
print("Now re-run:  python sign_to_word_webcam.py")
