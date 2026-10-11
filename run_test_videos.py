"""
Runs the system on the sample videos listed in data/test_videos.csv and saves the
results so they appear on the Monitor dashboard (and in the Accuracy tab).
The equipment numbers in that file are SIMULATED sample numbers (marked as 'Test data').
Run:   python run_test_videos.py
"""
import os, sys, time, uuid, json
import pandas as pd
import config, core, store
from engine import FaceEngine, load_enrolled, find_duplicates, read_frames, BASE

store.init_data()
CLIPS = os.path.join(BASE, "clips")
print("Loading face system ...")
engine = FaceEngine()
people, problems = load_enrolled(engine, store.ENROL, store.LEARNED)
for p in problems:
    print("  WARNING:", p)
dups = find_duplicates(engine, people)
removed = store.delete_test_sessions()
if removed:
    print(f"Removed {removed} older test results.")

tv = store.read_csv(os.path.join(store.DATA, "test_videos.csv"))
for _, r in tv.iterrows():
    path = os.path.join(CLIPS, r["clip"].strip())
    if not os.path.exists(path):
        print(f"SKIP: video not found: {path}  (copy it into the 'clips' folder)")
        continue
    print(f"\nChecking {r['clip']} ...")
    rate, _ = store.centre_settings(r["centre"])
    frames = read_frames(path, config.SAMPLES_PER_CLIP)
    an = core.analyse_frames(engine, frames, people)
    claimed = r["claimed_ids"].upper().split()
    hours = float(r["class_hours"] or 3)
    res = core.evaluate(an, people, claimed, hours, rate, trainer_id=None)
    found, methods = {}, {}
    for part in r["declared_items"].split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            found[k.strip()] = int(v)
            methods[k.strip()] = "Test data (simulated)"
    infra_rows, infra_score = core.compare_infra(store.gov_rows(r["centre"]), found, methods)
    s = res["summary"]
    trust, parts = core.trust_score(s["attendance_score"], infra_score, s["camera_status"] == "OK", s["trainer_status"])
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ev = core.draw_evidence(an, people)
    report = {"id": sid, "time": time.strftime("%Y-%m-%d %H:%M"), "centre": r["centre"], "batch": r["batch"],
              "trainer_user": "test", "trainer_id": "", "source_name": r["clip"], "source_kind": "test video",
              "mode": "normal", "video_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
              "summary": s, "people": res["rows"],
              "infra": {"method": "Test data (simulated)", "score": infra_score, "rows": infra_rows},
              "trust": trust, "colour": core.colour_of(trust), "parts": parts,
              "checklist": core.make_checklist(s, res["rows"], infra_rows, dups), "dups": dups}
    kb = store.save_session(report, core.jpeg_bytes(ev, config.FULL_IMAGE_WIDTH, config.FULL_IMAGE_QUALITY),
                            core.jpeg_bytes(ev, config.FULL_IMAGE_WIDTH, config.FULL_IMAGE_QUALITY))
    print(f"  claimed {s['claimed']} | present {s['present']} | partial {s['partial']} | "
          f"needs confirmation {s['needs_confirmation']} | not seen {s['not_seen']} | trust {trust} {report['colour']}")

truth = store.truth_map()
sess = store.load_sessions()
ids = sorted({p for s in sess for p in [x["id"] for x in s["people"] if x["role"] == "Trainee"]})
acc = core.accuracy(sess, truth, ids)
print("\n=== ACCURACY on the sample videos (automatic result, before any trainer confirmation) ===")
for k in ("TP", "FP", "FN", "TN"):
    print(f"  {k}: {acc[k]}")
print("  precision:", None if acc["precision"] is None else round(acc["precision"], 2),
      "| recall:", None if acc["recall"] is None else round(acc["recall"], 2))
print("\nDONE. Now run:  streamlit run app.py   and log in as monitor.")
