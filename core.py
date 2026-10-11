"""
The brain of SkillSentinel. The app (app.py) calls these functions.
You do NOT need to read or change this file.
"""
import os, json, time
import cv2
import numpy as np
import config
from engine import camera_problem


# ------------------------------------------------------------------
# 1. FIND PEOPLE IN PICTURES (faces -> "seat tracks" -> who is it?)
# ------------------------------------------------------------------
def _crop(img, box, size=96):
    x, y, w, h = box
    pad = int(0.25 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(c, (size, size))


def analyse_frames(engine, frames, people):
    """Look at every picture, follow each seated person across the pictures
    (a 'track'), and work out who each track probably is."""
    n = len(frames)
    health, per_frame = [], []
    tracks = []
    for k, fr in enumerate(frames):
        health.append(camera_problem(fr))
        faces = engine.faces_and_features(fr)
        per_frame.append(len(faces))
        used = set()
        for f in faces:
            x, y, w, h = f["box"]
            cx, cy = x + w / 2, y + h / 2
            best, bd = None, 1e9
            for ti, t in enumerate(tracks):
                if ti in used or k in t["_frames"]:
                    continue
                d = max(abs(t["_cx"] - cx), abs(t["_cy"] - cy))
                tol = max(35, 0.7 * max(w, t["_w"]))
                if d < tol and d < bd:
                    best, bd = ti, d
            if best is None:
                tracks.append({"_cx": cx, "_cy": cy, "_w": w, "_frames": [k], "_feats": [f["feat"]],
                               "_boxes": {k: f["box"]}, "_best": (k, f["box"], w * h),
                               "crop": _crop(fr, f["box"])})
                used.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["_cx"], t["_cy"] = 0.8 * t["_cx"] + 0.2 * cx, 0.8 * t["_cy"] + 0.2 * cy
                t["_w"] = 0.8 * t["_w"] + 0.2 * w
                t["_frames"].append(k)
                t["_feats"].append(f["feat"])
                t["_boxes"][k] = f["box"]
                if w * h > t["_best"][2]:
                    t["_best"] = (k, f["box"], w * h)
                    t["crop"] = _crop(fr, f["box"])
                used.add(best)

    min_frames = max(1, int(round(config.MIN_TRACK_SHARE * n))) if n > 1 else 1
    tracks = [t for t in tracks if len(t["_frames"]) >= min_frames]
    ids = list(people)

    for i, t in enumerate(tracks):
        t["id"] = i
        t["frames"] = sorted(t["_frames"])
        sims = {}
        for pid in ids:
            vals = [max(engine.similarity(ft, ref) for ref in people[pid]["feats"]) for ft in t["_feats"]]
            sims[pid] = float(np.mean(vals))
        t["sims"] = sims
        ordered = sorted(sims.items(), key=lambda kv: -kv[1])
        t["candidates"] = ordered[:3]
        if ordered:
            top1 = ordered[0][1]
            top2 = ordered[1][1] if len(ordered) > 1 else 0.0
            t["top"], t["top_score"], t["margin"] = ordered[0][0], top1, top1 - top2
        else:
            t["top"], t["top_score"], t["margin"] = None, 0.0, 0.0
        if t["top"] and t["top_score"] >= config.MATCH_THRESHOLD and t["margin"] >= config.MIN_MARGIN:
            t["kind"], t["person"] = "confident", t["top"]
        elif t["top"] and t["top_score"] >= config.MATCH_THRESHOLD:
            t["kind"], t["person"] = "ambiguous", None
        else:
            t["kind"], t["person"] = "unknown", None
        m = np.mean([f / (np.linalg.norm(f) + 1e-9) for f in t["_feats"][:8]], axis=0)
        t["mean_feat"] = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32).reshape(1, -1)

    # Ambiguous faces: give each one the best STILL-FREE person, one person per face,
    # and mark it "probable" (needs a human to confirm).
    taken = {t["person"] for t in tracks if t["kind"] == "confident"}
    pairs = sorted(((t["sims"][p], t["id"], p) for t in tracks if t["kind"] == "ambiguous"
                    for p in ids if p not in taken), reverse=True)
    done_t, done_p = set(), set()
    for s, tid, p in pairs:
        if tid in done_t or p in done_p or s < config.MATCH_THRESHOLD:
            continue
        tracks[tid]["kind"], tracks[tid]["person"] = "probable", p
        done_t.add(tid)
        done_p.add(p)
    for t in tracks:
        if t["kind"] == "ambiguous":
            t["kind"] = "unknown"
    return {"n": n, "health": health, "tracks": tracks, "frames": frames,
            "peak_faces": max(per_frame) if per_frame else 0, "per_frame_faces": per_frame}


def verify_trainer_selfie(engine, people, trainer_id, img):
    """Trainer takes a selfie at the start. Returns (status, score)."""
    if not trainer_id or trainer_id not in people:
        return "NOT ENROLLED", 0.0
    faces = engine.faces_and_features(img)
    if not faces:
        return "NOT MATCHED", 0.0
    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    score = max(engine.similarity(faces[0]["feat"], ref) for ref in people[trainer_id]["feats"])
    others = [max(engine.similarity(faces[0]["feat"], ref) for ref in people[p]["feats"])
              for p in people if p != trainer_id]
    second = max(others) if others else 0.0
    ok = score >= max(config.MATCH_THRESHOLD, 0.45) and (score - second) >= config.MIN_MARGIN
    return ("PRESENT" if ok else "NOT MATCHED"), float(score)


# ------------------------------------------------------------------
# 2. ATTENDANCE: turn tracks into a per-person table
# ------------------------------------------------------------------
def _status_table(an, claimed_all, confirmations):
    n = max(1, an["n"])
    auto_f, prob_f, confirmed = {}, {}, set()
    unknown = 0
    for t in an["tracks"]:
        c = confirmations.get(t["id"])
        if c == "OTHER":
            unknown += 1
            continue
        if c:
            auto_f.setdefault(c, set()).update(t["frames"])
            confirmed.add(c)
        elif t["kind"] == "confident":
            auto_f.setdefault(t["person"], set()).update(t["frames"])
        elif t["kind"] == "probable":
            prob_f.setdefault(t["person"], set()).update(t["frames"])
        else:
            unknown += 1
    table = {}
    for pid in sorted(set(claimed_all) | set(auto_f) | set(prob_f)):
        fa, fp = auto_f.get(pid, set()), prob_f.get(pid, set())
        ra, rp = len(fa) / n, len(fp) / n
        claimed = pid in claimed_all
        if ra >= config.PRESENT_IF_SEEN_AT_LEAST:
            status = "PRESENT"
        elif ra > 0:
            status = "PARTIAL"
        elif rp > 0:
            status = "NEEDS CONFIRMATION"
        else:
            status = "NOT SEEN"
        if not claimed:
            status = "SEEN BUT NOT ON REGISTER"
        shown = fa | fp
        timeline = "".join("1" if k in shown else "0" for k in range(n))
        by = "Trainer confirmed" if pid in confirmed else ("AI (unsure)" if rp > 0 and ra == 0 else "AI")
        table[pid] = {"status": status, "ratio": round(max(ra, rp) if ra == 0 else ra, 2),
                      "ratio_auto": round(ra, 2), "timeline": timeline, "by": by}
    return table, unknown, len(confirmed)


def evaluate(an, people, register_ids, hours, rate, confirmations=None, trainer_id=None,
             trainer_selfie="NOT CHECKED"):
    """register_ids = the people the TRAINER says were present (the claim)."""
    confirmations = confirmations or {}
    claimed = [p.upper() for p in register_ids]
    claimed_trainees = [p for p in claimed if not p.startswith("TR")]
    trainer_in = trainer_id if (trainer_id and trainer_id in people) else None
    claimed_all = set(claimed_trainees) | ({trainer_in} if trainer_in else set())

    auto_tab, _, _ = _status_table(an, claimed_all, {})
    tab, unknown, n_confirmed = _status_table(an, claimed_all, confirmations)

    rows = []
    for pid, r in tab.items():
        role = "Trainer" if pid.startswith("TR") else "Trainee"
        rows.append({"id": pid, "name": people.get(pid, {}).get("name", "(not enrolled)"), "role": role,
                     "on_register": "Yes" if pid in claimed_trainees else ("Trainer" if pid == trainer_in else "No"),
                     "status": r["status"], "auto_status": auto_tab[pid]["status"],
                     "seen_ratio": r["ratio"], "timeline": r["timeline"], "by": r["by"]})

    n_present = sum(1 for p in claimed_trainees if tab[p]["status"] == "PRESENT")
    n_partial = sum(1 for p in claimed_trainees if tab[p]["status"] == "PARTIAL")
    n_conf = sum(1 for p in claimed_trainees if tab[p]["status"] == "NEEDS CONFIRMATION")
    n_not = sum(1 for p in claimed_trainees if tab[p]["status"] == "NOT SEEN")
    extra = [p for p in tab if p not in claimed_all and not p.startswith("TR")]
    credit = sum(1.0 if tab[p]["status"] == "PRESENT" else (tab[p]["ratio_auto"] if tab[p]["status"] == "PARTIAL" else 0.0)
                 for p in claimed_trainees)
    n_claimed = len(claimed_trainees)
    att_score = round(100 * credit / n_claimed) if n_claimed else 100
    claimed_hours = n_claimed * hours
    verified_hours = credit * hours
    money = max(0.0, (claimed_hours - verified_hours) * rate)

    # trainer
    trainer_status = trainer_selfie
    if trainer_in and trainer_in in tab and tab[trainer_in]["status"] == "PRESENT":
        trainer_status = "PRESENT"

    # camera
    bad = [h for h in an["health"] if h]
    bad_pct = len(bad) / max(1, len(an["health"]))
    camera_ok = bad_pct <= config.BAD_FRAMES_ALLOWED
    camera_text = "OK" if camera_ok else "PROBLEM: " + max(set(bad), key=bad.count)

    summary = {"claimed": n_claimed, "present": n_present, "partial": n_partial,
               "needs_confirmation": n_conf, "not_seen": n_not, "extra_seen": len(extra),
               "unknown_people": unknown, "peak_faces": an["peak_faces"],
               "trainer_status": trainer_status, "camera_status": camera_text,
               "bad_frames_pct": round(100 * bad_pct), "class_hours": hours, "rate_per_hour": rate,
               "claimed_hours": round(claimed_hours, 1), "verified_hours": round(verified_hours, 1),
               "money_at_risk": round(money), "attendance_score": att_score,
               "trainer_confirmed": n_confirmed, "checks": an["n"]}
    return {"rows": rows, "summary": summary, "extra_ids": extra}


def track_labels(an, people, confirmations=None):
    """For drawing the evidence picture: track id -> (text, colour)."""
    confirmations = confirmations or {}
    out = {}
    for t in an["tracks"]:
        c = confirmations.get(t["id"])
        if c == "OTHER":
            out[t["id"]] = ("OTHER", (40, 40, 230))
        elif c:
            out[t["id"]] = (f"{c} {people[c]['name']} (confirmed)", (60, 180, 75))
        elif t["kind"] == "confident":
            out[t["id"]] = (f"{t['person']} {people[t['person']]['name']}", (60, 180, 75))
        elif t["kind"] == "probable":
            out[t["id"]] = (f"{t['person']}? CONFIRM", (0, 165, 255))
        else:
            out[t["id"]] = ("UNKNOWN", (40, 40, 230))
    return out


def draw_evidence(an, people, confirmations=None):
    """Pick the picture with the most faces and draw labelled boxes on it."""
    if not an["frames"]:
        return None
    k = int(np.argmax(an["per_frame_faces"])) if an["per_frame_faces"] else 0
    img = an["frames"][k].copy()
    labels = track_labels(an, people, confirmations)
    for t in an["tracks"]:
        if k in t["_boxes"]:
            x, y, w, h = t["_boxes"][k]
            text, col = labels[t["id"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, text, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def jpeg_bytes(img, width, quality):
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


# ------------------------------------------------------------------
# 3. INFRASTRUCTURE: compare what is seen with the GOVERNMENT list
# ------------------------------------------------------------------
AI_ITEMS = {"chair": "chair", "chairs": "chair", "table": "table", "tables": "table", "desk": "table",
            "desks": "table", "bench": "bench", "benches": "bench", "laptop": "laptop",
            "laptops": "laptop", "computer": "laptop", "computers": "laptop",
            "monitor": "monitor", "monitors": "monitor", "tv": "monitor", "screen": "monitor"}


def ai_key(item):
    return AI_ITEMS.get(str(item).strip().lower())


def compare_infra(gov_rows, found, method):
    """gov_rows: list of {item, approved_qty}. found: {item_name: count}.
    method: one text (e.g. 'AI count') OR a dict {item_name: text}. Returns (rows, score or None)."""
    rows, got, want = [], 0, 0
    found_l = {str(k).strip().lower(): v for k, v in found.items()}
    for g in gov_rows:
        item = str(g["item"]).strip()
        appr = int(float(g["approved_qty"]))
        c = found_l.get(item.lower())
        if c is None:
            rows.append({"item": item, "approved": appr, "found": None, "status": "NOT CHECKED - inspector must verify",
                         "method": "-"})
            continue
        c = int(c)
        status = "OK" if c >= appr else f"SHORT BY {appr - c}"
        m = method.get(item, method.get(item.lower(), "-")) if isinstance(method, dict) else method
        rows.append({"item": item, "approved": appr, "found": c, "status": status, "method": m})
        got += min(c, appr)
        want += appr
    return rows, (round(100 * got / want) if want else None)


# ------------------------------------------------------------------
# 4. TRUST SCORE and the inspector's TO-DO LIST
# ------------------------------------------------------------------
def trust_score(att_score, infra_score, camera_ok, trainer_status):
    parts = {"Attendance": (att_score, config.W_ATTENDANCE)}
    if infra_score is not None:
        parts["Infrastructure"] = (infra_score, config.W_INFRA)
    parts["Camera"] = (100 if camera_ok else 0, config.W_CAMERA)
    if trainer_status == "PRESENT":
        parts["Trainer"] = (100, config.W_TRAINER)
    elif trainer_status == "NOT MATCHED":
        parts["Trainer"] = (0, config.W_TRAINER)
    total_w = sum(w for _, w in parts.values())
    score = sum(v * w for v, w in parts.values()) / total_w
    return round(score), {k: round(v) for k, (v, _) in parts.items()}


def colour_of(score):
    return "GREEN" if score >= config.GREEN_FROM else ("AMBER" if score >= config.AMBER_FROM else "RED")


def make_checklist(summary, rows, infra_rows, duplicates=None):
    todo = []
    names = lambda st, only_register=True: [r["name"] if r["name"] != "(not enrolled)" else r["id"]
                                           for r in rows if r["status"] == st and r["role"] == "Trainee"
                                           and (r["on_register"] == "Yes" or not only_register)]
    ns = names("NOT SEEN")
    if ns:
        todo.append(f"Ask to meet these trainees (marked present, but not seen in class): {', '.join(ns)}.")
    nc = names("NEEDS CONFIRMATION")
    if nc:
        todo.append(f"Check photos of {', '.join(nc)}: the camera saw someone who may be them, but could not be sure.")
    ex = names("SEEN BUT NOT ON REGISTER", only_register=False)
    if ex:
        todo.append(f"These people were in class but not on the register: {', '.join(ex)}. Check the register.")
    if summary["unknown_people"]:
        todo.append(f"{summary['unknown_people']} face(s) in class could not be matched to any enrolled trainee. "
                    "Ask who they are.")
    if summary["trainer_status"] == "NOT MATCHED":
        todo.append("The trainer's selfie did not match the enrolled trainer photo. Verify who took this class.")
    if summary["camera_status"] != "OK":
        todo.append(f"Camera {summary['camera_status'].lower()}. Inspect the camera and its position.")
    short = [r for r in infra_rows if str(r["status"]).startswith("SHORT")]
    if short:
        todo.append("Equipment below the government-approved list: " +
                    ", ".join(f"{r['item']} ({r['found']} of {r['approved']})" for r in short) + ".")
    unchecked = [r["item"] for r in infra_rows if str(r["status"]).startswith("NOT CHECKED")]
    if unchecked:
        todo.append("Items the AI cannot check, verify on site: " + ", ".join(unchecked) + ".")
    if summary["trainer_confirmed"]:
        todo.append(f"The trainer manually confirmed {summary['trainer_confirmed']} face(s). "
                    "Spot-check these against the evidence photo.")
    if duplicates:
        todo.append("Same face enrolled under two IDs: " +
                    "; ".join(f"{d['id_1']} & {d['id_2']}" for d in duplicates) + ".")
    if not todo:
        todo.append("Nothing unusual found. Routine follow-up only.")
    return todo


# ------------------------------------------------------------------
# 5. ACCURACY (needs a truth sheet that says who REALLY was there)
# ------------------------------------------------------------------
def accuracy(sessions, truth, roster_ids, use="auto_status"):
    """sessions: list of session reports. truth: {source_name: set(ids)}.
    A person counts as 'detected' if status is PRESENT or PARTIAL."""
    tp = fp = fn = tn = uncertain = 0
    detail = []
    for s in sessions:
        key = s.get("source_name")
        if key not in truth:
            continue
        real = truth[key]
        st = {p["id"]: p[use] for p in s["people"]}
        s_tp = s_fp = s_fn = s_tn = s_unc = 0
        for pid in roster_ids:
            detected = st.get(pid) in ("PRESENT", "PARTIAL")
            maybe = st.get(pid) in ("NEEDS CONFIRMATION",) or (
                st.get(pid) == "SEEN BUT NOT ON REGISTER" and False)
            is_real = pid in real
            if is_real and detected:
                s_tp += 1
            elif is_real and not detected:
                s_fn += 1
                s_unc += 1 if maybe else 0
            elif (not is_real) and detected:
                s_fp += 1
            else:
                s_tn += 1
        tp, fp, fn, tn, uncertain = tp + s_tp, fp + s_fp, fn + s_fn, tn + s_tn, uncertain + s_unc
        detail.append({"session": key, "really_present": len(real), "found_correctly": s_tp,
                       "missed": s_fn, "wrongly_added": s_fp, "of_missed_were_uncertain": s_unc})
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec,
            "uncertain_among_missed": uncertain, "detail": detail}
