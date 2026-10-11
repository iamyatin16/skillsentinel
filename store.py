"""
Saving and loading: users, class lists, government equipment list, sessions.
You do NOT need to read or change this file.
"""
import os, json, csv, time, uuid, hashlib, shutil, hmac
import pandas as pd
import config
from engine import BASE, parse_enrol_name

DATA = os.path.join(BASE, "data")
REC = os.path.join(BASE, "records")
CENTRAL = os.path.join(REC, "central")            # what the "server" knows
CENTRAL_SESS = os.path.join(CENTRAL, "sessions")
CENTRAL_EV = os.path.join(CENTRAL, "evidence")
OUTBOX = os.path.join(REC, "outbox")              # waiting to be uploaded
LOCAL_EV = os.path.join(REC, "local", "evidence")  # full pictures stay on the centre computer
LEARNED = os.path.join(DATA, "learned")
ENROL = os.path.join(BASE, "enrol")
CLIPS = os.path.join(BASE, "clips")

USERS_CSV = os.path.join(DATA, "users.csv")
ROSTER_CSV = os.path.join(DATA, "roster.csv")
CENTRES_CSV = os.path.join(DATA, "centres.csv")
GOV_CSV = os.path.join(DATA, "gov_inventory.csv")
TRUTH_CSV = os.path.join(DATA, "truth.csv")
FEEDBACK_CSV = os.path.join(CENTRAL, "feedback.csv")
AUDIT_CSV = os.path.join(REC, "audit_log.csv")


# ---------------------------------------------------------------- setup
def _hash(salt, pw):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()


def init_data():
    for d in (DATA, REC, CENTRAL_SESS, CENTRAL_EV, OUTBOX, LOCAL_EV, LEARNED, ENROL, CLIPS):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(USERS_CSV):
        rows = []
        for user, pw, role, name, centre, tid in [
                ("trainer1", "trainer@123", "trainer", "Trainer Pune", "Centre-Pune", "TR01"),
                ("monitor", "monitor@123", "monitor", "Scheme Monitoring Officer", "", "")]:
            salt = uuid.uuid4().hex[:8]
            rows.append({"username": user, "salt": salt, "password_hash": _hash(salt, pw), "role": role,
                         "name": name, "centre": centre, "trainer_id": tid})
        pd.DataFrame(rows).to_csv(USERS_CSV, index=False)
    if not os.path.exists(CENTRES_CSV):
        pd.DataFrame([{"centre": "Centre-Pune", "rate_per_hour": 100, "default_class_hours": 3}]).to_csv(CENTRES_CSV, index=False)
    if not os.path.exists(ROSTER_CSV):
        ids = []
        for fn in sorted(os.listdir(ENROL)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                pid, _ = parse_enrol_name(fn)
                if not pid.startswith("TR") and pid not in ids:
                    ids.append(pid)
        pd.DataFrame([{"centre": "Centre-Pune", "batch": "Batch-A", "trainee_id": i} for i in ids],
                     columns=["centre", "batch", "trainee_id"]).to_csv(ROSTER_CSV, index=False)
    if not os.path.exists(GOV_CSV):
        pd.DataFrame([
            {"centre": "Centre-Pune", "item": "Chair", "approved_qty": 12},
            {"centre": "Centre-Pune", "item": "Table", "approved_qty": 8},
            {"centre": "Centre-Pune", "item": "Bench", "approved_qty": 4},
            {"centre": "Centre-Pune", "item": "Laptop", "approved_qty": 2},
            {"centre": "Centre-Pune", "item": "Ceiling fan", "approved_qty": 2},
            {"centre": "Centre-Pune", "item": "Whiteboard", "approved_qty": 1},
        ]).to_csv(GOV_CSV, index=False)


# ---------------------------------------------------------------- login
def check_login(username, password):
    df = pd.read_csv(USERS_CSV, dtype=str).fillna("")
    hit = df[df["username"].str.lower() == username.strip().lower()]
    if hit.empty:
        return None
    r = hit.iloc[0]
    if hmac.compare_digest(_hash(r["salt"], password), r["password_hash"]):
        return {k: r[k] for k in ("username", "role", "name", "centre", "trainer_id")}
    return None


def log_event(user, event):
    new = not os.path.exists(AUDIT_CSV)
    with open(AUDIT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "user", "event"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user, event])


# ---------------------------------------------------------------- lists
def read_csv(path, **kw):
    return pd.read_csv(path, dtype=str, **kw).fillna("")


def batches_for(centre):
    df = read_csv(ROSTER_CSV)
    return sorted(df[df["centre"] == centre]["batch"].unique().tolist())


def roster_ids(centre, batch):
    df = read_csv(ROSTER_CSV)
    return df[(df["centre"] == centre) & (df["batch"] == batch)]["trainee_id"].str.upper().tolist()


def add_to_roster(centre, batch, pid):
    df = read_csv(ROSTER_CSV)
    if not ((df["centre"] == centre) & (df["batch"] == batch) & (df["trainee_id"].str.upper() == pid.upper())).any():
        df.loc[len(df)] = [centre, batch, pid.upper()]
        df.to_csv(ROSTER_CSV, index=False)


def centre_settings(centre):
    df = read_csv(CENTRES_CSV)
    hit = df[df["centre"] == centre]
    if hit.empty:
        return 100.0, 3.0
    r = hit.iloc[0]
    try:
        return float(r["rate_per_hour"]), float(r["default_class_hours"])
    except ValueError:
        return 100.0, 3.0


def gov_rows(centre):
    df = read_csv(GOV_CSV)
    df = df[df["centre"] == centre]
    return [{"item": r["item"], "approved_qty": r["approved_qty"]} for _, r in df.iterrows()]


def gov_all():
    return read_csv(GOV_CSV)


def replace_gov(df):
    need = {"centre", "item", "approved_qty"}
    if not need.issubset(set(df.columns)):
        raise ValueError("The file must have these columns: centre, item, approved_qty")
    df = df[["centre", "item", "approved_qty"]].astype(str)
    for q in df["approved_qty"]:
        float(q)
    df.to_csv(GOV_CSV, index=False)


def truth_map():
    if not os.path.exists(TRUTH_CSV):
        return {}
    df = read_csv(TRUTH_CSV)
    out = {}
    for _, r in df.iterrows():
        out[r["clip"].strip()] = set(r["really_present_ids"].upper().split())
    return out


# ---------------------------------------------------------------- sessions
def _write_session_files(folder_json, folder_img, sid, report, img_bytes):
    os.makedirs(folder_json, exist_ok=True)
    with open(os.path.join(folder_json, sid + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, separators=(",", ":"))
    if img_bytes:
        os.makedirs(folder_img, exist_ok=True)
        with open(os.path.join(folder_img, sid + ".jpg"), "wb") as f:
            f.write(img_bytes)


def save_session(report, full_img_bytes, send_img_bytes, upload_now=True):
    """Keep the full picture on this computer. Send only the report (+ optional small picture)."""
    sid = report["id"]
    os.makedirs(LOCAL_EV, exist_ok=True)
    if full_img_bytes:
        with open(os.path.join(LOCAL_EV, sid + ".jpg"), "wb") as f:
            f.write(full_img_bytes)
    report["evidence"] = (sid + ".jpg") if send_img_bytes else ""
    size = len(json.dumps(report, separators=(",", ":")).encode()) + len(send_img_bytes or b"")
    report["upload_kb"] = round(size / 1024, 1)
    if upload_now:
        _write_session_files(CENTRAL_SESS, CENTRAL_EV, sid, report, send_img_bytes)
    else:
        _write_session_files(OUTBOX, OUTBOX, sid, report, send_img_bytes)
    return report["upload_kb"]


def outbox_ids():
    if not os.path.isdir(OUTBOX):
        return []
    return sorted(f[:-5] for f in os.listdir(OUTBOX) if f.endswith(".json"))


def sync_outbox():
    """Upload everything that was saved while offline."""
    n = 0
    os.makedirs(CENTRAL_SESS, exist_ok=True)
    os.makedirs(CENTRAL_EV, exist_ok=True)
    for sid in outbox_ids():
        shutil.move(os.path.join(OUTBOX, sid + ".json"), os.path.join(CENTRAL_SESS, sid + ".json"))
        img = os.path.join(OUTBOX, sid + ".jpg")
        if os.path.exists(img):
            shutil.move(img, os.path.join(CENTRAL_EV, sid + ".jpg"))
        n += 1
    return n


def load_sessions():
    out = []
    if os.path.isdir(CENTRAL_SESS):
        for fn in sorted(os.listdir(CENTRAL_SESS)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(CENTRAL_SESS, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception:
                    pass
    out.sort(key=lambda s: s.get("time", ""), reverse=True)
    return out


def delete_test_sessions():
    n = 0
    for s in load_sessions():
        if s.get("source_kind") == "test video":
            for p in (os.path.join(CENTRAL_SESS, s["id"] + ".json"), os.path.join(CENTRAL_EV, s["id"] + ".jpg"),
                      os.path.join(LOCAL_EV, s["id"] + ".jpg")):
                if os.path.exists(p):
                    os.remove(p)
            n += 1
    return n


def evidence_path(sid):
    for p in (os.path.join(CENTRAL_EV, sid + ".jpg"),):
        if os.path.exists(p):
            return p
    return None


def add_feedback(sid, decision, user):
    new = not os.path.exists(FEEDBACK_CSV)
    os.makedirs(CENTRAL, exist_ok=True)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "session", "decision", "by"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), sid, decision, user])


def load_feedback():
    if not os.path.exists(FEEDBACK_CSV):
        return pd.DataFrame(columns=["time", "session", "decision", "by"])
    return pd.read_csv(FEEDBACK_CSV, dtype=str)


def save_learned(pid, sid, tid, feat):
    import numpy as np
    os.makedirs(LEARNED, exist_ok=True)
    np.save(os.path.join(LEARNED, f"{pid}__{sid}_{tid}.npy"), feat)
