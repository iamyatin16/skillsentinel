"""
SkillSentinel - the app.
Start it with:   streamlit run app.py
"""
import os, time, uuid, tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config, core, store, infra_ai
from engine import FaceEngine, load_enrolled, find_duplicates, read_frames, shrink, parse_enrol_name, BASE

st.set_page_config(page_title="SkillSentinel", page_icon="🛡️", layout="wide")
store.init_data()

COLOUR_ICON = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}


# ------------------------------------------------------------------ helpers
@st.cache_resource
def get_engine():
    return FaceEngine()


@st.cache_resource
def get_people():
    eng = get_engine()
    people, problems = load_enrolled(eng, store.ENROL, store.LEARNED)
    return people, problems, find_duplicates(eng, people)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_upload(uploaded):
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bar(timeline):
    return "".join("█" if c == "1" else "░" for c in timeline)


def people_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["seen in"] = df["seen_ratio"].apply(lambda v: f"{int(round(v * 100))}% of checks")
    df["timeline"] = df["timeline"].apply(bar)
    show = df[["id", "name", "role", "on_register", "status", "seen in", "timeline", "by"]]
    return show.rename(columns={"id": "ID", "name": "Name", "role": "Role", "on_register": "On register?",
                                "status": "Result", "timeline": "Seen over time (█ = seen)", "by": "Decided by"})


STATUS_HELP = (
    "**PRESENT** = seen in most checks. **PARTIAL** = seen only for a part of the class. "
    "**NEEDS CONFIRMATION** = someone was there but the camera could not be sure who. "
    "**NOT SEEN** = marked present on the register but nobody matching was seen. "
    "**SEEN BUT NOT ON REGISTER** = in class but not marked present.")


# ------------------------------------------------------------------ login
def login_page():
    st.title("🛡️ SkillSentinel")
    st.caption("AI check of training-centre attendance and equipment against the government record.")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            go = st.form_submit_button("Login")
        if go:
            user = store.check_login(username, password)
            if user:
                st.session_state["user"] = user
                store.log_event(user["username"], "login")
                st.rerun()
            else:
                st.error("Wrong username or password.")
        st.info("Demo accounts - Trainer: **trainer1** / **trainer@123**  |  "
                "Monitoring officer: **monitor** / **monitor@123**")


# ------------------------------------------------------------------ trainer
def reset_session_state():
    for k in ("an", "confirm", "meta", "infra", "selfie"):
        st.session_state.pop(k, None)


def trainer_page(user):
    centre, tid = user["centre"], user["trainer_id"]
    people, problems, dups = get_people()
    st.title(f"👩‍🏫 Trainer desk - {centre}")
    st.caption(f"Logged in as {user['name']} ({tid}). Do the steps in order: 1 attendance, 2 equipment, 3 submit.")
    for p in problems:
        st.warning(p)

    t1, t2, t3, t4, t5 = st.tabs(["1️⃣ Attendance", "2️⃣ Equipment", "3️⃣ Submit", "➕ Enrol photo", "📜 My sessions"])
    with t1:
        step_attendance(user, people)
    with t2:
        step_equipment(user)
    with t3:
        step_submit(user, people, dups)
    with t4:
        step_enrol(user, people)
    with t5:
        mine = [s for s in store.load_sessions() if s.get("trainer_user") == user["username"]]
        if not mine:
            st.info("You have not submitted any session yet.")
        else:
            st.dataframe(pd.DataFrame([{"time": s["time"], "batch": s["batch"],
                                        "trust": f"{COLOUR_ICON[s['colour']]} {s['trust']}",
                                        "present": s["summary"]["present"], "claimed": s["summary"]["claimed"]}
                                       for s in mine]), hide_index=True, width="stretch")
        pend = store.outbox_ids()
        if pend:
            st.warning(f"{len(pend)} session(s) are saved on this computer and not uploaded yet.")
            if st.button("Upload them now (needs internet)"):
                st.success(f"Uploaded {store.sync_outbox()} session(s).")


def step_attendance(user, people):
    centre, tid = user["centre"], user["trainer_id"]
    rate, default_hours = store.centre_settings(centre)
    batches = store.batches_for(centre)
    if not batches:
        st.error("No batch found for your centre. Add trainees in data/roster.csv.")
        return
    c1, c2 = st.columns(2)
    batch = c1.selectbox("Batch", batches)
    hours = c2.number_input("Class hours today", min_value=0.5, max_value=12.0, value=float(default_hours), step=0.5)

    st.subheader("A. Your selfie (proves the trainer is at the centre)")
    if tid in people:
        selfie_file = st.camera_input("Take a selfie")
    else:
        selfie_file = None
        st.info(f"Your own photo ({tid}) is not enrolled yet, so the selfie check is skipped. "
                "Open the **➕ Enrol photo** tab and add your photo (use ID " + tid + ").")

    st.subheader("B. Register - tick the trainees who are present")
    ids = store.roster_ids(centre, batch)
    reg_df = pd.DataFrame({"Present?": True, "ID": ids,
                           "Name": [people.get(i, {}).get("name", "(no photo yet)") for i in ids]})
    edited = st.data_editor(reg_df, hide_index=True, disabled=["ID", "Name"], key=f"reg_{batch}",
                            width="stretch")
    register_ids = edited[edited["Present?"]]["ID"].tolist()
    st.caption(f"You marked {len(register_ids)} of {len(ids)} trainees as present.")

    st.subheader("C. Show the class to the camera")
    how = st.radio("How?", ["📷 Take a photo now", "🎞️ Upload a class video", "📁 Pick a video from the 'clips' folder"],
                   horizontal=True)
    photo = video = None
    clip_choice = None
    if how.startswith("📷"):
        photo = st.camera_input("Photo of the whole class (all faces visible)", key="class_photo")
    elif how.startswith("🎞️"):
        video = st.file_uploader("Class video (1-2 minutes, faces towards the camera)",
                                 type=["mp4", "mov", "avi", "mkv"])
    else:
        clips = sorted(f for f in os.listdir(store.CLIPS) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
        if clips:
            clip_choice = st.selectbox("Video", clips)
        else:
            st.info("The 'clips' folder is empty.")

    if st.button("🔍 Check attendance with AI", type="primary"):
        if not register_ids:
            st.error("Tick at least one trainee on the register.")
        elif how.startswith("📷") and photo is None:
            st.error("Take a class photo first.")
        elif how.startswith("🎞️") and video is None:
            st.error("Upload a video first.")
        elif how.startswith("📁") and not clip_choice:
            st.error("Pick a video.")
        else:
            with st.spinner("Checking faces... this can take up to a minute for a video."):
                eng = get_engine()
                try:
                    if photo is not None:
                        frames = [shrink(decode_upload(photo))]
                        name, kind, mb = "camera photo", "photo", 0.0
                    elif video is not None:
                        suffix = os.path.splitext(video.name)[1] or ".mp4"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(video.getvalue())
                        frames = read_frames(tmp.name, config.SAMPLES_PER_CLIP)
                        os.remove(tmp.name)
                        name, kind, mb = video.name, "video", len(video.getvalue()) / 1024 / 1024
                    else:
                        path = os.path.join(store.CLIPS, clip_choice)
                        frames = read_frames(path, config.SAMPLES_PER_CLIP)
                        name, kind, mb = clip_choice, "video", os.path.getsize(path) / 1024 / 1024
                    an = core.analyse_frames(eng, frames, people)
                    sstat, sscore = "NOT CHECKED", 0.0
                    if selfie_file is not None:
                        sstat, sscore = core.verify_trainer_selfie(eng, people, tid, decode_upload(selfie_file))
                    elif tid not in people:
                        sstat = "NOT ENROLLED"
                except Exception as e:      # show a friendly message instead of a crash
                    st.error(f"Could not check: {e}")
                    return
            st.session_state["an"] = an
            st.session_state["confirm"] = {}
            st.session_state["infra"] = None
            st.session_state["meta"] = {"batch": batch, "hours": hours, "rate": rate, "register": register_ids,
                                        "source_name": name, "source_kind": kind, "video_mb": round(mb, 1),
                                        "selfie": sstat, "selfie_score": round(sscore, 2), "id": uuid.uuid4().hex[:4]}

    if "an" not in st.session_state:
        return
    an, meta = st.session_state["an"], st.session_state["meta"]
    confirm = st.session_state["confirm"]
    res = core.evaluate(an, people, meta["register"], meta["hours"], meta["rate"], confirm, tid, meta["selfie"])
    s = res["summary"]

    st.divider()
    st.subheader("Result")
    if meta["selfie"] == "PRESENT":
        st.success(f"Trainer selfie matched (score {meta['selfie_score']}).")
    elif meta["selfie"] == "NOT MATCHED":
        st.error("Trainer selfie did NOT match the enrolled trainer photo. This will be flagged.")
    m = st.columns(6)
    m[0].metric("On register", s["claimed"])
    m[1].metric("Present", s["present"])
    m[2].metric("Partial", s["partial"])
    m[3].metric("Needs confirmation", s["needs_confirmation"])
    m[4].metric("Not seen", s["not_seen"])
    m[5].metric("Faces seen together (peak)", s["peak_faces"])
    ev = core.draw_evidence(an, people, confirm)
    if ev is not None:
        st.image(bgr2rgb(ev), caption="Evidence: green = recognised, orange = please confirm, red = unknown")
    st.dataframe(people_table(res["rows"]), hide_index=True, width="stretch")
    st.caption(STATUS_HELP)

    # --- human confirmation for faces the AI is not sure about
    unsure = [t for t in an["tracks"] if t["kind"] in ("probable", "unknown")]
    if unsure:
        st.subheader("Please confirm these faces")
        st.caption("The AI is not sure who these people are. Choose the right name for each, "
                   "or 'someone else'. Your answers are saved and shown to the inspector.")
        cols = st.columns(min(4, len(unsure)))
        picks = {}
        for i, t in enumerate(unsure):
            with cols[i % len(cols)]:
                st.image(bgr2rgb(t["crop"]), caption=f"Face {t['id'] + 1}", width=110)
                opts = ["(not decided)"]
                cand_ids = [p for p, _ in t["candidates"]]
                for p in cand_ids:
                    opts.append(f"{p} {people[p]['name']}")
                opts.append("Someone else / not enrolled")
                default = 0
                if t["kind"] == "probable" and t["person"] in cand_ids:
                    default = 0
                picks[t["id"]] = st.selectbox("Who is this?", opts, index=default,
                                              key=f"pick_{meta['id']}_{t['id']}")
        if st.button("Apply my confirmations"):
            newc = {}
            for tid_, choice in picks.items():
                if choice == "(not decided)":
                    continue
                newc[tid_] = "OTHER" if choice.startswith("Someone") else choice.split()[0]
            st.session_state["confirm"] = newc
            st.rerun()


def step_equipment(user):
    centre = user["centre"]
    gov = store.gov_rows(centre)
    st.subheader("Government-approved list for your centre")
    st.dataframe(pd.DataFrame(gov).rename(columns={"item": "Item", "approved_qty": "Approved quantity"}),
                 hide_index=True, width="stretch")
    if not gov:
        st.warning("No government list found for your centre (data/gov_inventory.csv).")
        return
    if "an" not in st.session_state:
        st.info("First finish step 1 (Attendance). The class pictures are also used to count equipment.")

    ai_ok = infra_ai.available()
    keys = {g["item"]: core.ai_key(g["item"]) for g in gov}
    ai_items = [i for i, k in keys.items() if k]
    imgs = st.file_uploader("Photos of the classroom (equipment must be visible). Optional if you use the AI count.",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    found, methods = {}, {}
    if ai_ok and ai_items:
        st.success("AI equipment counter is installed.")
        use_video = "an" in st.session_state and st.checkbox("Also use the pictures from the class video", value=True)
        if st.button("🤖 Count equipment with AI"):
            pics = [decode_upload(f) for f in imgs]
            if use_video:
                pics += list(st.session_state["an"]["frames"][:6])
            if not pics:
                st.error("Upload a photo or check the class first.")
            else:
                with st.spinner("Counting..."):
                    try:
                        cnt = infra_ai.count_items(pics, sorted({keys[i] for i in ai_items}))
                        st.session_state["ai_counts"] = {i: cnt[keys[i]] for i in ai_items}
                    except Exception as e:
                        st.error(f"AI counter failed: {e}")
        for i, v in st.session_state.get("ai_counts", {}).items():
            found[i], methods[i] = v, "AI count"
    else:
        st.warning("AI equipment counter is not installed on this computer, so you declare the counts yourself "
                   "and attach photos. These items will be marked 'Trainer declared' for the inspector. "
                   "(To install the AI counter: pip install ultralytics)")

    st.markdown("**Declare what you have** (for anything the AI did not count):")
    cols = st.columns(3)
    for n, g in enumerate(gov):
        item = g["item"]
        if item in found:
            continue
        v = cols[n % 3].number_input(f"{item} (approved {g['approved_qty']})", min_value=0, max_value=500,
                                     value=0, step=1, key=f"decl_{item}")
        if v > 0:
            found[item], methods[item] = int(v), "Trainer declared"
    rows, score = core.compare_infra(gov, found, methods)
    st.session_state["infra"] = {"rows": rows, "score": score, "found": found, "methods": methods,
                                 "photos": [f.getvalue() for f in imgs]}
    st.subheader("Match with the government list")
    st.dataframe(pd.DataFrame(rows).rename(columns={"item": "Item", "approved": "Approved", "found": "Found",
                                                    "status": "Result", "method": "How checked"}),
                 hide_index=True, width="stretch")
    if score is not None:
        st.metric("Equipment score", f"{score}%")
    if any(m == "Trainer declared" for m in methods.values()) and not imgs:
        st.warning("You declared counts but attached no photo. Please upload photos as proof.")


def step_submit(user, people, dups):
    if "an" not in st.session_state:
        st.info("Finish step 1 first.")
        return
    an, meta, confirm = st.session_state["an"], st.session_state["meta"], st.session_state["confirm"]
    infra = st.session_state.get("infra") or {"rows": [], "score": None, "methods": {}}
    res = core.evaluate(an, people, meta["register"], meta["hours"], meta["rate"], confirm,
                        user["trainer_id"], meta["selfie"])
    s = res["summary"]
    trust, parts = core.trust_score(s["attendance_score"], infra["score"], s["camera_status"] == "OK",
                                    s["trainer_status"])
    col = core.colour_of(trust)
    st.subheader(f"Trust score: {trust} {COLOUR_ICON[col]} {col}")
    st.caption("Parts: " + ", ".join(f"{k} {v}" for k, v in parts.items()))
    unsure_left = sum(1 for t in an["tracks"] if t["kind"] in ("probable", "unknown") and t["id"] not in confirm)
    if unsure_left:
        st.warning(f"{unsure_left} face(s) are still not confirmed (step 1). You can still submit; "
                   "the inspector will see them.")
    todo = core.make_checklist(s, res["rows"], infra["rows"], None)
    st.markdown("**What the inspector will see:**")
    for t in todo:
        st.write("• " + t)

    mode = st.radio("How should the report be sent?",
                    ["Normal (summary + evidence photo, about 80 KB)",
                     "Low bandwidth (summary + small photo, about 15-20 KB)",
                     "Very low (text summary only, about 3 KB)"], index=1)
    offline = st.checkbox("No internet now - keep it on this computer and upload later")
    learn = st.checkbox("Teach the system the faces I confirmed (helps next time)", value=True)

    if st.button("✅ Submit attendance & equipment report", type="primary"):
        sid = time.strftime("%Y%m%d-%H%M%S-") + meta["id"]
        ev = core.draw_evidence(an, people, confirm)
        full = core.jpeg_bytes(ev, config.FULL_IMAGE_WIDTH, config.FULL_IMAGE_QUALITY) if ev is not None else b""
        if mode.startswith("Normal"):
            send, m = full, "normal"
        elif mode.startswith("Low"):
            send, m = core.jpeg_bytes(ev, config.LITE_IMAGE_WIDTH, config.LITE_IMAGE_QUALITY), "lite"
        else:
            send, m = b"", "text"
        report = {"id": sid, "time": time.strftime("%Y-%m-%d %H:%M"), "centre": user["centre"],
                  "batch": meta["batch"], "trainer_user": user["username"], "trainer_id": user["trainer_id"],
                  "source_name": meta["source_name"], "source_kind": meta["source_kind"], "mode": m,
                  "video_mb": meta["video_mb"], "summary": s, "people": res["rows"],
                  "infra": {"method": ", ".join(sorted(set(infra["methods"].values()))) or "none",
                            "score": infra["score"], "rows": infra["rows"]},
                  "trust": trust, "colour": col, "parts": parts, "checklist": todo,
                  "dups": [], "selfie_score": meta["selfie_score"]}
        kb = store.save_session(report, full, send, upload_now=not offline)
        if learn:
            for tid_, pid in confirm.items():
                if pid != "OTHER":
                    tr = next((t for t in an["tracks"] if t["id"] == tid_), None)
                    if tr is not None:
                        store.save_learned(pid, sid, tid_, tr["mean_feat"])
            get_people.clear()
        store.log_event(user["username"], f"submitted {sid} (confirmed faces: {len(confirm)})")
        where = "saved on this computer (upload later)" if offline else "uploaded"
        saved = ""
        if meta["video_mb"]:
            saved = f" The video was {meta['video_mb']} MB - that is {100 * (1 - kb / 1024 / meta['video_mb']):.2f}% less data."
        st.success(f"Report {where}. Data sent: {kb} KB.{saved}")
        reset_session_state()
        st.balloons()


def step_enrol(user, people):
    st.subheader("Add a trainee (or your own) photo")
    st.caption("One clear photo, face straight to the camera, good light. "
               "Tip: take the photo with the same camera/room used for the class - recognition works better.")
    c1, c2 = st.columns(2)
    pid = c1.text_input("ID (example T009, your trainer ID is " + user["trainer_id"] + ")").strip().upper()
    name = c2.text_input("Name (one word)").strip().replace("_", "").replace(" ", "")
    batches = store.batches_for(user["centre"]) or ["Batch-A"]
    batch = st.selectbox("Batch (for trainees)", batches, key="enrol_batch")
    cam = st.camera_input("Take photo", key="enrol_cam")
    up = st.file_uploader("...or upload a photo", type=["jpg", "jpeg", "png"], key="enrol_up")
    if st.button("Save enrolment"):
        src = cam or up
        if not pid or not name or src is None:
            st.error("Fill the ID, the name and give a photo.")
            return
        img = decode_upload(src)
        eng = get_engine()
        faces = eng.faces_and_features(shrink(img))
        if not faces:
            st.error("No face found. Use a clearer, front-facing photo.")
            return
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        if len(faces) > 1:
            st.warning("More than one face found - the biggest face is used.")
        for other, info in people.items():
            if other == pid:
                continue
            sc = max(eng.similarity(faces[0]["feat"], f) for f in info["feats"])
            if sc >= config.DUPLICATE_THRESHOLD:
                st.error(f"This face is already enrolled as {other} {info['name']} (score {sc:.2f}). "
                         "Not saved - one person must have one ID.")
                return
        cv2.imwrite(os.path.join(store.ENROL, f"{pid}_{name}.jpg"), img)
        if not pid.startswith("TR"):
            store.add_to_roster(user["centre"], batch, pid)
        get_people.clear()
        store.log_event(user["username"], f"enrolled {pid}")
        st.success(f"Saved {pid} {name}.")


# ------------------------------------------------------------------ monitor
def sessions_df(sessions):
    rows = []
    for s in sessions:
        sm = s["summary"]
        rows.append({"status": f"{COLOUR_ICON[s['colour']]} {s['colour']}", "trust": s["trust"], "centre": s["centre"],
                     "batch": s["batch"], "time": s["time"], "source": s["source_name"],
                     "claimed": sm["claimed"], "present": sm["present"], "partial": sm["partial"],
                     "needs confirmation": sm["needs_confirmation"], "not seen": sm["not_seen"],
                     "equipment %": s["infra"]["score"] if s["infra"]["score"] is not None else "-",
                     "trainer": sm["trainer_status"], "camera": sm["camera_status"],
                     "money at risk (Rs)": sm["money_at_risk"], "id": s["id"]})
    return pd.DataFrame(rows)


def monitor_page(user):
    people, problems, dups = get_people()
    sessions = store.load_sessions()
    st.title("🛡️ SkillSentinel - Monitoring dashboard")
    st.caption("Lowest trust first = inspect first. Open the 'How to read this' tab if anything is unclear.")
    tabs = st.tabs(["📊 Overview", "🔎 Look inside a session", "🚨 Alerts", "🏛️ Government record",
                    "🎯 Accuracy", "📶 Bandwidth & sync", "📖 How to read this"])
    pend = store.outbox_ids()
    if pend:
        st.warning(f"{len(pend)} report(s) are waiting on a centre computer (not uploaded yet).")

    with tabs[0]:
        if not sessions:
            st.info("No sessions yet. Run `python run_test_videos.py` or log in as a trainer and submit one.")
        else:
            red = sum(1 for s in sessions if s["colour"] == "RED")
            money = sum(s["summary"]["money_at_risk"] for s in sessions)
            short = sum(1 for s in sessions for r in s["infra"]["rows"] if str(r["status"]).startswith("SHORT"))
            k = st.columns(5)
            k[0].metric("Sessions checked", len(sessions))
            k[1].metric("🔴 Red sessions", red)
            k[2].metric("Equipment shortages", short)
            k[3].metric("Camera problems", sum(1 for s in sessions if s["summary"]["camera_status"] != "OK"))
            k[4].metric("💰 Money at risk (Rs)", f"{money:,}")
            df = sessions_df(sessions).sort_values("trust")
            st.subheader("Inspection priority (lowest trust first)")
            st.dataframe(df.drop(columns=["id"]), hide_index=True, width="stretch")
            st.subheader("Centres")
            cdf = df.groupby("centre").agg(sessions=("trust", "count"), average_trust=("trust", "mean"),
                                           lowest_trust=("trust", "min")).reset_index()
            cdf["average_trust"] = cdf["average_trust"].round()
            st.dataframe(cdf, hide_index=True, width="stretch")
            st.bar_chart(df.set_index("source")["trust"])
            st.subheader("👻 Ghost trainees (on the register 2+ times, never seen)")
            seen, onreg = {}, {}
            for s in sessions:
                for p in s["people"]:
                    if p["role"] != "Trainee":
                        continue
                    key = (s["centre"], p["id"])
                    if p["on_register"] == "Yes":
                        onreg[key] = onreg.get(key, 0) + 1
                        if p["status"] in ("PRESENT", "PARTIAL", "NEEDS CONFIRMATION"):
                            seen[key] = seen.get(key, 0) + 1
            ghosts = [{"centre": c, "id": i, "times on register": n} for (c, i), n in onreg.items()
                      if n >= 2 and seen.get((c, i), 0) == 0]
            if ghosts:
                st.error("These trainees were marked present again and again but never seen:")
                st.dataframe(pd.DataFrame(ghosts), hide_index=True, width="stretch")
            else:
                st.success("No ghost trainees found.")
        st.subheader("🧬 Duplicate enrolment (same face, two IDs)")
        if dups:
            st.error("The same person looks enrolled under two IDs:")
            st.dataframe(pd.DataFrame(dups), hide_index=True, width="stretch")
        else:
            st.success(f"No duplicate faces found (alert level: similarity {config.DUPLICATE_THRESHOLD} or more).")

    with tabs[1]:
        if not sessions:
            st.info("No sessions yet.")
        else:
            labels = {s["id"]: f"{s['time']} | {s['centre']} | {s['source_name']} | {s['colour']} {s['trust']}"
                      for s in sessions}
            sid = st.selectbox("Choose a session", list(labels), format_func=lambda x: labels[x])
            s = next(x for x in sessions if x["id"] == sid)
            sm = s["summary"]
            a, b = st.columns([3, 2])
            with a:
                p = store.evidence_path(sid)
                if p:
                    st.image(p, caption="Evidence picture sent by the centre")
                else:
                    st.info("This centre sent text only (very low bandwidth mode), so there is no picture.")
                st.dataframe(people_table(s["people"]), hide_index=True, width="stretch")
                st.caption(STATUS_HELP)
            with b:
                st.markdown(f"### Trust score: {s['trust']} {COLOUR_ICON[s['colour']]} {s['colour']}")
                st.write("Parts of the score: " + ", ".join(f"{k} {v}" for k, v in s["parts"].items()))
                st.write(f"Trainer check: **{sm['trainer_status']}** | Camera: **{sm['camera_status']}**")
                st.write(f"Claimed hours: **{sm['claimed_hours']}** | Verified hours: **{sm['verified_hours']}**")
                st.write(f"Money at risk: **Rs {sm['money_at_risk']:,}**")
                st.write(f"Data sent to server: **{s.get('upload_kb', '?')} KB**"
                         + (f" (video was {s['video_mb']} MB)" if s.get("video_mb") else ""))
                if sm["trainer_confirmed"]:
                    st.write(f"Faces confirmed by hand by the trainer: **{sm['trainer_confirmed']}**")
                st.markdown("#### 📋 Inspector checklist (made automatically)")
                for t in s["checklist"]:
                    st.write("• " + t)
                st.markdown("#### 🏛️ Equipment vs government list")
                st.dataframe(pd.DataFrame(s["infra"]["rows"]).rename(columns={
                    "item": "Item", "approved": "Approved", "found": "Found", "status": "Result", "method": "How checked"}),
                    hide_index=True, width="stretch") if s["infra"]["rows"] else st.write("No equipment report.")
                st.markdown("#### Your decision (helps the system learn)")
                decision = st.radio("Was this alert correct?", ["Confirmed", "False alarm"], key=f"dec_{sid}")
                if st.button("Save my decision", key=f"save_{sid}"):
                    store.add_feedback(sid, decision, user["username"])
                    st.success("Saved.")

    with tabs[2]:
        items = []
        for s in sessions:
            for t in s["checklist"]:
                if not t.startswith("Nothing unusual"):
                    items.append({"severity": s["colour"], "time": s["time"], "centre": s["centre"],
                                  "session": s["source_name"], "alert": t})
        order = {"RED": 0, "AMBER": 1, "GREEN": 2}
        items.sort(key=lambda r: order[r["severity"]])
        if items:
            st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
        else:
            st.success("No alerts.")

    with tabs[3]:
        st.subheader("Government sanctioned inventory")
        st.dataframe(store.gov_all(), hide_index=True, width="stretch")
        up = st.file_uploader("Replace with a new government list (CSV with columns: centre, item, approved_qty)",
                              type=["csv"])
        if up is not None and st.button("Use this list"):
            try:
                store.replace_gov(pd.read_csv(up, dtype=str))
                st.success("Government list updated.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        st.subheader("Latest check per centre")
        latest = {}
        for s in sessions:
            latest.setdefault(s["centre"], s)
        for c, s in latest.items():
            st.write(f"**{c}** - {s['time']} - equipment score: {s['infra']['score']}")
            if s["infra"]["rows"]:
                st.dataframe(pd.DataFrame(s["infra"]["rows"]), hide_index=True, width="stretch")

    with tabs[4]:
        truth = store.truth_map()
        st.write("Accuracy compares the AI result with the truth sheet (`data/truth.csv`) that says who REALLY "
                 "sat in each class. It uses the **automatic** result, before any trainer confirmation.")
        if not truth:
            st.info("No data/truth.csv found.")
        else:
            ids = sorted({p["id"] for s in sessions for p in s["people"] if p["role"] == "Trainee"})
            acc = core.accuracy(sessions, truth, ids)
            if acc["TP"] + acc["FP"] + acc["FN"] + acc["TN"] == 0:
                st.info("No session matches a clip name in truth.csv yet.")
            else:
                c = st.columns(4)
                c[0].metric("Found correctly (TP)", acc["TP"])
                c[1].metric("Wrongly added (false positive)", acc["FP"])
                c[2].metric("Missed (false negative)", acc["FN"])
                c[3].metric("Correctly absent (TN)", acc["TN"])
                st.write(f"Precision: **{'-' if acc['precision'] is None else round(acc['precision'], 2)}** "
                         f"(when it says present, how often is it right) | "
                         f"Recall: **{'-' if acc['recall'] is None else round(acc['recall'], 2)}** "
                         f"(of the people really there, how many it found).")
                st.write(f"Of the {acc['FN']} missed people, {acc['uncertain_among_missed']} were marked "
                         "'NEEDS CONFIRMATION' (a human can fix those).")
                st.dataframe(pd.DataFrame(acc["detail"]), hide_index=True, width="stretch")

    with tabs[5]:
        if sessions:
            rows = []
            for s in sessions:
                mb = s.get("video_mb") or 0
                kb = s.get("upload_kb", 0)
                rows.append({"session": s["source_name"], "mode": s["mode"], "video (MB)": mb,
                             "sent (KB)": kb,
                             "data saved": f"{100 * (1 - kb / 1024 / mb):.2f}%" if mb else "-"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(f"Reports waiting to upload: **{len(pend)}**")
        if pend and st.button("Sync now"):
            st.success(f"Uploaded {store.sync_outbox()} report(s).")
            st.rerun()

    with tabs[6]:
        st.markdown(HOW_TO_READ)


HOW_TO_READ = """
### How to read this dashboard (very simple)

**Trust score (0-100):** one number per class. 80+ 🟢 looks fine. 50-79 🟠 check soon. Below 50 🔴 inspect first.
It mixes attendance (45%), equipment (25%), camera health (15%) and trainer check (15%).

**Claimed / Present / Partial / Needs confirmation / Not seen:** the trainer ticks who is present (the *claim*).
The AI looks at the class video and compares. *Present* = seen most of the time. *Partial* = seen for only part of the class.
*Needs confirmation* = a face was there, but the AI could not be sure who. *Not seen* = marked present but nobody matched.

**Money at risk:** (hours claimed - hours really verified) x rate per hour. It is an estimate to help prioritise.

**Equipment vs government list:** each item is compared with the sanctioned quantity. *SHORT BY 3* = three fewer than approved.
*Declared* = the trainer typed the number (not AI-verified). *NOT CHECKED* = the AI cannot see that item, so an inspector must.

**Ghost trainees:** marked present in 2 or more classes, never seen even once.
**Duplicate enrolment:** the same face is registered under two IDs.
**Camera problem:** most pictures are too dark or blurry (lens covered or smeared).
**Trainer check:** the trainer's selfie matched (or not) the trainer's enrolled photo.
**Data sent:** only a tiny report (and an optional small picture) goes to the server, not the video.
"""


# ------------------------------------------------------------------ main
def main():
    user = st.session_state.get("user")
    if not user:
        login_page()
        return
    with st.sidebar:
        st.write(f"👤 **{user['name']}**")
        st.write(f"Role: {user['role']}")
        if st.button("Logout"):
            store.log_event(user["username"], "logout")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    if user["role"] == "trainer":
        trainer_page(user)
    else:
        monitor_page(user)


main()
