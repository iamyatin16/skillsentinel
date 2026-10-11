"""Checks that everything needed is installed. Run:  python check_setup.py"""
import importlib, os, sys
print("Python:", sys.version.split()[0])
ok = True
for m in ("cv2", "numpy", "pandas", "streamlit"):
    try:
        importlib.import_module(m); print(f"  OK       {m}")
    except Exception:
        ok = False; print(f"  MISSING  {m}   -> run:  pip install -r requirements.txt")
try:
    import ultralytics; print("  OK       ultralytics (AI equipment counter available)")
except Exception:
    print("  OPTIONAL ultralytics not installed -> trainers will DECLARE equipment counts (works fine)")
here = os.path.dirname(os.path.abspath(__file__))
for f in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"):
    p = os.path.join(here, "models", f)
    good = os.path.exists(p) and os.path.getsize(p) > 100000
    print(("  OK       " if good else "  MISSING  ") + f)
    ok = ok and good
n = len([f for f in os.listdir(os.path.join(here, "enrol")) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
print(f"  Enrolled photos: {n}")
print("\nALL GOOD. Run:  streamlit run app.py" if ok else "\nFix the MISSING items above, then run this again.")
