"""
Optional equipment counter (AI). Works only if you ran:  pip install ultralytics
If it is not installed, the app lets the trainer DECLARE the counts and attach photos instead.
The free AI knows chair, table, bench, laptop and screen. It does NOT know workbench or lathe.
"""
COCO_IDS = {"chair": 56, "table": 60, "bench": 13, "laptop": 63, "monitor": 62}


def available():
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


_model = None


def count_items(images_bgr, wanted_keys):
    """images_bgr: list of pictures of the room. Returns {key: biggest count seen in any picture}."""
    global _model
    from ultralytics import YOLO
    if _model is None:
        _model = YOLO("yolo11n.pt")          # downloads itself the first time (small file)
    best = {k: 0 for k in wanted_keys}
    for img in images_bgr:
        res = _model.predict(img, conf=0.30, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int).tolist()
        for k in wanted_keys:
            best[k] = max(best[k], cls.count(COCO_IDS[k]))
    return best
