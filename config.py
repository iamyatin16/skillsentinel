# ============================================================
#  SETTINGS - you can change these numbers if you want
#  (change a number, press Ctrl+S, then run the app again)
# ============================================================

# How many moments we check inside a video. More = more accurate but slower.
SAMPLES_PER_CLIP = 16

# --- Who is this face? -------------------------------------------------
# Lowest score for saying "this looks like that enrolled person".
MATCH_THRESHOLD = 0.40
# The best match must beat the 2nd best by at least this much.
# If it does not, the system says "NEEDS CONFIRMATION" instead of guessing.
MIN_MARGIN = 0.08
# Same face enrolled under two IDs: only warn if the score is VERY high.
# (0.80 = almost certainly the same person.)
DUPLICATE_THRESHOLD = 0.80

# A person counts as PRESENT if we saw them in at least this share of checks.
PRESENT_IF_SEEN_AT_LEAST = 0.6

# A face must be seen in at least this share of checks to count as a real person
# (stops a passer-by or a flicker from counting).
MIN_TRACK_SHARE = 0.15

# Big video frames are shrunk to this width to save time.
MAX_FRAME_WIDTH = 1280

# Camera health: a frame is "bad" if it is too dark or too blurry.
DARK_LIMIT = 25
BLUR_LIMIT = 20
BAD_FRAMES_ALLOWED = 0.30

# Trust score weights (they are re-balanced automatically if a part is missing)
W_ATTENDANCE = 0.45
W_INFRA = 0.25
W_CAMERA = 0.15
W_TRAINER = 0.15

# Colours: trust score at or above GREEN_FROM is green, at or above AMBER_FROM is amber.
GREEN_FROM = 80
AMBER_FROM = 50

# Low-bandwidth upload sizes
LITE_IMAGE_WIDTH = 480      # width of the small evidence picture
LITE_IMAGE_QUALITY = 45     # JPEG quality (lower = smaller)
FULL_IMAGE_WIDTH = 960
FULL_IMAGE_QUALITY = 80
