"""APTOS fundus image preprocessing — shared module.

Every function the three-person work split produced lives here, and the
pipeline (`preprocess`) applies them in order:

    Read -> Quality check -> Auto-crop -> CLAHE -> Square -> Resize

Other scripts and the Colab notebook import from this module rather than
keeping their own copies, so a change here reaches everything at once.
"""
import cv2
import numpy as np

# ImageNet statistics, to match the pretrained backbone weights.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Unusable-image thresholds. These answer "does this image carry any
# information at all?" — a near-black or blown-out white frame — not "is this
# image unusual?". Measured brightness in this dataset spans 15.0-129.6, so in
# practice these never fire. That is a sign the data is clean, not a bug; use
# brightness_outliers() to find unusual-but-usable images.
HARD_DARK = 8.0
HARD_BRIGHT = 250.0


# ------------------------------------------------------------------ auto-crop

def auto_crop(img, tol=7):
    """Remove the black frame surrounding the retina.

    Fundus photographs are rectangles containing a circular field of view; the
    corners are black. Cropping them saves space and, more importantly, keeps
    that dead area out of CLAHE's histogram.

    The tol=7 threshold was validated empirically: cropping at tol=7 and tol=15
    give nearly identical results (11.25% vs 11.48% area removed), meaning the
    retina edge sits well above both and the crop boundary is on a stable
    plateau. tol=2 under-crops, leaving sensor noise inside.

    img: BGR or grayscale numpy array
    tol: pixels above this value count as content
    """
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not mask.any():
        return img

    rows, cols = mask.any(1), mask.any(0)
    cropped = img[np.ix_(rows, cols)]
    # Guard against the pathological case where everything falls below tol.
    return img if 0 in cropped.shape[:2] else cropped


def pad_to_square(img):
    """Pad to a square with black bars, preserving the aspect ratio."""
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    out = np.zeros((size, size, 3) if img.ndim == 3 else (size, size), dtype=img.dtype)
    out[top:top + h, left:left + w] = img
    return out


def to_square(img, size, mode="squash"):
    """Bring an image to the square model input size.

    The geometry of this dataset was measured on 80 samples: the retinal disc is
    clipped by the sensor frame at the top (75% of images) and bottom (52%), and
    never at the sides. So a cropped image is naturally wide (median aspect
    ratio 1.27) and already 86.5% retina.

      pad    : adds black bars. Preserves the aspect ratio, but 28.5% of the
               512px output ends up black — the bars stack on top of the corner
               gaps that a circular field of view already leaves behind.
      squash : resizes directly. Introduces a ~1.27x horizontal compression,
               but it is consistent across images and no tissue is lost. Black
               area drops to 13.4%, giving 21% more effective retina pixels.

    A centre-square crop was also measured: it cuts black to 6.2% but discards
    10.6% of the retina. Peripheral lesions matter, so it was rejected.
    """
    if mode == "pad":
        img = pad_to_square(img)
    elif mode != "squash":
        raise ValueError(f"unknown square_mode: {mode}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------- CLAHE

def apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """Boost local contrast to make vessels and lesions stand out.

    CLAHE is applied to the L (lightness) channel of LAB only. Applying it to
    the three RGB channels separately breaks colour balance and leaves an
    artificial tint on fundus images.

    Call this *after* auto_crop. Run on an uncropped image, the wide black
    border skews the histogram.
    """
    if img.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------- quality control

def image_quality(img, dark_threshold=HARD_DARK, bright_threshold=HARD_BRIGHT):
    """Basic quality measures and whether the image is usable at all.

    Returns: (passed, metrics dict)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean = float(gray.mean())

    metrics = {
        "brightness": mean,
        "std": float(gray.std()),
        "too_dark": mean < dark_threshold,
        "too_bright": mean > bright_threshold,
    }
    metrics["ok"] = not (metrics["too_dark"] or metrics["too_bright"])
    return metrics["ok"], metrics


def brightness_outliers(values, k=3.5):
    """Flag images that deviate from the distribution, using median absolute
    deviation (MAD).

    Why not a fixed threshold: fundus photographs are inherently dark, so a
    general-purpose "too bright" cutoff either flags nothing or flags
    everything. MAD measures against the median and is unaffected by outliers,
    so it adapts to the data's own scale.

    Why not a percentile: a percentile always flags a fixed fraction, even when
    the data is perfectly clean. MAD flags nothing when nothing genuinely
    deviates — so a zero here is informative.

    values: array of brightness measurements
    k: how many MADs count as an outlier (3.5 is a common choice)

    Returns: boolean array, True where the value is an outlier
    """
    v = np.asarray(values, dtype=float)
    median = np.median(v)
    mad = np.median(np.abs(v - median))
    if mad == 0:
        return np.zeros(len(v), dtype=bool)
    # The 0.6745 factor scales MAD to be comparable to a standard deviation
    # under a normal distribution.
    return np.abs(0.6745 * (v - median) / mad) > k


def dhash(img, size=8):
    """Perceptual hash — catches resized or recompressed copies of the same
    image, which a byte-level file hash would miss.

    Note this is a *candidate generator*, not proof: every fundus image is a
    bright disc on black, so different retinas can collide. Verify candidates
    at pixel level before calling them duplicates. Measured on this dataset,
    181 of 312 dHash candidate groups were false positives.

    Returns: 16-character hex string
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return f"{int(''.join('1' if b else '0' for b in bits.flatten()), 2):016x}"


# ------------------------------------------------------------------- pipeline

def preprocess(path, size=512, use_clahe=True, normalize=False,
               clip_limit=2.0, quality_check=True, square_mode="squash"):
    """Full pipeline: read -> quality -> crop -> CLAHE -> square -> normalize.

    path: image file
    size: output edge length
    use_clahe: whether to apply CLAHE (turn off to build a control set)
    normalize: True returns float32 with ImageNet normalisation (model input);
               False returns uint8 BGR (for writing to disk)

    Returns: (image | None, info dict). Unreadable images and images failing
    the quality gate return None, with the reason in info["error"].
    """
    info = {"path": str(path)}

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        info["error"] = "unreadable"
        return None, info

    info["orig_height"], info["orig_width"] = img.shape[:2]

    if quality_check:
        ok, metrics = image_quality(img)
        info.update(metrics)
        if not ok:
            info["error"] = "too dark" if metrics["too_dark"] else "too bright"
            return None, info

    img = auto_crop(img)
    info["crop_height"], info["crop_width"] = img.shape[:2]

    if use_clahe:
        img = apply_clahe(img, clip_limit=clip_limit)

    img = to_square(img, size, mode=square_mode)

    if normalize:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD, info

    return img, info
