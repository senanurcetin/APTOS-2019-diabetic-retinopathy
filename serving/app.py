"""Inference service for the APTOS grader.

Deliberately narrow. It loads the fold ensemble, runs the same preprocessing the
training cache was built with, and returns a grade plus the referral decision -
which is the part that survived external validation.

Three things this service will not do, for reasons the project measured:

  * It will not present the five-way grade as the headline. On IDRiD the model
    issues 8 grade-4 predictions where 64 exist; the grade compresses under
    distribution shift while the referral decision holds.
  * It will not hide the confound. The response carries the caveats, and so does
    the page, because a demo that omits them contradicts the analysis it is
    demonstrating.
  * It will not claim to be a medical device. It is a benchmark model trained on
    one public dataset.

One known numerical wrinkle, measured rather than assumed. Training reads the
cached 512px JPEGs; this service preprocesses the upload from source. The JPEG
round-trip the cache went through (quality 95) is therefore absent here, so raw
scores differ slightly - on six held-out test images the offline and served
scores differed by up to 0.12 while every predicted grade agreed. It is small,
but it is a real training/serving difference and not worth pretending away.

Two backends serve the same ensemble. With an ONNX export under models/onnx/
the service uses ONNX Runtime and never imports torch - that is the deployed
path, sized for a 512 MB host. Without one it falls back to the PyTorch
checkpoints. APTOS_BACKEND=onnx|torch forces either.

Run:
    pip install -e ".[train,serve]"
    uvicorn serving.app:app --reload
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from aptos.config import GRADES, REFERABLE_FROM, Config
from aptos.modeling.thresholds import apply_thresholds

MODEL_CARD = {
    "model": "EfficientNet-B0, ordinal regression with validation-fitted thresholds",
    "training_data": "APTOS-2019, 3247 images after removing 49 leaked duplicates",
    "input": "colour fundus photograph",
    "reported": {
        "aptos_test_qwk": 0.9091,
        "aptos_referable_sensitivity": 0.956,
        "aptos_referable_specificity": 0.917,
        "aptos_referable_roc_auc": 0.983,
        "idrid_qwk": 0.8045,
        "idrid_referable_sensitivity": 0.885,
        "idrid_referable_specificity": 0.987,
        "idrid_referable_roc_auc": 0.984,
        "messidor2_qwk": 0.4928,
        "messidor2_referable_sensitivity": 0.328,
        "messidor2_referable_specificity": 0.990,
        "messidor2_referable_roc_auc": 0.819,
    },
    # How much the five folds disagree on ordinary images: the standard
    # deviation of the fold scores on each of the 366 APTOS test images, at
    # every 5th percentile (0, 5, ..., 100). Lets a response's fold_spread be
    # read against something measured instead of an invented cut-off.
    "fold_spread_reference": {
        "source": "APTOS test, 366 images, sweep baseline-6e66147526",
        "percentiles_step": 5,
        "values": [0.048, 0.123, 0.14, 0.166, 0.188, 0.207, 0.229, 0.239, 0.254,
                   0.263, 0.272, 0.281, 0.303, 0.317, 0.332, 0.345, 0.379, 0.395,
                   0.431, 0.472, 0.725],
    },
    "known_limitations": [
        "Not a medical device. No clinical validation, no regulatory clearance.",
        "A metadata-only classifier reaches QWK 0.652 on this dataset without "
        "reading the retina, so acquisition correlates with disease prevalence "
        "in the training data.",
        "Labels carry roughly 29% disagreement between duplicate pairs, putting "
        "a single label's accuracy near 84%. The model cannot exceed its labels.",
        "Moderate disease without hard exudates is under-graded. Against "
        "Messidor-2's adjudicated labels, 83% of Moderate eyes were graded below "
        "2 and referral ROC AUC fell to 0.819. Read the referral flag as "
        "'exudate-level disease or worse'.",
        "Calibration drifts between sites and must be set locally: both external "
        "sets are under-confident, and a threshold fitted at one new site did not "
        "transfer to the other.",
        "Trained on a single population and measured on two others: IDRiD (455 "
        "images, India) and Messidor-2 (1744 images, France). Performance "
        "anywhere else is unknown.",
    ],
}


class Grader:
    """The fold ensemble, loaded once."""

    def __init__(self, sweep: pathlib.Path, cfg: Config):
        import torch

        from aptos.training.loop import build_model, build_transforms, pick_device

        self.cfg = cfg
        self.device = pick_device()
        self.torch = torch

        checkpoints = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
        if not checkpoints:
            raise FileNotFoundError(f"no fold checkpoints under {sweep}")

        self.states, thresholds = [], []
        for path in checkpoints:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            self.states.append(blob["state_dict"])
            thresholds.append(blob["thresholds"])
        # Carried over exactly as fitted on APTOS validation - the same
        # thresholds the reported numbers were produced with.
        self.thresholds = np.mean(thresholds, axis=0)

        # One model instance per fold, held in memory. Reloading five state
        # dicts on every request cost ~2s per image; EfficientNet-B0 is about
        # 21 MB, so five of them is a cheap trade for an interactive demo.
        self.models = []
        for state in self.states:
            model = build_model(cfg, self.device, pretrained=False)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

        _, self.transform = build_transforms(cfg.train.size, cfg.augment)
        self.sweep = sweep.name

    def grade(self, image_bytes: bytes) -> dict[str, Any]:
        import cv2
        from PIL import Image

        from aptos.preprocessing import preprocess_array

        # The same pipeline the cache was built with, run on the decoded upload.
        # preprocess() delegates to this, so there is one implementation and a
        # served prediction cannot drift from a trained one.
        array = np.frombuffer(image_bytes, np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("could not decode that file as an image")

        processed, info = preprocess_array(
            decoded,
            size=self.cfg.preprocess.size,
            use_clahe=self.cfg.preprocess.clahe,
            clip_limit=self.cfg.preprocess.clip_limit or 2.0,
            square_mode=self.cfg.preprocess.square_mode,
            normalize=False,
        )
        if processed is None:
            raise ValueError(f"image rejected by the quality gate: {info.get('error')}")

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        raws = []
        with self.torch.no_grad():
            for model in self.models:
                raws.append(float(model(tensor).squeeze().cpu()))

        raw = float(np.mean(raws))
        grade = int(apply_thresholds([raw], self.thresholds)[0])
        return {
            "grade": grade,
            "grade_label": GRADES[grade],
            "referable": grade >= REFERABLE_FROM,
            "raw_score": round(raw, 4),
            "fold_spread": round(float(np.std(raws, ddof=1)), 4),
            "thresholds": [round(float(t), 4) for t in self.thresholds],
            "fold_scores": [round(r, 4) for r in raws],
            "model_version": self.sweep,
            "_processed": processed,
        }


class OnnxGrader:
    """The same ensemble, served through ONNX Runtime without importing torch.

    This is what fits a 512 MB host. It is a second path to the same numbers,
    so both halves of it were measured before it was trusted:

      * the exported graphs against their torch originals - max raw-score
        difference 3.1e-05 on 40 held-out images, zero grade mismatches
        (serving/export_onnx.py, recorded in export.json);
      * the numpy input transform against torchvision's
        Resize -> ToTensor -> Normalize - identical, difference 0.0.

    Preprocessing is not reimplemented at all: it is the same
    `preprocess_array` the training cache was built with.
    """

    def __init__(self, export_dir: pathlib.Path):
        import onnxruntime as ort

        meta = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
        self.size = int(meta["input_size"])
        self.preprocess_settings = meta["preprocess"]
        self.thresholds = np.asarray(meta["thresholds"], dtype=float)
        self.sweep = meta["sweep"]

        options = ort.SessionOptions()
        # One thread per session: the free tier has a fraction of one CPU, and
        # five sessions each spawning a pool would only contend for it.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # The CPU memory arena is what made this not fit. Each session
        # preallocates and holds its own pool; measured with five sessions it
        # cost +318 MB against +100 MB with the arena off, and pushed the whole
        # service to 613 MB on a 512 MB host. Batch-1 inference gains nothing
        # from the arena worth that.
        options.enable_cpu_mem_arena = False
        self.sessions = [
            ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for path in sorted(export_dir.glob("fold*.onnx"), key=lambda p: int(p.stem[4:]))
        ]
        if not self.sessions:
            raise FileNotFoundError(f"no fold*.onnx under {export_dir}")

    def _to_input(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        from aptos.preprocessing import IMAGENET_MEAN, IMAGENET_STD

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Matches torchvision's Resize((size, size)) on a PIL image exactly.
        resized = Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(resized, dtype=np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def grade(self, image_bytes: bytes) -> dict[str, Any]:
        import cv2

        from aptos.preprocessing import preprocess_array

        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("could not decode that file as an image")

        s = self.preprocess_settings
        processed, info = preprocess_array(
            decoded, size=s["size"], use_clahe=s["clahe"],
            clip_limit=s["clip_limit"] or 2.0, square_mode=s["square_mode"],
            normalize=False,
        )
        if processed is None:
            raise ValueError(f"image rejected by the quality gate: {info.get('error')}")

        x = self._to_input(processed)
        raws = [float(sess.run(None, {"image": x})[0].squeeze()) for sess in self.sessions]
        raw = float(np.mean(raws))
        grade = int(apply_thresholds([raw], self.thresholds)[0])
        return {
            "grade": grade,
            "grade_label": GRADES[grade],
            "referable": grade >= REFERABLE_FROM,
            "raw_score": round(raw, 4),
            "fold_spread": round(float(np.std(raws, ddof=1)), 4),
            "thresholds": [round(float(t), 4) for t in self.thresholds],
            "fold_scores": [round(r, 4) for r in raws],
            "model_version": self.sweep,
            "backend": "onnx",
            "_processed": processed,
        }


app = FastAPI(title="APTOS retinopathy grader", version="0.2.0")
_grader: Grader | OnnxGrader | None = None


def get_grader() -> Grader | OnnxGrader:
    """ONNX if an export is present, torch otherwise; APTOS_BACKEND overrides."""
    global _grader
    if _grader is None:
        cfg = Config.load("configs/baseline.yaml")
        backend = os.environ.get("APTOS_BACKEND", "auto")

        exports = sorted((cfg.paths.models / "onnx").glob("baseline-*"))
        if backend == "onnx" or (backend == "auto" and exports):
            if not exports:
                raise HTTPException(503, "no ONNX export found under models/onnx/")
            _grader = OnnxGrader(exports[-1])
        else:
            cfg.train = Config.load("configs/cv.yaml").train
            sweeps = sorted((cfg.paths.models / "cv").glob("baseline-*"))
            if not sweeps:
                raise HTTPException(503, "no trained sweep found under models/cv/")
            _grader = Grader(sweeps[-1], cfg)
    return _grader


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _grader is not None}


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    return MODEL_CARD


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="deliberately not offered - see reports/attention.md"),
    preview: bool = Query(False, description="return the preprocessed image the model scored"),
) -> JSONResponse:
    grader = get_grader()
    started = time.time()
    try:
        result = grader.grade(await file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    result["latency_ms"] = round((time.time() - started) * 1000, 1)
    processed = result.pop("_processed")
    if preview:
        result["preview"] = preview_data_url(processed)
    result["disclaimer"] = (
        "Not a medical device. The referable flag held up on one external set "
        "(IDRiD) but under-calls Moderate disease without exudates against "
        "adjudicated labels (Messidor-2); the five-way grade degrades under "
        "distribution shift."
    )
    if explain:
        result["explain"] = (
            "Not offered. Grad-CAM was implemented and measured on this model: "
            "attention concentration varies as much between five identically "
            "trained folds as between trained and untrained weights, so a "
            "heatmap here would look informative without being so. See "
            "reports/attention.md."
        )
    return JSONResponse(result)


def preview_data_url(bgr: np.ndarray, size: int = 320) -> str:
    """The preprocessed image as a small JPEG data URL.

    Shows what the model actually scored - cropped, padded to square - which is
    not the same picture that was uploaded, and is the honest thing to display
    next to a grade.
    """
    import base64

    import cv2

    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


PAGE = pathlib.Path(__file__).with_name("static") / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Numbers and caveats are injected from MODEL_CARD, so the page cannot
    # quote a figure the API does not.
    return (PAGE.read_text(encoding="utf-8")
            .replace("__CARD__", json.dumps(MODEL_CARD["known_limitations"]))
            .replace("__REPORTED__", json.dumps(MODEL_CARD["reported"]))
            .replace("__SPREAD__", json.dumps(MODEL_CARD["fold_spread_reference"])))
