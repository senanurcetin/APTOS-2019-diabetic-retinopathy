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

Run:
    pip install -e ".[train,serve]"
    uvicorn serving.app:app --reload
"""
from __future__ import annotations

import json
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
        "idrid_qwk": 0.8045,
        "idrid_referable_sensitivity": 0.885,
        "idrid_referable_specificity": 0.987,
    },
    "known_limitations": [
        "Not a medical device. No clinical validation, no regulatory clearance.",
        "A metadata-only classifier reaches QWK 0.652 on this dataset without "
        "reading the retina, so acquisition correlates with disease prevalence "
        "in the training data.",
        "Labels carry roughly 29% disagreement between duplicate pairs, putting "
        "a single label's accuracy near 84%. The model cannot exceed its labels.",
        "Grade calibration degrades under distribution shift: on an external "
        "dataset the model under-calls the most severe grade. The referable "
        "decision is the robust output.",
        "Trained on a single population. Performance elsewhere is unknown "
        "except for one external set of 455 images.",
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
            model = build_model(cfg, self.device)
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
            "model_version": self.sweep,
        }


app = FastAPI(title="APTOS retinopathy grader", version="0.2.0")
_grader: Grader | None = None


def get_grader() -> Grader:
    global _grader
    if _grader is None:
        cfg = Config.load("configs/baseline.yaml")
        base = Config.load("configs/cv.yaml")
        cfg.train = base.train
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
    explain: bool = Query(False, description="not implemented yet"),
) -> JSONResponse:
    grader = get_grader()
    started = time.time()
    try:
        result = grader.grade(await file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    result["latency_ms"] = round((time.time() - started) * 1000, 1)
    result["disclaimer"] = (
        "Not a medical device. The referable flag is the output validated "
        "externally; the five-way grade degrades under distribution shift."
    )
    if explain:
        result["explain"] = "Grad-CAM is not implemented yet."
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE.replace("__CARD__", json.dumps(MODEL_CARD["known_limitations"]))


_PAGE = """
<!doctype html><meta charset="utf-8">
<title>APTOS retinopathy grader</title>
<style>
 body{font:16px/1.6 system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem}
 .warn{border-left:3px solid #b45309;background:#fffbeb;padding:.75rem 1rem;margin:1rem 0}
 li{margin:.35rem 0} pre{background:#f6f8fa;padding:1rem;overflow:auto}
</style>
<h1>APTOS retinopathy grader</h1>
<p>Upload a colour fundus photograph. The service returns an ICDRSS grade and a
referral decision.</p>
<div class="warn"><strong>Not a medical device.</strong> This is a benchmark
model trained on one public dataset, with no clinical validation and no
regulatory clearance. Do not use it for care.</div>
<input type="file" id="f" accept="image/*">
<button onclick="go()">Grade</button>
<pre id="out">—</pre>
<h2>What this model is known to get wrong</h2>
<ul id="lim"></ul>
<script>
 document.getElementById('lim').innerHTML =
   __CARD__.map(t => '<li>' + t + '</li>').join('');
 async function go(){
   const f = document.getElementById('f').files[0];
   if(!f){ return; }
   const d = new FormData(); d.append('file', f);
   document.getElementById('out').textContent = 'scoring...';
   const r = await fetch('/predict', {method:'POST', body:d});
   document.getElementById('out').textContent =
     JSON.stringify(await r.json(), null, 2);
 }
</script>
"""
