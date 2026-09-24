"""Export the fold ensemble to ONNX, and prove the export changed nothing.

Why ONNX at all: the free Render tier gives 512 MB of RAM. Importing torch alone
takes a large share of that, and five EfficientNet-B0 models on top would not
fit. ONNX Runtime serves the same graphs at a fraction of the footprint.

Why the parity check is not optional: an export is a second implementation of
the model, and this project exists partly to show what happens when two
implementations of "the same thing" quietly disagree. So every fold is compared
against its torch original on held-out images, and the export is refused if the
raw scores or the resulting grades differ beyond tolerance.

    python serving/export_onnx.py --sweep models/cv/baseline-<id>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Tolerance on the raw regression score. Float32 graph rewrites typically land
# around 1e-5; the grades are what finally matter, and they must agree exactly.
MAX_ABS_DIFF = 1e-3


def export(sweep: pathlib.Path, out: pathlib.Path, n_check: int = 40) -> dict:
    import torch
    from PIL import Image

    from aptos.config import Config
    from aptos.data import labels as labels_mod
    from aptos.modeling.thresholds import apply_thresholds
    from aptos.training.loop import build_model, build_transforms

    variant = sweep.name.split("-")[0]
    cfg = Config.load(f"configs/{variant}.yaml")
    cfg.train = Config.load("configs/cv.yaml").train
    size = cfg.train.size

    out.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    if not checkpoints:
        raise FileNotFoundError(f"no fold checkpoints under {sweep}")

    torch_models, thresholds = [], []
    for ckpt in checkpoints:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(cfg, "cpu", pretrained=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        torch_models.append(model)
        thresholds.append(blob["thresholds"])

        dummy = torch.randn(1, 3, size, size)
        target = out / f"{ckpt.stem}.onnx"
        torch.onnx.export(
            model, (dummy,), str(target),
            input_names=["image"], output_names=["score"],
            dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        print(f"  exported {target.name}  {target.stat().st_size / 1e6:.1f} MB")

    # ------------------------------------------------------------ parity check
    import onnxruntime as ort

    sessions = [ort.InferenceSession(str(out / f"{c.stem}.onnx"),
                                     providers=["CPUExecutionProvider"])
                for c in checkpoints]
    _, eval_tf = build_transforms(size, cfg.augment)

    df = labels_mod.load_labels(cfg)
    _, holdout = labels_mod.pool_and_holdout(df, cfg)
    sample = holdout.groupby("diagnosis", group_keys=False).head(max(1, n_check // 5))

    mean_thr = np.mean(thresholds, axis=0)
    worst, grade_mismatch = 0.0, 0
    for _, row in sample.iterrows():
        path = cfg.data_dir / row["orig_split"] / f"{row['id_code']}.jpg"
        tensor = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            t_raw = np.array([float(m(tensor).squeeze()) for m in torch_models])
        o_raw = np.array([float(s.run(None, {"image": tensor.numpy()})[0].squeeze())
                          for s in sessions])
        worst = max(worst, float(np.abs(t_raw - o_raw).max()))
        t_grade = int(apply_thresholds([t_raw.mean()], mean_thr)[0])
        o_grade = int(apply_thresholds([o_raw.mean()], mean_thr)[0])
        grade_mismatch += int(t_grade != o_grade)

    report = {
        "sweep": sweep.name,
        "variant": variant,
        "folds": len(checkpoints),
        "input_size": size,
        "thresholds": [float(t) for t in mean_thr],
        "preprocess": cfg.manifest_fields(),
        "parity": {
            "images_checked": int(len(sample)),
            "max_abs_diff_raw": worst,
            "grade_mismatches": grade_mismatch,
            "tolerance": MAX_ABS_DIFF,
        },
    }
    (out / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  parity on {len(sample)} held-out images: max |torch - onnx| = {worst:.2e}, "
          f"grade mismatches = {grade_mismatch}")
    if worst > MAX_ABS_DIFF or grade_mismatch:
        raise RuntimeError("ONNX export does not reproduce the torch model - refusing it")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sweep = pathlib.Path(args.sweep).resolve()
    out = pathlib.Path(args.out) if args.out else ROOT / "models" / "onnx" / sweep.name
    export(sweep, out)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
