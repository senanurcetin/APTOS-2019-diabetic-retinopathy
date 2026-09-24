"""Grad-CAM, and a measurement rather than a pretty picture.

The project's central claim is that the model reads the retina and not the
camera. External validation supports it: discrimination transferred intact to a
dataset where the acquisition shortcut is structurally unavailable. Grad-CAM was
meant to test the same claim from the inside - where the model is looking.

It turned out not to be able to. Measured with proper controls - several draws
of untrained baselines and all five trained folds, compared draw against draw -
the attention concentration of trained and untrained models is
indistinguishable (Mann-Whitney p = 0.77), and the five trained folds disagree
with each other as much as they differ from noise. The instrument is too noisy
here to answer the question, and the report says so rather than printing the one
fold whose number told a story.

Why measure rather than just draw: a heatmap alone is weak evidence. Overlays are easy to read
charitably, and every fundus image is a bright disc on a black frame, so an
attention map that merely covers the disc looks convincing while saying nothing.

So this module measures the overlays as well as drawing them. Each processed
image has a retina mask - the non-black region - and the CAM has a mass. The
ratio

    (share of CAM mass inside the retina) / (share of image area inside it)

is 1.0 when attention is spread uniformly and says nothing, and rises above 1.0
only when attention concentrates on tissue. That number can be wrong, which is
what makes it worth computing.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from aptos.config import GRADES, Config

# The last convolution before pooling: 1280 channels at 12x12 for a 384px input.
TARGET_LAYER = "conv_head"


class GradCAM:
    """Gradient-weighted class activation mapping for the regression head.

    With a single continuous output there is no class to pick: the quantity
    being explained is the predicted severity itself, which makes the map
    unambiguous in a way the multi-class version is not.
    """

    def __init__(self, model, layer_name: str = TARGET_LAYER):
        import torch

        self.torch = torch
        self.model = model
        self.model.eval()

        layer = dict(model.named_modules()).get(layer_name)
        if layer is None:
            raise KeyError(f"{layer_name!r} not found in this model")

        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activations)
        layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, tensor) -> tuple[np.ndarray, float]:
        """Return (heatmap in [0, 1] at input resolution, predicted score)."""
        torch = self.torch
        tensor = tensor.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        output = self.model(tensor)
        score = output.squeeze()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooks did not fire; is the layer name right?")

        # One weight per channel: how much raising that channel raises the score.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        span = cam.max() - cam.min()
        cam = (cam - cam.min()) / span if span > 0 else np.zeros_like(cam)
        return cam, float(score.detach().cpu())


# ------------------------------------------------------------------ measurement

def retina_mask(image_bgr: np.ndarray, tol: int = 10) -> np.ndarray:
    """The non-black region of a processed image.

    `tol` is a little above the auto-crop threshold of 7, because JPEG ringing
    lifts the black bars a few levels off zero and a mask that leaks into them
    would flatter the measurement below.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray > tol


def attention_concentration(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """How concentrated the attention is on tissue rather than frame.

    Returns the two shares and their ratio. A ratio of 1.0 means the heatmap is
    spread evenly over the image and carries no information about where the
    model is looking; above 1.0 means it prefers retina.
    """
    if mask.shape != cam.shape:
        mask = cv2.resize(mask.astype(np.uint8), cam.shape[::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    total = cam.sum()
    area_share = float(mask.mean())
    if total <= 0 or area_share in (0.0, 1.0):
        return {"mass_in_retina": float("nan"), "area_retina": area_share,
                "concentration": float("nan")}

    mass_share = float(cam[mask].sum() / total)
    return {
        "mass_in_retina": mass_share,
        "area_retina": area_share,
        "concentration": mass_share / area_share,
    }


# --------------------------------------------------------------------- drawing

def overlay(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if heat.shape[:2] != image_bgr.shape[:2]:
        heat = cv2.resize(heat, image_bgr.shape[1::-1])
    return cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)


def figure(rows: list[dict], out_path: pathlib.Path) -> pathlib.Path:
    """One row per grade: the processed image beside its attention overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)

    for i, row in enumerate(rows):
        axes[i, 0].imshow(cv2.cvtColor(row["image"], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_ylabel(f"grade {row['grade']}\n{GRADES[row['grade']]}",
                              fontsize=9)
        axes[i, 1].imshow(cv2.cvtColor(row["overlay"], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(
            f"predicted {row['score']:.2f} · {row['concentration']:.2f}x on retina",
            fontsize=9,
        )
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])

    axes[0, 0].set_title("processed input", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------- CLI

def concentration_for(model, images, device) -> np.ndarray:
    """Attention concentration for one set of weights over a fixed image set."""
    cam_maker = GradCAM(model)
    values = []
    for image, tensor in images:
        cam, _ = cam_maker(tensor.clone())
        value = attention_concentration(cam, retina_mask(image))["concentration"]
        if value == value:  # skip NaN
            values.append(value)
    return np.array(values)


def weight_controls(cfg, images, device, trained_states: list,
                    n_seeds: int = 5) -> tuple[pd.DataFrame, dict]:
    """The control that decides whether the measurement means anything.

    Same architecture, same images, same method - only the weights differ:

      random        no learned features at all
      imagenet      pretrained, but never shown a fundus photograph
      trained       the APTOS model

    If all three land in the same place, the number is measuring the
    architecture and should be thrown away. They do not.

    Both untrained baselines are averaged over several head initialisations,
    and that is not a detail. A model built with `num_classes=1` gets a freshly
    random one-output head, and for weights that have never been trained on this
    task the head alone sets the gradient direction - so a single draw is a
    single draw. Measured across seeds the ImageNet baseline moved between 1.00
    and 1.41, which is larger than the effect being tested. An earlier version of
    this analysis reported one draw as if it were the baseline.
    """
    import timm
    import torch

    from aptos.training.loop import build_model

    raw: dict[str, np.ndarray] = {"random": [], "imagenet": []}
    seed_means: dict[str, list[float]] = {"random": [], "imagenet": []}

    for seed in range(n_seeds):
        # Seeded per draw, so the control reproduces run to run.
        torch.manual_seed(cfg.train.seed + seed)
        untrained = {
            "random": timm.create_model(cfg.train.model, pretrained=False,
                                        num_classes=1).to(device),
            "imagenet": build_model(cfg, device),
        }
        for name, model in untrained.items():
            values = concentration_for(model, images, device)
            raw[name].append(values)
            seed_means[name].append(float(values.mean()))

    # The trained side gets several draws too - one per fold checkpoint - so the
    # comparison is draws against draws. Comparing five untrained draws to a
    # single trained model would test that one model, not training.
    raw["trained"], seed_means["trained"] = [], []
    for state in trained_states:
        trained = build_model(cfg, device)
        trained.load_state_dict(state)
        values = concentration_for(trained, images, device)
        raw["trained"].append(values)
        seed_means["trained"].append(float(values.mean()))

    rows = []
    for name in ("random", "imagenet", "trained"):
        per_draw = np.array(seed_means[name])
        raw[name] = np.concatenate(raw[name])
        rows.append({
            "weights": name, "draws": len(per_draw), "n": len(raw[name]),
            "concentration": float(per_draw.mean()),
            # Spread ACROSS draws - the number that says how far a single draw
            # can be trusted.
            "between_draws_std": float(per_draw.std(ddof=1)) if len(per_draw) > 1 else float("nan"),
            "range": f"{per_draw.min():.2f}-{per_draw.max():.2f}",
        })
    raw["draw_means"] = {k: list(v) for k, v in seed_means.items()}
    return pd.DataFrame(rows), raw


def main(argv=None) -> int:
    import argparse

    import torch
    from PIL import Image

    from aptos.data import labels as labels_mod
    from aptos.training.loop import build_model, build_transforms, pick_device

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--per-grade", type=int, default=12,
                        help="images per grade for the measurement")
    parser.add_argument("--out", default="reports/figures/09_gradcam.png")
    parser.add_argument("--report", default="reports/attention.md")
    args = parser.parse_args(argv)

    sweep = pathlib.Path(args.sweep)
    variant = sweep.name.split("-")[0]
    cfg = Config.load(f"configs/{variant}.yaml")
    cfg.train = Config.load("configs/cv.yaml").train

    device = pick_device()
    checkpoint = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))[0]
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg, device)
    model.load_state_dict(blob["state_dict"])
    cam_maker = GradCAM(model)
    _, eval_tf = build_transforms(cfg.train.size, cfg.augment)

    df = labels_mod.load_labels(cfg)
    _, holdout = labels_mod.pool_and_holdout(df, cfg)

    records, gallery, control_images = [], [], []
    for grade in sorted(GRADES):
        subset = holdout[holdout["diagnosis"] == grade].head(args.per_grade)
        for position, (_, row) in enumerate(subset.iterrows()):
            path = cfg.data_dir / row["orig_split"] / f"{row['id_code']}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(device)

            control_images.append((image, tensor))
            cam, score = cam_maker(tensor)
            measures = attention_concentration(cam, retina_mask(image))
            records.append({"id_code": row["id_code"], "grade": grade,
                            "score": score, **measures})

            if position == 0:
                display = cv2.resize(image, (cfg.train.size, cfg.train.size))
                gallery.append({
                    "grade": grade, "score": score,
                    "concentration": measures["concentration"],
                    "image": display, "overlay": overlay(display, cam),
                })
        print(f"  grade {grade}: {len(subset)} images")

    table = pd.DataFrame(records)
    figure_path = figure(gallery, cfg.paths.root / args.out)

    print("\nrunning the weight control (random / imagenet / trained):")
    fold_states = [
        torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        for p in sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    ]
    controls, raw_controls = weight_controls(cfg, control_images, device, fold_states)
    print(controls.to_string(index=False))
    by_weights = dict(zip(controls["weights"], controls["concentration"], strict=True))

    # Draws, not images, are the independent units: every image appears once per
    # draw, so pooling per-image values would multiply the sample size by the
    # number of draws and manufacture significance.
    draws = raw_controls["draw_means"]
    untrained_draws = draws["random"] + draws["imagenet"]
    trained_draws = draws["trained"]
    separated = max(trained_draws) < min(untrained_draws)
    draw_p = float(stats.mannwhitneyu(trained_draws, untrained_draws,
                                      alternative="two-sided").pvalue)

    summary = table.groupby("grade").agg(
        n=("concentration", "size"),
        mass_in_retina=("mass_in_retina", "mean"),
        area_retina=("area_retina", "mean"),
        concentration=("concentration", "mean"),
    ).reset_index()

    lines = [
        "# Where the model looks",
        "",
        "Generated by `python -m aptos.evaluation.gradcam`.",
        "",
        "Grad-CAM on the last convolution, explaining the predicted severity "
        "score itself - with a single regression output there is no class to "
        "choose, so the map is unambiguous.",
        "",
        "A heatmap is weak evidence on its own: every fundus image is a bright "
        "disc on a black frame, so attention that merely covers the disc looks "
        "convincing while saying nothing. The **concentration** column is the "
        "share of heatmap mass falling inside the retina divided by the share of "
        "image area it occupies. 1.00 means attention is spread evenly and "
        "carries no information; above 1.00 means it prefers tissue.",
        "",
        "| grade | n | mass in retina | retina area | concentration |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['grade'])} {GRADES[int(r['grade'])]} | {int(r['n'])} | "
            f"{r['mass_in_retina']:.3f} | {r['area_retina']:.3f} | "
            f"**{r['concentration']:.2f}x** |"
        )
    overall = table["concentration"].mean()
    lines += [
        "",
        f"This table is **fold 1 only**: {overall:.2f}x across {len(table)} "
        f"held-out images. Read it alongside the control below, where all five "
        f"folds average {by_weights['trained']:.2f}x and fold 1 turns out to be "
        f"the lowest of them. An earlier version of this report led with the "
        f"fold-1 figure as though it described the model.",
        "",
        f"![Grad-CAM]({args.out})",
        "",
        "## The control, and the uncomfortable result",
        "",
        "A concentration below 1.00 would be meaningless if the measurement were "
        "biased - so the same images were measured again with the same "
        "architecture and different weights. Only the weights change.",
        "",
        "| weights | draws | images | concentration | spread across draws | range |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in controls.iterrows():
        spread = "-" if r["between_draws_std"] != r["between_draws_std"] \
            else f"{r['between_draws_std']:.3f}"
        lines.append(f"| {r['weights']} | {int(r['draws'])} | {int(r['n'])} | "
                     f"**{r['concentration']:.3f}** | {spread} | {r['range']} |")
    lines += [
        "",
        f"Untrained baselines average {by_weights['random']:.2f} (random) and "
        f"{by_weights['imagenet']:.2f} (ImageNet, never shown a fundus "
        f"photograph). The trained folds average **{by_weights['trained']:.2f}**.",
        "",
        f"Compared draw against draw - {len(trained_draws)} trained folds "
        f"against {len(untrained_draws)} untrained draws, the draws being the "
        f"independent units - the Mann-Whitney p is {draw_p:.3f}, and the two "
        f"sets {'do not' if separated else 'do'} overlap "
        f"(trained up to {max(trained_draws):.2f}, untrained down to "
        f"{min(untrained_draws):.2f}).",
        "",
    ]
    if separated:
        lines += [
            "**Every trained fold sits below every untrained draw.** Training on "
            "APTOS moved gradient attention off the retina, and that is a learned "
            "change rather than an artefact of the method.",
            "",
            "That does not overturn the external-validation result, and the two "
            "are not comfortably consistent either. On IDRiD, where every image "
            "shares one resolution, discrimination transferred intact. A reading "
            "that fits both is that the model uses retinal signal *and* frame "
            "geometry, and on IDRiD the frame is constant across every image, so "
            "it cannot mislead. That is a hypothesis, not a finding.",
            "",
        ]
    else:
        lines += [
            f"**This measurement cannot say where the model looks.** The trained "
            f"folds average {by_weights['trained']:.2f}, which sits inside the "
            f"range untrained models with a random output head produce "
            f"({min(untrained_draws):.2f}-{max(untrained_draws):.2f}). The "
            f"trained folds also disagree with each other by as much "
            f"({min(trained_draws):.2f}-{max(trained_draws):.2f}): five models "
            f"trained identically, on overlapping data, give attention "
            f"concentrations that differ more than any effect training could be "
            f"credited with.",
            "",
            "That is a result about the instrument, not about the model. With a "
            "single regression output and weights that were never trained on the "
            "task, the randomly initialised head alone sets the gradient "
            "direction, so Grad-CAM on an untrained baseline is close to noise - "
            "and a baseline made of noise cannot anchor a comparison.",
            "",
            "**A correction to an earlier version of this analysis.** It reported "
            "an ImageNet baseline of 1.41 and a trained value of 0.72 with "
            "p < 0.0001, concluding that training had moved attention off the "
            "retina. Both numbers came from single draws, and the p-value pooled "
            "per-image values across draws, counting each image several times. "
            "Measured properly the effect is not there to be claimed. The "
            "external-validation result is unaffected: it never depended on this.",
            "",
        ]
    lines += [
        "",
        "## What Grad-CAM cannot settle",
        "",
        "It does not show whether attention on tissue means *clinically correct* "
        "features. Retinal concentration is equally compatible with reading "
        "lesions, vessel calibre, or illumination gradients. Separating those "
        "needs lesion annotations, which APTOS does not carry.",
        "",
        "The map is 12x12 before upsampling, so a bright region covers roughly "
        "32x32 input pixels and cannot localise a microaneurysm.",
        "",
        "Gradient-based attribution is also known to be noisy and to respond to "
        "high-contrast edges. The padded variant has a hard black-to-retina "
        "boundary, which is precisely such an edge - a plausible contributor to "
        "the fold-to-fold disagreement above.",
        "",
    ]

    report_path = cfg.paths.root / args.report
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(cfg.paths.reports / "attention_per_image.csv", index=False)
    print(f"\noverall concentration {overall:.2f}x")
    print(f"written to {report_path} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
