# A second external set, Messidor-2: predictions written before the experiment

**Written 26 September 2026, before any Messidor-2 image has been downloaded or
passed through the model.** As with IDRiD
([`external-validation-prediction.md`](external-validation-prediction.md)), the
point of writing this first is that a prediction made afterwards explains any
outcome. These can be wrong, and each says what would count as wrong.

## Why a second set

The external claim in this repository rests on one dataset: 455 images from a
Kaggle copy of IDRiD, with 129 healthy eyes. At that size, one image moves the
specificity figure by 0.008. A single external result can be luck in either
direction. A second, independent population makes the result either much
harder to dismiss or wrong in an informative way.

It also makes possible an experiment one external set cannot support. The
calibration drift seen on IDRiD was left uncorrected because correcting it needs
labelled data from the target population. With two external populations, one
can serve as the "small labelled sample from a new site" and the other as the
site it is tested on. That turns a documented gap into a measurement.

## Why Messidor-2

- **A different population again**: France, where APTOS is India and IDRiD is
  India from a different clinic.
- **Better labels than either earlier set.** The grades come from
  `google-brain/messidor2-dr-grades` (CC0): each image adjudicated by a panel of
  three retina specialists (Krause et al., *Ophthalmology* 2018). APTOS labels
  disagree with themselves on 29% of duplicate pairs. Against adjudicated
  labels, measured error belongs to the model, not to the reference.
- **Small**: 1748 images, about 400 MB for the image mirror.

What it is *not* is a second structural shortcut test. Messidor-2 was captured
at more than one resolution, and the Kaggle image mirror
(`mariaherrerot/messidor2preprocess`) has already been cropped by a third party.
IDRiD remains the shortcut test; Messidor-2 tests transfer and calibration.

## What is fixed in advance

- **Model**: the baseline 5-fold ensemble, sweep `baseline-6e66147526`, the one
  deployed and used for IDRiD. No fine-tuning.
- **Thresholds**: the APTOS validation thresholds, not refitted.
- **Operating point**: the APTOS out-of-fold cut chosen for sensitivity >= 0.90
  (`reports/calibration.json`), not refitted.
- **Preprocessing**: the baseline pipeline (auto-crop, pad to square, 512px),
  as for IDRiD.
- **Exclusions**: only images the adjudication panel marked ungradable. Nothing
  is excluded after looking at predictions.
- **Primary outcome**: referable DR (grade >= 2), as on IDRiD. The five-way grade
  is reported but not used to judge success.

## The predictions

**P1 - Discrimination transfers.** Referable ROC AUC >= 0.93. APTOS test gave
0.983 and IDRiD 0.984. *Wrong if* AUC < 0.93, which would mean the IDRiD result
did not generalise.

**P2 - The calibration drift recurs, in the same direction.** With the APTOS
operating point, sensitivity falls below the 0.90 it was chosen for, and the
APTOS-fitted Platt calibrator is under-confident (observed referable rate above
the mean predicted probability) with ECE > 0.05. This is what IDRiD showed:
sensitivity 0.816 at the cut, ECE 0.117. *Wrong if* sensitivity stays >= 0.90
or the miscalibration runs the other way, which would make the IDRiD drift look
site-specific rather than a property of the model.

**P3 - The top of the scale compresses again.** Fewer grade-4 predictions than
true grade-4 cases, as on IDRiD (8 predicted against 64). *Wrong if* grade 4 is
over-called or matched.

**P4 - A calibration fitted at one new site transfers to another.** A Platt
calibrator refitted on IDRiD gives lower ECE on Messidor-2 than the APTOS
calibrator does, and the reverse holds too (fitted on Messidor-2, tested on
IDRiD). Refitting the operating point on IDRiD brings Messidor-2 sensitivity
closer to 0.90 than the APTOS cut does. *Wrong if* either direction fails. That
would mean recalibration has to be done per site, not once per "not APTOS",
which is the more expensive answer and worth knowing.

No prediction is made for five-way QWK beyond "below APTOS test's 0.909". With
adjudicated labels and a different grade mix, there is no earlier result to
extrapolate from with any confidence.

## What would change the project's conclusions

- **P1 failing** would weaken the headline claim that the model reads the
  retina. IDRiD would then look like a favourable draw.
- **P2 failing** would narrow the calibration caveat to IDRiD specifically.
- **P4** is the result that decides whether the deployment note in the README
  can say "recalibrate on a small local sample", or has to say "recalibrate at
  every site".
