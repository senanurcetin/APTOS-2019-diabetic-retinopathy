# Fine-tuning on adjudicated labels: predictions written before the experiment

**Written 26 September 2026, before any fine-tuning run.** Third in the series
after [IDRiD](external-validation-prediction.md) and
[Messidor-2](second-external-validation-prediction.md). The design and the
predictions are fixed here, and each prediction says what would count as wrong.

## The question

Messidor-2 showed where the APTOS model fails: 83% of Moderate (grade 2) eyes
are graded below 2, and referable ROC AUC drops to 0.819. Moderate eyes with
hard exudates are caught; Moderate eyes without them are scored like Mild. The
post-hoc reading was that the model learned where APTOS's single graders draw
the Mild/Moderate boundary, and Messidor-2's adjudicating panel draws it lower.

That reading is a hypothesis, not a finding. This experiment tests it: give the
model some adjudicated labels and see whether the Moderate boundary moves.

## The confound this design has to handle

Fine-tuning on Messidor-2 changes two things at once: the labels (adjudicated)
and the images (Messidor-2's cameras). An improvement could come from either.
If it comes from the images, it is ordinary domain adaptation and says nothing
about labels.

So there is a **control**: the same fine-tuning on the same images, but with the
labels replaced by the original model's own grades for those images - labels
that draw the boundary where APTOS draws it. Same images, same domain, APTOS
boundary.

- If the gain comes from the **labels**, the adjudicated arm improves and the
  control does not.
- If it comes from the **images**, both improve.

## Design, fixed in advance

**Split.** Messidor-2's 1744 gradable images are split into a *tune* half and a
*test* half by patient group, never by image, because each patient contributes
both eyes. Groups are conservative: every original-Messidor image from the same
exam date is one group, and every chain of consecutive `IM` numbers is one group
(a patient's two eyes are numbered consecutively). A group may hold more than
one patient; it never splits one. `StratifiedGroupKFold(n_splits=2, shuffle,
random_state=0)` on the referable label assigns the halves. The tune half is
split the same way into train (80%) and validation (20%). The split is written
to `reports/messidor2_split.csv` and not changed after the first run.

**Model.** Each of the five fold models of sweep `baseline-6e66147526` - the
deployed ensemble - is fine-tuned separately. MSE on the ordinal target, AdamW,
learning rate 1e-4, weight decay 1e-4, cosine schedule, 8 epochs, batch 16,
384px, the training augmentation, seed 42. Each fold keeps the epoch with the
best validation QWK, and its thresholds are fitted on the validation split. The
ensemble averages the five scores and the five threshold sets, as the original
does.

**Arms.**

| arm | fine-tuned on | labels |
|---|---|---|
| none | - | the deployed ensemble, unchanged |
| adjudicated | Messidor-2 tune-train | the panel's grades |
| control | Messidor-2 tune-train | the unchanged ensemble's own grades for those images |

**Evaluation sets.** The Messidor-2 test half (never trained or validated on),
IDRiD (all 455), and the APTOS test split (366). Referable ROC AUC is the primary
measure because it does not depend on thresholds.

## The predictions

**F1 - The adjudicated labels fix discrimination.** Referable ROC AUC on the
Messidor-2 test half reaches **>= 0.90** in the adjudicated arm (the unchanged
ensemble scores 0.819 on the full set). *Wrong if* below 0.90.

**F2 - The Moderate boundary moves.** In the adjudicated arm, the share of true
Moderate test-half eyes graded below 2 falls to **<= 50%** (83% on the full set
before fine-tuning). *Wrong if* above 50%.

**F3 - Nothing else is lost.** After adjudicated fine-tuning, referable ROC AUC
stays **>= 0.95 on IDRiD and >= 0.95 on APTOS test** (0.984 and 0.983 before).
*Wrong if* either falls below 0.95. That would mean the fix was bought by
forgetting.

**F4 - It is the labels, not the images.** The control arm does **not** reach
the adjudicated arm's gains: its Messidor-2 test-half AUC stays **< 0.87** and
its Moderate-below-2 share stays **> 70%**. *Wrong if* the control reaches
either, which would make the improvement domain adaptation.

**F5 - The gain is where the hypothesis puts it.** In the adjudicated arm, the
median score of Moderate eyes **without** referable DME rises by more than that
of Moderate eyes **with** it. The hypothesis says the missed cases are the
subtle ones, so they should move most. *Wrong if* the with-DME group rises as
much or more.

## What the outcomes would mean

- F1, F2 and F4 holding together would turn the post-hoc reading into a tested
  result: the model's Messidor-2 failure was the training labels.
- F1 and F2 holding with F4 failing would mean fine-tuning helps but the label
  story is unproven; the camera is at least as good an explanation.
- F1 failing would mean a few hundred adjudicated images are not enough to move
  the boundary, or that the cause is something else, such as resolution.
- F3 failing would mean the fix cannot be deployed as it stands.

## What it cannot show

- Half of Messidor-2 is 872 images, and the test half holds only about 170
  Moderate eyes. Moderate-share estimates will carry several points of noise.
- A fine-tuned model has seen Messidor-2; its Messidor-2 score is no longer an
  external result. The external tests remain IDRiD for the fine-tuned model, and
  the numbers already reported for the unchanged one.

---

## Outcome (26 September 2026, after the run)

Added after the run, below the predictions, which are unchanged from commit
`238587b`. Full tables: `reports/finetune_messidor2.md`.

| | none | adjudicated | control |
|---|---|---|---|
| Messidor-2 test half, referable AUC | 0.830 | **0.925** | 0.834 |
| Messidor-2 test half, Moderate graded below 2 | 78.4% | **45.0%** | 82.5% |
| Messidor-2 test half, five-way QWK | 0.517 | 0.758 | 0.494 |
| IDRiD, referable AUC | 0.984 | 0.960 | 0.981 |
| APTOS test, referable AUC | 0.983 | 0.965 | 0.976 |
| APTOS test, five-way QWK | 0.909 | 0.867 | 0.867 |
| APTOS test, accuracy | 0.803 | 0.686 | 0.776 |

The control arm's labels - the unchanged ensemble's own grades - agree with the
panel on 59.4% of the tune images, so the two arms really were trained on
different boundaries.

**F1 - held.** Referable AUC on the Messidor-2 test half rises from 0.830 to
0.925.

**F2 - held.** The share of Moderate test-half eyes graded below 2 falls from
78.4% to 45.0%.

**F3 - held, with a cost the prediction did not cover.** Referable AUC stays
above 0.95 on IDRiD (0.960) and APTOS test (0.965). But the grade-level numbers
drop: APTOS test accuracy 0.803 to 0.686, QWK 0.909 to 0.867. The fine-tuned
ensemble scores everything higher, and its thresholds were fitted on Messidor-2,
so on APTOS it over-grades. Referral ranking survives; the five-way grade on the
original population does not.

**F4 - held.** The control, trained on the same images with the APTOS boundary,
gains nothing: AUC 0.834 against 0.830, and 82.5% of Moderate eyes still graded
below 2. Domain adaptation to Messidor-2's cameras does not explain the
improvement. The labels do.

**F5 - failed.** The hypothesis put the gain in Moderate eyes *without*
exudates. Their median score rose by 0.84 (0.40 to 1.25), but that of Moderate
eyes *with* exudates rose more, by 1.26 (1.15 to 2.40). Fine-tuning moved the
whole Moderate grade up rather than the subtle cases specifically. The
exudate-free cases now cross the referral line, but because everything moved,
not because the model learned to see what it was missing.

### What it means

- The post-hoc reading of the Messidor-2 failure is now a tested result, in its
  main part: **the failure was the training labels.** Adjudicated labels fix it
  and APTOS-boundary labels on the same images do not.
- Its finer part is not supported. There is no evidence the model learned new
  lesion features; it re-drew the boundary.
- The fine-tuned ensemble is not a drop-in replacement for the deployed one. It
  trades APTOS grade accuracy for Messidor-2 referral, and it has seen
  Messidor-2. The deployed model stays unchanged.
