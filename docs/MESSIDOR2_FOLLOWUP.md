# Messidor-2 examination-disjoint follow-up

**25 October 2026. Split preparation completed; new model training not started.**
This is a follow-up on an already inspected cohort, not a fresh external
confirmation. The historical experiment, split and AUC values remain unchanged.

## Fixed allocation

The supplied [pairing transcription](MESSIDOR2_ACQUISITION.md) defines the unit:
one bilateral examination, not an authenticated longitudinal patient. Each pair
gets a stable key from its two sorted image stems. Neither filename proximity nor
exam-date prefixes determine grouping. Missing pair mappings, changed grades,
duplicate IDs and cross-partition examinations are rejected.

The allocation uses seed **42**, the first split of a two-fold stratified group
split, and then the first split of a five-fold allocation within development.
The target is referable disease (`diagnosis >= 2`). Image rows are canonically
sorted before splitting. There is no seed search or model-score optimization.
Input reordering and exchanging left/right columns produce the same allocation.
The software versions and input/code hashes are recorded for reproduction.

| Partition | Images | Examinations | Referable eyes |
|---|---:|---:|---:|
| Training | 698 | 350 | 184 |
| Validation | 174 | 87 | 45 |
| Test | 872 | 437 | 228 |
| Total | 1,744 | 874 | 457 |

**Zero examinations cross partitions**, versus seven in the historical split.
All original gradable eyes and their diagnoses are preserved. This does not
prove longitudinal patient independence or verify the original ADCIS spreadsheet.
The [aggregate plan](../reports/research/messidor2_examination_v2_plan.json)
contains class counts and the local CSV fingerprint, without individual eye IDs.
The CSV remains under ignored `outputs/`.

Of the new test eyes, **339 were in historical training and 87 in historical
validation**: 426 in historical development. Reusing old fine-tuned weights would
therefore expose nearly half this new test split to prior development. The
transition counts are recorded in the aggregate plan; both future arms must
restart from the APTOS-only base, never from either historical fine-tuned arm.

## Scientific limits and evaluation contract

- This fixes one known splitting defect. It does not erase prior inspection of
  the dataset. Historical AUC **0.925 must not be assigned to this split**.
- Validation contains **four grade-3 and two grade-4 eyes**; test contains 42
  grade-3 and 24 grade-4 eyes. Rare-grade recalls and validation-fitted ordinal
  thresholds can be unstable. Report supports and class recalls alongside QWK,
  macro F1, referable sensitivity/specificity and false negatives.
- Any new fine-tuning run must use a separate experiment directory, record the
  starting checkpoint hashes, save both arms' per-eye scores, and forbid reuse
  of checkpoints trained on the old split. A changed baseline is a new experiment,
  not a numerical reproduction of the historical comparison.
- Both arms must start from the same frozen APTOS fold models and use the same
  development images, optimizer settings and seed. Fit thresholds/choose epochs
  using validation only. Preserve unchanged-baseline thresholds as a separate
  operating-point comparison. Never select thresholds or seeds on the test set.
- Cluster uncertainty by **examination**, resampling the same test examinations
  for both models. This does not turn examinations into patients. Multiplicity,
  repeated cohort use and original-table verification remain limitations.
- No new performance claim or clinical-use recommendation follows from a clean
  split alone. The project remains noncommercial research with zero service cost.

The completed public Kaggle narrative v9 still reports the historical experiment
with its overlap correction. It does not claim a completed v2 fine-tuning run.

## Reproduce the allocation

After the [source downloads](MESSIDOR2_ACQUISITION.md), choose new output paths:

```bash
python -m aptos.evaluation.examination_split \
  --pairs data/external/messidor2/messidor2_pairs.csv \
  --labels data/external/messidor2/google_grades/messidor_data.csv \
  --historical-split reports/messidor2_split.csv \
  --seed 42 \
  --output outputs/finetune/messidor2-examination-v2/split.csv \
  --plan outputs/finetune/messidor2-examination-v2/plan.json
```

The command refuses existing output paths. To reproduce again, use a different
local directory; the historical `reports/messidor2_split.csv` is not replaced.
Legacy `aptos.training.finetune split/run` now requires `--legacy-replay` so it
cannot silently start another filename-proxy experiment. That option is archival
reproduction only and does not consume this new split.

## Examination uncertainty interface

When actual matched predictions exist, use a CSV containing `id_code`, `diagnosis`,
`baseline_grade`, `candidate_grade` and the supplied `exam_id`:

```bash
python -m aptos.evaluation.paired_uncertainty \
  --predictions outputs/followup/test_predictions.csv \
  --unit examination --draws 2000 --seed 42 \
  --output outputs/followup/examination_intervals.json
```

The API also accepts `unit="examination", examination_ids=...`. It rejects
substituting patient IDs for examination IDs. The report labels its unit explicitly
and exports aggregate differences only. No interval was computed for v2 because
the new arms have not been trained; synthetic tests verify the software contract
and are not reported as scientific results.
