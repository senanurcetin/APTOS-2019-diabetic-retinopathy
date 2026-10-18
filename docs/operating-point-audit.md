# Why the referral flag and the operating-point audit differ

The ordinal five-class ensemble emits severity scores. This public research
repository has **two distinct decision rules**:

1. **Ordinal-grade threshold:** Predict grade 0–4 using the APTOS-trained
   four thresholds, then call predicted grades 2–4 referable. Published in
   `reports/external_validation*.json`.
2. **Out-of-fold selected raw-score cut:** Use a separate operating threshold
   (cut 1.3359375) chosen on the APTOS OOF training pool for target sensitivity
   >= 0.90. Apply the unchanged cut to the held-out APTOS and two external
   sets. Published in `reports/calibration.json`.

These are **not interchangeable**. For instance, the Messidor-2 grade rule
missed 307/457 referable eyes, whereas the OOF raw cut missed 338/457;
IDRiD misses were 35/304 and 56/304 respectively. These are recorded cohort
results, not new measurements. Different rules can change the referral rate
and false-negative count, and neither is proven suitable for clinical care.

## Offline reproduction

```bash
python -m aptos.reporting.operating_point_audit --reports-dir reports
python -m aptos.reporting.operating_point_audit --format json
pytest -m pure tests/test_operating_point_audit.py
```

The audit recovers integer counts from high-precision aggregate ratios only
when those ratios uniquely map to whole counts. It checks recall, specificity,
referral rate, PPV, selection provenance and unchanged external cohorts,
then summarizes exploratory cross-site recalibration. It **does not** know
case-level overlap or patient IDs; the intervals are Wilson descriptive
approximations under an independence assumption. See
`docs/paired-uncertainty.md` for comparisons that require paired raw predictions.

### Scientific boundaries

- OOF selection performance is not an independent estimate of generalization.
- Site-to-site recalibration in the existing report is exploratory and must not
  be presented as a prospectively validated operating policy.
- The observed ECE change may improve while sensitivity/specificity tradeoffs
  become unfavorable: inspect all metrics together.
- No clinical use, productization, new GPU result or data export is implied.
