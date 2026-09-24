## External validation on IDRiD

- 455 images, scored by the baseline 4-threshold fold ensemble from sweep `baseline-6e66147526`
- no fine-tuning, and the APTOS validation thresholds were **not refitted**

| metric | IDRiD |
|---|---|
| QWK | 0.8045 |
| accuracy | 0.5868 |
| macro F1 | 0.4609 |
| referable sensitivity | 0.8849 |
| referable specificity | 0.9868 |

Healthy eyes (grade 0): 129. 82.9% were called healthy; 1.6% were called referable.

Severe cases missed: 2/148.
