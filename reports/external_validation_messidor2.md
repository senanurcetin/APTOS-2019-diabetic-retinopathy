## External validation on Messidor-2

- 1744 images, scored by the baseline 4-threshold fold ensemble from sweep `baseline-6e66147526`
- no fine-tuning, and the APTOS validation thresholds were **not refitted**

| metric | Messidor-2 |
|---|---|
| QWK | 0.4928 |
| accuracy | 0.5940 |
| macro F1 | 0.2923 |
| referable sensitivity | 0.3282 |
| referable specificity | 0.9899 |

Healthy eyes (grade 0): 1017. 93.0% were called healthy; 1.2% were called referable.

Severe cases missed: 18/110.
