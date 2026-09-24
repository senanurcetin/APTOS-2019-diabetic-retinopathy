## External validation on IDRiD

- 455 images, scored by the squash 4-threshold fold ensemble from sweep `squash-d9db05d81f`
- no fine-tuning, and the APTOS validation thresholds were **not refitted**

| metric | IDRiD |
|---|---|
| QWK | 0.7859 |
| accuracy | 0.5648 |
| macro F1 | 0.4468 |
| referable sensitivity | 0.8618 |
| referable specificity | 0.9934 |

Healthy eyes (grade 0): 129. 79.8% were called healthy; 0.8% were called referable.

Severe cases missed: 3/148.
