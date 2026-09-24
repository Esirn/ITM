# MDM Multi-Seed Control Stability

Negative differences favor paired IMU. Confidence intervals bootstrap motions after averaging the five repeated seeds.

## head

| Seed | Paired-Text error (m) | Better than Text | Paired-Shuffled error (m) | Better than Shuffled | Jerk change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | +0.0013 | 15/30 | +0.0038 | 14/30 | -0.041 |
| 2345 | -0.0089 | 18/30 | -0.0075 | 19/30 | +0.078 |
| 3456 | -0.0110 | 15/30 | +0.0009 | 15/30 | -0.087 |
| 4567 | -0.0017 | 15/30 | +0.0036 | 12/30 | +0.009 |
| 5678 | +0.0016 | 15/30 | -0.0073 | 17/30 | +0.044 |

- Motion-averaged paired-Text: -0.0038 m [-0.0108, +0.0031].
- Motion-averaged paired-shuffled: -0.0013 m [-0.0088, +0.0056].
- Motion-averaged jerk change: +0.001 [-0.086, +0.087].
- Consistent on >=3/5 seeds: 13/30 vs Text, 15/30 vs shuffled.

## wrists

| Seed | Paired-Text error (m) | Better than Text | Paired-Shuffled error (m) | Better than Shuffled | Jerk change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | -0.0144 | 21/30 | -0.0096 | 23/30 | -0.028 |
| 2345 | -0.0337 | 25/30 | -0.0174 | 20/30 | +0.041 |
| 3456 | -0.0272 | 22/30 | -0.0177 | 19/30 | -0.127 |
| 4567 | -0.0322 | 22/30 | -0.0172 | 18/30 | +0.035 |
| 5678 | -0.0329 | 20/30 | -0.0277 | 24/30 | -0.179 |

- Motion-averaged paired-Text: -0.0281 m [-0.0414, -0.0160].
- Motion-averaged paired-shuffled: -0.0179 m [-0.0296, -0.0078].
- Motion-averaged jerk change: -0.052 [-0.147, +0.037].
- Consistent on >=3/5 seeds: 24/30 vs Text, 22/30 vs shuffled.

## Length-stratified motion averages

Each bin contains 10 motions and each motion is averaged over five seeds.

| Sensor | Length | Paired-Text error (m), 95% CI | Paired-Shuffled error (m), 95% CI | Jerk change, 95% CI |
| --- | --- | ---: | ---: | ---: |
| head | short_1_80 | +0.0062 [-0.0067, +0.0163] | +0.0014 [-0.0080, +0.0089] | +0.078 [-0.092, +0.243] |
| head | medium_81_140 | -0.0097 [-0.0214, +0.0014] | +0.0012 [-0.0134, +0.0152] | -0.149 [-0.282, -0.015] |
| head | long_141_196 | -0.0078 [-0.0178, +0.0033] | -0.0065 [-0.0213, +0.0053] | +0.073 [-0.004, +0.166] |
| wrists | short_1_80 | -0.0349 [-0.0687, -0.0066] | -0.0278 [-0.0570, -0.0024] | -0.039 [-0.268, +0.148] |
| wrists | medium_81_140 | -0.0225 [-0.0359, -0.0111] | -0.0191 [-0.0313, -0.0082] | -0.157 [-0.285, -0.023] |
| wrists | long_141_196 | -0.0269 [-0.0456, -0.0098] | -0.0069 [-0.0169, +0.0031] | +0.041 [-0.058, +0.132] |
