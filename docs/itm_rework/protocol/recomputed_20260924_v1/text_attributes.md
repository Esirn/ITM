# Text Attribute Control under Fixed Head IMU

Positive values are the desired direction for the four semantic contrasts.

| Model | Quick-Slow speed | Turn-Base yaw | Swing-Base arms | Large-Base stride | Head error change |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage2b | -0.0199 (5/30) | +2.3689 (25/30) | +0.0979 (29/30) | -0.0237 (12/30) | +0.0074 m |

Each motion retains its own effective length; comparisons are paired within the same motion, head IMU, seed, and initial diffusion noise.

Secondary checks:

| Model | Quick-Slow cadence | Turn-Base net yaw | Turn efficiency | Large-Base foot separation |
| --- | ---: | ---: | ---: | ---: |
| stage2b | +0.2204 (19/30) | +3.2573 (26/30) | +0.3196 (23/30) | -0.0101 (14/30) |
