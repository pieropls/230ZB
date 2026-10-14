# Extension 3 · the watermark as a filter for sentiment data

Signal: FinBERT S = P(pos) − P(neg) per text. 100 buckets of 50 human texts; half the buckets attacked, half of those pushed up (top third of the AI pool by S) and half down (bottom third); AI texts added until share c. Errors over buckets against the truth (the bucket's human mean); 500 full re-draws; square brackets are 2.5-97.5 percentiles over draws (Wilson for rates). Recovery = 1 − (RMSE_filter − RMSE_perfect) / (RMSE_none − RMSE_perfect). Two predictions assume escaping is unrelated to sentiment: 'random at rate p' removes each AI text independently with the pool's detection rate p (and each human text with the false-positive rate) inside the same draws; 'same count, shuffled' removes exactly the texts the detector flags but with their identities permuted within each slant pool on every draw (the H5 test is actual minus shuffled). The 'formula' is the bias prediction 1 − (1 − p)/(1 − cp) of section 1 of the spec; it sits above the simulated predictions because RMSE is a root mean square, not a bias. Human texts: the 1,000 human baselines (idx < 1000), cut to the AI length; without replacement within a bucket, with replacement across buckets. Slanted AI pools are reused across buckets (stated in the inventory).

## Main table

| scenario | AI texts | tokens (median) | c (actual) | AI pool / slant | detection rate p [95% CI] | human texts removed per draw | RMSE none / filter / perfect | Spearman none / filter / perfect | recovery [95% CI] | predicted: random at rate p [95% CI] | formula | predicted: same count, shuffled [95% CI] | actual − shuffled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | watermarked | 25 | 0.1 (0.107) | 493 / 164 | 0.103 [0.080, 0.133] | 0.00 of 5000 | 0.0284 / 0.0261 / 0.0000 | 0.860 / 0.879 / 1.000 | 0.081 [0.043, 0.123] | 0.084 [0.046, 0.127] | 0.093 | 0.081 [0.038, 0.133] | -0.000 [-0.067, 0.062] |
| A | watermarked | 25 | 0.3 (0.296) | 493 / 164 | 0.103 [0.080, 0.133] | 0.00 of 5000 | 0.0757 / 0.0704 / 0.0000 | 0.490 / 0.521 / 1.000 | 0.071 [0.051, 0.089] | 0.073 [0.053, 0.094] | 0.075 | 0.071 [0.037, 0.111] | -0.001 [-0.046, 0.039] |
| A | watermarked | 50 | 0.1 (0.107) | 493 / 164 | 0.533 [0.489, 0.577] | 0.00 of 5000 | 0.0327 / 0.0190 / 0.0000 | 0.831 / 0.935 / 1.000 | 0.419 [0.354, 0.492] | 0.454 [0.383, 0.527] | 0.505 | 0.443 [0.359, 0.518] | -0.024 [-0.127, 0.085] |
| A | watermarked | 50 | 0.3 (0.296) | 493 / 164 | 0.533 [0.489, 0.577] | 0.00 of 5000 | 0.0879 / 0.0531 / 0.0000 | 0.450 / 0.665 / 1.000 | 0.397 [0.361, 0.434] | 0.433 [0.397, 0.471] | 0.446 | 0.417 [0.360, 0.480] | -0.021 [-0.090, 0.047] |
| A | watermarked | 100 | 0.1 (0.107) | 493 / 164 | 0.939 [0.914, 0.957] | 0.00 of 5000 | 0.0331 / 0.0052 / 0.0000 | 0.825 / 0.995 / 1.000 | 0.844 [0.785, 0.904] | 0.864 [0.810, 0.920] | 0.932 | 0.874 [0.809, 0.931] | -0.030 [-0.111, 0.055] |
| A | watermarked | 100 | 0.3 (0.296) | 493 / 164 | 0.939 [0.914, 0.957] | 0.00 of 5000 | 0.0894 / 0.0118 / 0.0000 | 0.451 / 0.974 / 1.000 | 0.868 [0.836, 0.896] | 0.883 [0.853, 0.909] | 0.916 | 0.895 [0.854, 0.929] | -0.027 [-0.071, 0.023] |
| A | watermarked | 200 | 0.1 (0.107) | 493 / 164 | 0.986 [0.971, 0.993] | 0.00 of 5000 | 0.0309 / 0.0019 / 0.0000 | 0.827 / 0.999 / 1.000 | 0.939 [0.890, 1.000] | 0.945 [0.897, 1.000] | 0.984 | 0.969 [0.909, 1.000] | -0.030 [-0.093, 0.058] |
| A | watermarked | 200 | 0.3 (0.296) | 493 / 164 | 0.986 [0.971, 0.993] | 0.00 of 5000 | 0.0833 / 0.0038 / 0.0000 | 0.453 / 0.996 / 1.000 | 0.954 [0.938, 0.973] | 0.956 [0.935, 0.975] | 0.980 | 0.976 [0.946, 0.994] | -0.022 [-0.050, 0.014] |
| B | paraphrase, light | 151 | 0.1 (0.107) | 200 / 66 | 0.220 [0.168, 0.282] | 0.00 of 5000 | 0.0398 / 0.0329 / 0.0000 | 0.752 / 0.814 / 1.000 | 0.172 [0.132, 0.216] | 0.184 [0.138, 0.231] | 0.201 | 0.194 [0.138, 0.256] | -0.022 [-0.100, 0.053] |
| B | paraphrase, light | 151 | 0.3 (0.296) | 200 / 66 | 0.220 [0.168, 0.282] | 0.00 of 5000 | 0.1083 / 0.0922 / 0.0000 | 0.381 / 0.421 / 1.000 | 0.149 [0.130, 0.167] | 0.162 [0.139, 0.187] | 0.166 | 0.169 [0.126, 0.214] | -0.020 [-0.069, 0.022] |
| B | paraphrase, medium | 177 | 0.1 (0.107) | 200 / 66 | 0.035 [0.017, 0.070] | 0.00 of 5000 | 0.0427 / 0.0420 / 0.0000 | 0.729 / 0.735 / 1.000 | 0.018 [0.005, 0.033] | 0.028 [0.010, 0.049] | 0.031 | 0.025 [0.007, 0.049] | -0.007 [-0.035, 0.015] |
| B | paraphrase, medium | 177 | 0.3 (0.296) | 200 / 66 | 0.035 [0.017, 0.070] | 0.00 of 5000 | 0.1164 / 0.1148 / 0.0000 | 0.372 / 0.374 / 1.000 | 0.014 [0.008, 0.019] | 0.024 [0.015, 0.035] | 0.025 | 0.021 [0.006, 0.037] | -0.007 [-0.025, 0.009] |
| B | paraphrase, full | 178 | 0.1 (0.107) | 200 / 66 | 0.015 [0.005, 0.043] | 0.00 of 5000 | 0.0427 / 0.0423 / 0.0000 | 0.728 / 0.732 / 1.000 | 0.008 [0.002, 0.017] | 0.012 [0.001, 0.026] | 0.013 | 0.013 [0.002, 0.031] | -0.005 [-0.024, 0.009] |
| B | paraphrase, full | 178 | 0.3 (0.296) | 200 / 66 | 0.015 [0.005, 0.043] | 0.00 of 5000 | 0.1166 / 0.1159 / 0.0000 | 0.373 / 0.374 / 1.000 | 0.006 [0.004, 0.008] | 0.010 [0.005, 0.017] | 0.011 | 0.011 [0.002, 0.024] | -0.005 [-0.018, 0.005] |
| D | unwatermarked | 200 | 0.1 (0.107) | 492 / 164 | 0.002 [0.000, 0.011] | 0.00 of 5000 | 0.0302 / 0.0302 / 0.0000 | 0.833 / 0.833 / 1.000 | 0.000 [0.000, 0.000] | 0.002 [0.000, 0.009] | 0.002 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| D | unwatermarked | 200 | 0.3 (0.296) | 492 / 164 | 0.002 [0.000, 0.011] | 0.00 of 5000 | 0.0815 / 0.0815 / 0.0000 | 0.457 / 0.457 / 1.000 | 0.000 [0.000, 0.000] | 0.002 [-0.000, 0.005] | 0.001 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## Escapers (H5): sentiment of the AI texts the filter misses

| scenario | AI texts | tokens | threshold | up pool: escaped / n | μ_A' − μ_A (up) [95% CI] | down pool: escaped / n | μ_A' − μ_A (down) [95% CI] |
|---|---|---|---|---|---|---|---|
| A | watermarked | 25 | z > 4.00 | 146 / 164 | 0.000 [-0.015, 0.014] | 149 / 164 | -0.000 [-0.016, 0.015] |
| A | watermarked | 50 | z > 4.00 | 83 / 164 | 0.039 [-0.004, 0.085] | 75 / 164 | 0.010 [-0.042, 0.061] |
| A | watermarked | 100 | z > 4.00 | 9 / 164 | -0.029 [-0.174, 0.131] | 9 / 164 | -0.241 [-0.402, -0.042] |
| A | watermarked | 200 | z > 4.00 | 1 / 164 | – | 1 / 164 | – |
| B | paraphrase, light | 151 | z > 4.00 | 50 / 66 | -0.002 [-0.039, 0.037] | 52 / 66 | -0.027 [-0.060, 0.002] |
| B | paraphrase, medium | 177 | z > 4.00 | 64 / 66 | 0.001 [-0.010, 0.014] | 64 / 66 | -0.008 [-0.020, 0.000] |
| B | paraphrase, full | 178 | z > 4.00 | 65 / 66 | 0.004 [0.000, 0.014] | 65 / 66 | -0.001 [-0.004, 0.000] |
| D | unwatermarked | 200 | z > 4.00 | 164 / 164 | 0.000 [0.000, 0.000] | 164 / 164 | 0.000 [0.000, 0.000] |

## Controls (section 6)

| source | tokens (nominal / median) | n | neutral / positive / negative | μ (S) ± sd | flagged at z > 4 [95% CI] | largest attainable z | slant pools: n, μ_up, μ_down | flagged in up / down pool |
|---|---|---|---|---|---|---|---|---|
| human | 25 / 25 | 1000 | 0.739 / 0.095 / 0.166 | -0.026 ± 0.400 | 0.000 [0.000, 0.004] | 5.0 | – | – |
| human | 50 / 50 | 1000 | 0.711 / 0.101 / 0.188 | -0.056 ± 0.408 | 0.000 [0.000, 0.004] | 7.1 | – | – |
| human | 100 / 100 | 1000 | 0.719 / 0.088 / 0.193 | -0.076 ± 0.407 | 0.000 [0.000, 0.004] | 10.0 | – | – |
| human | 200 / 200 | 1000 | 0.739 / 0.058 / 0.203 | -0.103 ± 0.381 | 0.000 [0.000, 0.004] | 14.1 | – | – |
| watermarked | 25 / 25 | 493 | 0.777 / 0.097 / 0.126 | -0.011 ± 0.377 | 0.103 [0.080, 0.133] | 5.0 | 164, +0.327, -0.378 | 0.110 / 0.091 |
| watermarked | 50 / 50 | 493 | 0.712 / 0.122 / 0.166 | -0.035 ± 0.418 | 0.533 [0.489, 0.577] | 7.1 | 164, +0.357, -0.468 | 0.494 / 0.543 |
| watermarked | 100 / 100 | 493 | 0.702 / 0.093 / 0.205 | -0.078 ± 0.414 | 0.939 [0.914, 0.957] | 10.0 | 164, +0.306, -0.531 | 0.945 / 0.945 |
| watermarked | 200 / 200 | 493 | 0.748 / 0.053 / 0.199 | -0.111 ± 0.386 | 0.986 [0.971, 0.993] | 14.1 | 164, +0.232, -0.544 | 0.994 / 0.994 |
| unwatermarked | 200 / 200 | 492 | 0.726 / 0.059 / 0.215 | -0.105 ± 0.374 | 0.002 [0.000, 0.011] | 14.1 | 164, +0.235, -0.526 | 0.000 / 0.000 |
| paraphrase, light | 200 / 151 | 200 | 0.625 / 0.130 / 0.245 | -0.067 ± 0.471 | 0.220 [0.168, 0.282] | 12.3 | 66, +0.417, -0.608 | 0.242 / 0.212 |
| paraphrase, medium | 200 / 177 | 200 | 0.590 / 0.175 / 0.235 | -0.034 ± 0.503 | 0.035 [0.017, 0.070] | 13.3 | 66, +0.488, -0.613 | 0.030 / 0.030 |
| paraphrase, full | 200 / 178 | 200 | 0.565 / 0.170 / 0.265 | -0.037 ± 0.496 | 0.015 [0.005, 0.043] | 13.3 | 66, +0.488, -0.616 | 0.015 / 0.015 |

## Timing

| step | n | seconds | per second | device |
|---|---|---|---|---|
| score_pilot | 600 | 3.7 | 164.1 | mps torch.float16 |
| score | 7064 | 33.1 | 213.5 | mps torch.float16 |
| score_docs (scenario C) | 1600 | 40.1 | 39.9 | mps torch.float16 |
| simulation A, B, D | 16 cells x 500 draws | 31.9 |  | cpu  |
| simulation A, calibrated threshold | 8 cells x 500 draws | 16.2 |  | cpu  |
| simulation C | 18 cells x 500 draws | 34.9 |  | cpu  |

## Tier 2: threshold calibrated to a 1% false-positive rate on human text of the same length (scenario A)

The threshold is the 99th percentile of z over the 1,000 human texts at that length (`stats.tpr_at_fpr`); the realised FPR is below 1% because z is discrete at short lengths.

| tokens | c | threshold | AI texts removed (p) [95% CI] | human texts removed (FPR) | human removed per draw (of 5000) | RMSE none / filter | recovery [95% CI] | predicted (random) [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.1 | z > 4.00 (fixed) | 0.103 [0.080, 0.133] | 0.000 [≤ 0.004] | 0.0 | 0.0284 / 0.0261 | 0.081 [0.043, 0.123] | 0.084 [0.046, 0.127] |
| 25 | 0.1 | z > 2.20 (1% FPR) | 0.738 [0.698, 0.775] | 0.004 [≤ 0.010] | 20.1 | 0.0284 / 0.0112 | 0.605 [0.532, 0.680] | 0.617 [0.542, 0.696] |
| 25 | 0.3 | z > 4.00 (fixed) | 0.103 [0.080, 0.133] | 0.000 [≤ 0.004] | 0.0 | 0.0757 / 0.0704 | 0.071 [0.051, 0.089] | 0.073 [0.053, 0.094] |
| 25 | 0.3 | z > 2.20 (1% FPR) | 0.738 [0.698, 0.775] | 0.004 [≤ 0.010] | 19.8 | 0.0757 / 0.0282 | 0.628 [0.584, 0.667] | 0.634 [0.587, 0.674] |
| 50 | 0.1 | z > 4.00 (fixed) | 0.533 [0.489, 0.577] | 0.000 [≤ 0.004] | 0.0 | 0.0327 / 0.0190 | 0.419 [0.354, 0.492] | 0.454 [0.383, 0.527] |
| 50 | 0.1 | z > 2.26 (1% FPR) | 0.949 [0.926, 0.965] | 0.003 [≤ 0.009] | 15.2 | 0.0327 / 0.0048 | 0.854 [0.807, 0.897] | 0.841 [0.786, 0.892] |
| 50 | 0.3 | z > 4.00 (fixed) | 0.533 [0.489, 0.577] | 0.000 [≤ 0.004] | 0.0 | 0.0879 / 0.0531 | 0.397 [0.361, 0.434] | 0.433 [0.397, 0.471] |
| 50 | 0.3 | z > 2.26 (1% FPR) | 0.949 [0.926, 0.965] | 0.003 [≤ 0.009] | 14.9 | 0.0879 / 0.0093 | 0.895 [0.872, 0.917] | 0.889 [0.860, 0.916] |

## Tier 2: scenario C, watermarked passage pasted into human text

Documents of Ext. 2b rebuilt with their text (seed 0; z identical to `results/raw/dilution/docs.jsonl.gz`). Document score = mean FinBERT S of the two 300-token halves (FinBERT reads at most 512 tokens). Human pool: 1000 human-only documents (neutral share 0.811, μ_H = -0.120); AI pool: the 200 mixed documents per share, slanted by the pasted span's own S (span S of the up / down thirds: 10%: +0.33 / -0.59, 25%: +0.29 / -0.45, 50%: +0.20 / -0.52). Thresholds: z > 4, and the 1% FPR thresholds of Ext. 2b (full-text z > 2.20, WinMax > 4.02).

| watermarked span | detector | c | AI docs removed (p) [95% CI] | human docs removed (FPR) | μ_H / μ_up / μ_down (doc S) | RMSE none / filter | Spearman none / filter | recovery [95% CI] | predicted: random at p [95% CI] | predicted: shuffled [95% CI] | actual − shuffled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% (60 of 600 tokens) | whole text, z > 4 | 0.1 | 0.010 [0.003, 0.036] | 0.000 | -0.120 / -0.080 / -0.179 | 0.0097 / 0.0096 | 0.959 / 0.959 | 0.010 [-0.008, 0.034] | 0.004 [-0.019, 0.036] | 0.004 [-0.018, 0.044] | 0.006 [-0.035, 0.038] |
| 10% (60 of 600 tokens) | whole text, z > 4 | 0.3 | 0.010 [0.003, 0.036] | 0.000 | -0.120 / -0.080 / -0.179 | 0.0169 / 0.0166 | 0.875 / 0.879 | 0.019 [0.003, 0.039] | 0.003 [-0.019, 0.026] | 0.004 [-0.040, 0.060] | 0.015 [-0.040, 0.062] |
| 10% (60 of 600 tokens) | whole text, 1% FPR threshold | 0.1 | 0.115 [0.078, 0.167] | 0.008 | -0.120 / -0.080 / -0.179 | 0.0097 / 0.0100 | 0.959 / 0.957 | -0.029 [-0.144, 0.073] | -0.015 [-0.133, 0.095] | 0.000 [-0.141, 0.151] | -0.029 [-0.163, 0.093] |
| 10% (60 of 600 tokens) | whole text, 1% FPR threshold | 0.3 | 0.115 [0.078, 0.167] | 0.008 | -0.120 / -0.080 / -0.179 | 0.0169 / 0.0172 | 0.875 / 0.875 | -0.019 [-0.099, 0.056] | 0.023 [-0.067, 0.110] | 0.039 [-0.132, 0.196] | -0.058 [-0.232, 0.107] |
| 10% (60 of 600 tokens) | WinMax window, 1% FPR threshold | 0.1 | 0.865 [0.811, 0.906] | 0.006 | -0.120 / -0.080 / -0.179 | 0.0097 / 0.0039 | 0.959 / 0.992 | 0.598 [0.480, 0.699] | 0.507 [0.347, 0.643] | 0.595 [0.429, 0.727] | 0.003 [-0.138, 0.164] |
| 10% (60 of 600 tokens) | WinMax window, 1% FPR threshold | 0.3 | 0.865 [0.811, 0.906] | 0.006 | -0.120 / -0.080 / -0.179 | 0.0169 / 0.0057 | 0.875 / 0.985 | 0.661 [0.572, 0.732] | 0.563 [0.446, 0.665] | 0.625 [0.433, 0.764] | 0.035 [-0.120, 0.226] |
| 25% (150 of 600 tokens) | whole text, z > 4 | 0.1 | 0.285 [0.227, 0.351] | 0.000 | -0.120 / +0.016 / -0.206 | 0.0114 / 0.0080 | 0.944 / 0.971 | 0.296 [0.190, 0.399] | 0.193 [0.100, 0.300] | 0.187 [0.058, 0.323] | 0.109 [-0.076, 0.269] |
| 25% (150 of 600 tokens) | whole text, z > 4 | 0.3 | 0.285 [0.227, 0.351] | 0.000 | -0.120 / +0.016 / -0.206 | 0.0265 / 0.0179 | 0.743 / 0.864 | 0.326 [0.272, 0.381] | 0.192 [0.134, 0.252] | 0.173 [0.015, 0.306] | 0.153 [0.002, 0.323] |
| 25% (150 of 600 tokens) | whole text, 1% FPR threshold | 0.1 | 0.815 [0.755, 0.863] | 0.008 | -0.120 / +0.016 / -0.206 | 0.0114 / 0.0047 | 0.944 / 0.989 | 0.584 [0.466, 0.691] | 0.525 [0.381, 0.642] | 0.541 [0.384, 0.672] | 0.042 [-0.076, 0.185] |
| 25% (150 of 600 tokens) | whole text, 1% FPR threshold | 0.3 | 0.815 [0.755, 0.863] | 0.008 | -0.120 / +0.016 / -0.206 | 0.0265 / 0.0078 | 0.743 / 0.973 | 0.706 [0.657, 0.757] | 0.645 [0.566, 0.709] | 0.659 [0.504, 0.777] | 0.047 [-0.072, 0.211] |
| 25% (150 of 600 tokens) | WinMax window, 1% FPR threshold | 0.1 | 0.995 [0.972, 0.999] | 0.006 | -0.120 / +0.016 / -0.206 | 0.0114 / 0.0022 | 0.944 / 0.997 | 0.804 [0.739, 0.855] | 0.733 [0.605, 0.827] | 0.804 [0.739, 0.855] | 0.000 [0.000, 0.000] |
| 25% (150 of 600 tokens) | WinMax window, 1% FPR threshold | 0.3 | 0.995 [0.972, 0.999] | 0.006 | -0.120 / +0.016 / -0.206 | 0.0265 / 0.0022 | 0.743 / 0.997 | 0.916 [0.894, 0.936] | 0.879 [0.836, 0.919] | 0.916 [0.894, 0.936] | 0.000 [0.000, 0.000] |
| 50% (300 of 600 tokens) | whole text, z > 4 | 0.1 | 0.975 [0.943, 0.989] | 0.000 | -0.120 / +0.048 / -0.262 | 0.0140 / 0.0003 | 0.917 / 1.000 | 0.979 [0.961, 1.000] | 0.894 [0.818, 0.958] | 0.957 [0.870, 1.000] | 0.022 [-0.028, 0.114] |
| 50% (300 of 600 tokens) | whole text, z > 4 | 0.3 | 0.975 [0.943, 0.989] | 0.000 | -0.120 / +0.048 / -0.262 | 0.0346 / 0.0006 | 0.643 / 1.000 | 0.983 [0.976, 0.989] | 0.912 [0.879, 0.944] | 0.963 [0.910, 0.993] | 0.019 [-0.012, 0.074] |
| 50% (300 of 600 tokens) | whole text, 1% FPR threshold | 0.1 | 1.000 [0.981, 1.000] | 0.008 | -0.120 / +0.048 / -0.262 | 0.0140 / 0.0034 | 0.917 / 0.994 | 0.755 [0.672, 0.835] | 0.754 [0.669, 0.839] | 0.755 [0.672, 0.835] | 0.000 [0.000, 0.000] |
| 50% (300 of 600 tokens) | whole text, 1% FPR threshold | 0.3 | 1.000 [0.981, 1.000] | 0.008 | -0.120 / +0.048 / -0.262 | 0.0346 / 0.0035 | 0.643 / 0.994 | 0.899 [0.867, 0.932] | 0.901 [0.868, 0.934] | 0.899 [0.867, 0.932] | 0.000 [0.000, 0.000] |
| 50% (300 of 600 tokens) | WinMax window, 1% FPR threshold | 0.1 | 1.000 [0.981, 1.000] | 0.006 | -0.120 / +0.048 / -0.262 | 0.0140 / 0.0022 | 0.917 / 0.997 | 0.841 [0.797, 0.881] | 0.789 [0.698, 0.861] | 0.841 [0.797, 0.881] | 0.000 [0.000, 0.000] |
| 50% (300 of 600 tokens) | WinMax window, 1% FPR threshold | 0.3 | 1.000 [0.981, 1.000] | 0.006 | -0.120 / +0.048 / -0.262 | 0.0346 / 0.0022 | 0.643 / 0.997 | 0.935 [0.919, 0.950] | 0.914 [0.884, 0.946] | 0.935 [0.919, 0.950] | 0.000 [0.000, 0.000] |

Escapers (document S of the mixed documents the detector misses, pools slanted by the span's S):

| scenario | AI texts | tokens | threshold | up pool: escaped / n | μ_A' − μ_A (up) [95% CI] | down pool: escaped / n | μ_A' − μ_A (down) [95% CI] |
|---|---|---|---|---|---|---|---|
| C | 10% span, whole text, z > 4 | 600 | z > 4.00 | 66 / 66 | 0.000 [0.000, 0.000] | 65 / 66 | 0.004 [0.000, 0.012] |
| C | 10% span, whole text, 1% FPR threshold | 600 | z > 2.20 | 61 / 66 | -0.008 [-0.020, 0.001] | 53 / 66 | -0.021 [-0.060, 0.013] |
| C | 10% span, WinMax window, 1% FPR threshold | 600 | z > 4.02 | 7 / 66 | -0.025 [-0.253, 0.207] | 7 / 66 | 0.136 [-0.095, 0.315] |
| C | 25% span, whole text, z > 4 | 600 | z > 4.00 | 50 / 66 | -0.029 [-0.069, 0.007] | 46 / 66 | 0.019 [-0.023, 0.060] |
| C | 25% span, whole text, 1% FPR threshold | 600 | z > 2.20 | 13 / 66 | -0.001 [-0.095, 0.083] | 10 / 66 | 0.039 [-0.110, 0.175] |
| C | 25% span, WinMax window, 1% FPR threshold | 600 | z > 4.02 | 0 / 66 | – | 0 / 66 | – |
| C | 50% span, whole text, z > 4 | 600 | z > 4.00 | 1 / 66 | – | 0 / 66 | – |
| C | 50% span, whole text, 1% FPR threshold | 600 | z > 2.20 | 0 / 66 | – | 0 / 66 | – |
| C | 50% span, WinMax window, 1% FPR threshold | 600 | z > 4.02 | 0 / 66 | – | 0 / 66 | – |

## Hypotheses

- **H1 (contamination harms the signal)**: with no filter, RMSE at 200 tokens is 0.031 (c = 0.1) and 0.083 (c = 0.3) against 0 for the clean signal; Spearman with the truth falls from 1 to 0.83 / 0.45. Supported.
- **H2 (long, unattacked text: almost all damage undone)**: recovery at 200 tokens 0.94 [0.89, 1.00] (c = 0.1), 0.95 [0.94, 0.97] (c = 0.3); Spearman back to 0.999 / 0.996. Supported.
- **H3 (recovery falls with length, in line with detection)**: recovery at 25 / 50 / 100 / 200 tokens (c = 0.3) 0.07 / 0.40 / 0.87 / 0.95 against detection rates 0.10 / 0.53 / 0.94 / 0.99. Supported; at 25 tokens the largest attainable z is 5.0, so the fixed threshold z > 4 is the binding constraint.
- **H4 (rewrite attack: recovery collapses)**: full paraphrase 0.01 [0.00, 0.02] (c = 0.1), 0.01 [0.00, 0.01] (c = 0.3); light paraphrase 0.17 at best. Supported.
- **H5 (escaping is unrelated to sentiment)**: actual minus shuffled-flags recovery is between -0.030 and +0.000 over the 16 cells of scenarios A, B, D; 16 of 16 intervals cover zero. Not rejected for the watermarked texts themselves; the point estimates are all slightly negative, and the largest gap is at 50 tokens (-0.021 [-0.090, +0.047] at c = 0.3).
- **Scenario D (sanity)**: the filter removes 0.0 of 1050 unwatermarked AI texts per draw; recovery 0.00 [0.00, 0.00] / 0.00 [0.00, 0.00]. The filter is blind to a model that does not embed the watermark.
- **Tier 2, calibrated threshold**: at 25 tokens a 1% FPR threshold (z > 2.20) raises recovery from 0.07 to 0.63 (c = 0.3) while removing 20 of 5,000 human texts per draw.
- **H5 on pasted passages (scenario C)**: against the shuffled-flags prediction, actual minus predicted is between -0.058 and +0.153 and its interval excludes zero in 1 of 18 cells (25%, whole text, z > 4, c = 0.3: +0.15 [+0.00, +0.32]). Against independent random removal at rate p the filter looks better than chance in 3 cells (up to +0.13); that gap is a small-pool effect, not sentiment: with 66 documents per slant and a handful of fixed escapers, the number of escapers per bucket varies less than under independent removal, and the RMSE is driven by the buckets with the most escapers.
- **H6 (window detector on pasted passages)**: at a 10% span, recovery (c = 0.3) is whole text, z > 4: 0.02 [0.00, 0.04], whole text, 1% FPR threshold: -0.02 [-0.10, 0.06], WinMax window, 1% FPR threshold: 0.66 [0.57, 0.73]. The contamination itself is small because 90% of each pasted document is human text (no-filter RMSE 0.0169 against 0.0833 in scenario A).
