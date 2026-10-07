TriviaQA validation (first 500), greedy, up to 16 new tokens, γ = 0.5, δ = 2. Exact match against normalized aliases (first line of the answer).

| model            |   EM_nowm |   EM_wm |   F1_nowm |   F1_wm |   z_nowm |   z_wm |
|:-----------------|----------:|--------:|----------:|--------:|---------:|-------:|
| FLAN-UL2 (paper) |     0.374 |   0.336 |     0.415 |   0.378 |   -0.007 |  0.402 |
| BLOOMZ (paper)   |     0.296 |   0.259 |     0.343 |   0.312 |    0.008 |  0.255 |
| opt              |     0.000 |   0.004 |     0.002 |   0.028 |   -0.414 |  1.432 |
| qwen             |     0.072 |   0.070 |     0.111 |   0.108 |   -0.236 |  1.183 |
| qwen-it          |     0.202 |   0.212 |     0.279 |   0.269 |   -0.053 |  0.803 |
