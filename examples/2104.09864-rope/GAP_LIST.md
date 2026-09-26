# Gap list: RoPE (arXiv:2104.09864)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | §3.4.3 "inner-product upper bound decays with relative distance" not turned into an automated test | The asymptotic bound depends on vector structure; no deterministic criterion | That property only gets human/statistical verification; declared in TEST_PLAN |
| 2 | Long-context extrapolation quality (a headline selling point) | Real corpora + a trained model (GPU) | Out of repo scope: only the position-encoding math piece is tested |
| 3 | Downstream task quality (RoFormer vs Transformer baseline) | Same + training time | Same |
| 4 | fp16/bf16 numerical behavior | Half-precision environment | Production implementations must verify rounding effects themselves |
| 5 | Non-standard bases (base ≠ 10000) variants | Not in the paper; proposed by later work | Only the original Eq.(15) configuration reproduced |
