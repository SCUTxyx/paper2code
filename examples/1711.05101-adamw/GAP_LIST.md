# Gap list: AdamW (arXiv:1711.05101)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Generalization comparison vs Adam (Fig. 1-2) | GPU training runs + datasets | Empirical claims unverified; only the update rule's math is verified |
| 2 | η_t schedule multiplier interaction (§4) | Training setup | η_t fixed to 1 in this minimal reproduction |
| 3 | Hyperparameter-transfer claim (λ needs re-tuning ~1/η² under schedule) | Grid search | Untested; the paper's Table 2 numbers not recomputed |
| 4 | Optimizer-state memory / engineering concerns | Not math | Out of scope |
