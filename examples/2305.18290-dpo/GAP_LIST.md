# Gap list: DPO (arXiv:2305.18290)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Eq.(7)'s expectation not estimated on real preference data | Preference dataset (e.g. HH-RLHF) + trained policies | Effect and convergence unverified; only the loss's math piece is verified |
| 2 | Comparison against the RLHF-PPO baseline | GPU training environment + evaluation pipeline | Paper Tables/Figures not recomputed (out of scope) |
| 3 | β sensitivity and KL-deviation behavior | Training time | Only β's algebraic role (its position in the formula) is verified, not its empirical behavior |
| 4 | Source of log-probs | White-box model emitting log-probs | Not applicable to black-box API scenarios (same as the paper) |
| 5 | Length normalization for multi-turn/long answers | Unspecified in the paper; community-divergent | Implemented in the paper's original convention, no normalization added |
