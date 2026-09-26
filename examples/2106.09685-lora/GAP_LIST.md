# Gap list: LoRA (arXiv:2106.09685)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Task-quality comparison vs full fine-tuning (paper §5 tables) | GPU training + GLUE-style benchmarks | Effect claims unverified; only the adaptation math is verified |
| 2 | Multi-adapter serving / dynamic switching (§4.4 latency analysis) | Serving infrastructure | The merge/unmerge *identity* is verified; its latency benefit is not |
| 3 | QLoRA / quantized base weights | 4-bit quantization stack | Interaction of low-rank updates with quantized bases untested |
| 4 | Rank ablation (r = 1..64 curves) | Training runs | r's effect is empirical; only rank(ΔW) ≤ r is verified |
| 5 | Interaction with dropout / optimizers in real trainers | Training setup | Gradients here assume a plain linear loss without regularization |
