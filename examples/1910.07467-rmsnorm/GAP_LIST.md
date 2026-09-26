# Gap list: RMSNorm (arXiv:1910.07467)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Kernel fusion / wall-clock claims (§4.1.1) | GPU + CUDA kernel | The paper's speed motivation is systems-level; only the math piece is verified |
| 2 | Benchmark tables (§4.2, NLP tasks) | Training runs + datasets | Effect claims unverified (out of scope) |
| 3 | fp16/bf16 behavior (the usual motivation for ε) | Half-precision environment | Rounding of the ε term untested |
| 4 | pRMSNorm (partial-RMS variant, §5) | Paper's exact sampling scheme | Only full-RMS Eq.(4) reproduced |
| 5 | Gain-ordering conventions across codebases | Survey of real implementations | This repro pins g-after-norm per Eq.(4); real stacks vary |
