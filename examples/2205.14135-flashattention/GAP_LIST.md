# Gap list: FlashAttention tiled softmax (arXiv:2205.14135)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | IO-complexity analysis (§4: HBM access counts, O(n²d/M)) | GPU + SRAM/HBM cost model | The paper's headline contribution is systems-level; here only the math core (exactness of tiled softmax) is verified |
| 2 | Wall-clock speedup vs standard attention (Table 1, Fig 5) | GPU, kernels, baselines | Performance claims unverified (OUT of scope by design) |
| 3 | Backward-pass tiling with recomputation (Algorithm 2) | Separate tiled-backward implementation | Gradient *equality* is verified through the tiled forward, but the memory-efficient backward itself is not reproduced |
| 4 | fp16/bf16 accumulation behavior | Half-precision environment | Production kernels need to re-verify rounding effects of the block-wise accumulation order |
| 5 | SRAM tile sizes / block-size tuning (§4.2) | Hardware specs | Defaults (block 2×2) are for testing; not performance guidance |
| 6 | Long-context / multi-GPU benchmarks | Real workloads | OUT of scope |
