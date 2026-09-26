# Testability plan: FlashAttention tiled softmax

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| F1 | Tiled == direct softmax attention (exact) | Cross-check | test_crosscheck.py::test_tiled_equals_direct | 1e-12, multiple shapes |
| F2 | Shift invariance; tiled survives logits where naive overflows | Property + stability | test_properties.py::test_softmax_shift_invariance, ::test_never_overflows | naive overflow documented deliberately |
| F3 | Rescale-by-e^{Δm} identity; first block (m=−∞) exact-zero semantics | Anchor | test_anchor.py::test_rescale_when_max_increases, ::test_first_block_frozen_start | hand constants 1+2σ(2), 3σ(2)−1 |
| F4 | Output invariant to block partition (bq,bk) | Property | test_properties.py::test_block_partition_invariance | grid incl. b=1 |
| F5 | Causal mask composes with tiling | Property + cross-check | test_properties.py::test_causal_masked_blocks_zero + test_crosscheck | |
| F6 | Gradients through tiled path == analytic direct grads | Gradient check | test_gradients.py | analytic backward verified separately in calibration/02_attention |
| — | Single-query two-tile hand case | Anchor | test_anchor.py | derivation in comments |
| IO / speedup | Systems claims (HBM access count, wall-clock) | **Not automatable** | → GAP_LIST #1-2 | needs GPU + kernel |
