# Gap list: Diffusion Policy math core (arXiv:2303.04137)

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Real-robot success rates (Table 1-2) | Robot hardware + demonstration datasets | Headline empirical claims unverified; math core only |
| 2 | Sim benchmark tables (Push-T, Block Push, Robomimic) | Sim environments + trained checkpoints | Same |
| 3 | Visual encoder (ResNet/Transformer) and conditioning path | Trained weights | Tests use the true ε or biased ε; no learned components |
| 4 | Squared-cosine schedule + iDDPM variant (sim setting) | Schedule/variant specifics | This repro exercises linear schedule + DDIM only |
| 5 | Training dynamics / ablations (CNN vs transformer backbone, action horizon sweep) | GPU training | Paper's design-space claims unverified |
| 6 | Warm-starting between replans | Paper marks it optional; details sparse | Not modeled in the executor semantics |
