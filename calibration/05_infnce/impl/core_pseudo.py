"""CLIP contrastive loss — independent formulation: per-pair loop with an explicit
softmax (literal transcription of InfoNCE §2.2).

For each anchor: the "positive logit − logsumexp(all logits)" is computed
individually; no similarity matrix and no matrix softmax are formed. For the
cross-check in test_crosscheck.py.
"""

import numpy as np


def clip_loss_loop(I, T, tau=0.07):
    """Same semantics as core_eq.clip_loss, implemented per pair."""
    I = np.asarray(I, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    In = I / np.sqrt((I * I).sum(axis=1, keepdims=True))
    Tn = T / np.sqrt((T * T).sum(axis=1, keepdims=True))
    n = I.shape[0]

    def direction(A, B):
        total = 0.0
        for i in range(n):
            sims = np.array([A[i] @ B[j] for j in range(n)]) / tau
            pos = sims[i]
            lse = np.max(sims) + np.log(np.exp(sims - np.max(sims)).sum())
            total += -(pos - lse)
        return total / n

    return 0.5 * (direction(In, Tn) + direction(Tn, In))
