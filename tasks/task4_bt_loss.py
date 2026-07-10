"""Task 4 — Bradley-Terry preference loss [10 points].

Given chosen/rejected reward scores produced by the RM, return the
per-example BT loss:

    L_BT(s_+, s_-) = -log sigmoid(s_+ - s_-)

Returns a tensor of shape (batch,) — the caller .mean()s.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bt_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry preference loss for a batch of (chosen, rejected) RM scores.

    Element-wise:

        L_BT(s_+, s_-) = -log sigmoid(s_+ - s_-)

    The trainer takes ``.mean()`` over the batch.

    Args:
        chosen_scores: shape ``(batch,)`` — RM scores on the chosen completions.
        rejected_scores: shape ``(batch,)`` — RM scores on the rejected completions.

    Returns:
        Tensor of shape ``(batch,)`` — per-row BT loss.
    """
    # <YOUR CODE HERE>
    return -F.logsigmoid(chosen_scores - rejected_scores)

# --------------------------------------------------------------------------- #
# Self-check — run with: python -m tasks.task4_bt_loss                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    # Equal scores → sigmoid(0) = 0.5 → loss = log(2) ≈ 0.6931.
    a = torch.tensor([1.0, 1.0])
    b = torch.tensor([1.0, 1.0])
    assert abs(bt_loss(a, b).mean().item() - math.log(2.0)) < 1e-5, \
        "BT loss should be log(2) when chosen == rejected"

    # Three-pair fixture: chosen_scores - rejected_scores = [1, -1, 0].
    #   losses = [-log σ(1), -log σ(-1), -log σ(0)]
    #          = [0.313262,  1.313262,   0.693147]
    cs = torch.tensor([1.0,  2.0, -1.0])
    rs = torch.tensor([0.0,  3.0, -1.0])
    out = bt_loss(cs, rs)
    assert out.shape == (3,), f"expected shape (3,); got {out.shape}"
    expected = torch.tensor([0.313262, 1.313262, 0.693147])
    assert torch.allclose(out, expected, atol=1e-5), \
        f"bt_loss wrong: got {out}, expected {expected}"

    # Sign check: chosen LOSES by 1 → loss must exceed log(2).
    assert bt_loss(torch.tensor([0.0]), torch.tensor([1.0])).item() > math.log(2.0), \
        "BT loss must increase when chosen < rejected"

    print("bt_loss: all checks passed")
