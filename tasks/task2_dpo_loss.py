"""Task 2 — DPO loss [15 points].

Given log-probabilities of the chosen and rejected completion under
both the policy and a frozen reference, return:

    losses           — per-example shape (batch,)
    chosen_rewards   — beta * (policy_chosen - reference_chosen), detached
    rejected_rewards — beta * (policy_rejected - reference_rejected), detached

The DPO loss is:

    -log sigmoid( beta * (
        log pi(y+|x) - log pi_ref(y+|x)
      - log pi(y-|x) + log pi_ref(y-|x)
    ))

The chosen/rejected rewards do NOT feed the optimiser — they're a
logging signal: their margin should rise during training, and either
drifting strongly negative is a known DPO-collapse leading indicator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss for one batch of pairs.

    For each pair (chosen y_+, rejected y_-) on prompt x, compute

        L = -log sigmoid( beta * [
            log( pi(y_+|x) / pi_ref(y_+|x) )
          - log( pi(y_-|x) / pi_ref(y_-|x) )
        ] )

    where ``pi`` is the trainable policy and ``pi_ref`` is the frozen
    reference (the SFT-merged checkpoint in this homework). All four
    log-prob tensors are *per-row sums* of token log-probs over each
    completion — the caller aggregates per-row before calling this.

    Args:
        policy_chosen_logps: shape ``(batch,)`` — pi log-probs on chosen.
        policy_rejected_logps: shape ``(batch,)`` — pi log-probs on rejected.
        reference_chosen_logps: shape ``(batch,)`` — pi_ref log-probs on chosen.
        reference_rejected_logps: shape ``(batch,)`` — pi_ref log-probs on rejected.
        beta: KL anchoring strength (higher = stay closer to the reference).

    Returns:
        Tuple ``(losses, chosen_rewards, rejected_rewards)``:
        - ``losses``: shape ``(batch,)`` — per-row DPO loss. The caller
          takes ``.mean()`` to get the scalar batch loss.
        - ``chosen_rewards``: shape ``(batch,)`` —
          ``beta * (policy_chosen_logps - reference_chosen_logps).detach()``.
          Used for logging only — does NOT feed the optimiser.
        - ``rejected_rewards``: shape ``(batch,)`` — same on the rejected side.
    """
    # <YOUR CODE HERE>
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    losses = -F.logsigmoid(logits)

    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()

    return losses, chosen_rewards, rejected_rewards

# --------------------------------------------------------------------------- #
# Self-check — run with: python -m tasks.task2_dpo_loss                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hand-checkable fixture: beta=0.1, batch of 3.
    #   pcl-prl = [3, -1, 4],   rcl-rrl = [1, -3, 4],   delta = [2, 2, 0]
    #   logits  = beta * delta = [0.2, 0.2, 0.0]
    #   losses  = -log sigmoid(logits) = [0.598139, 0.598139, 0.693147]
    #   chosen_r   = beta * (pcl - rcl).detach() = 0.1 * [ 1,  1,  1] = [0.1,  0.1, 0.1]
    #   rejected_r = beta * (prl - rrl).detach() = 0.1 * [-1, -1,  1] = [-0.1,-0.1, 0.1]
    torch.manual_seed(0)
    pcl = torch.tensor([-12.0, -8.0,  -6.0])
    prl = torch.tensor([-15.0, -7.0, -10.0])
    rcl = torch.tensor([-13.0, -9.0,  -7.0])
    rrl = torch.tensor([-14.0, -6.0, -11.0])

    losses, cr, rr = dpo_loss(pcl, prl, rcl, rrl, beta=0.1)
    assert losses.shape == (3,) and cr.shape == (3,) and rr.shape == (3,), \
        f"shapes wrong: {losses.shape=}, {cr.shape=}, {rr.shape=}"
    assert torch.allclose(cr, torch.tensor([ 0.1, 0.1, 0.1])), f"chosen_rewards wrong: {cr}"
    assert torch.allclose(rr, torch.tensor([-0.1,-0.1, 0.1])), f"rejected_rewards wrong: {rr}"
    expected_loss = torch.tensor([0.598139, 0.598139, 0.693147])
    assert torch.allclose(losses, expected_loss, atol=1e-4), \
        f"loss wrong: got {losses}, expected {expected_loss}"

    # Rewards must NOT carry gradient (the optimiser only ever sees losses).
    pcl_g = torch.tensor([-12.0], requires_grad=True)
    prl_g = torch.tensor([-15.0], requires_grad=True)
    _, cr_g, rr_g = dpo_loss(pcl_g, prl_g, torch.tensor([-13.0]),
                             torch.tensor([-14.0]), beta=0.1)
    assert not cr_g.requires_grad, "chosen_rewards must be detached"
    assert not rr_g.requires_grad, "rejected_rewards must be detached"

    print("dpo_loss: all checks passed")
