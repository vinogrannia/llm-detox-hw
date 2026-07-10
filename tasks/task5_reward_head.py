"""Task 5 — Build the reward model + write the training step [20 points].

Two pieces:

``build_rm`` — instantiate the RM (AMFSC + LoRA), ready for BT training.
``rm_step`` — one forward pass through the RM for the chosen and
              rejected sides of a preference batch, returning the
              Bradley-Terry loss (your ``bt_loss``) and the two score
              tensors for logging.

``train_rm.py`` calls these. The provided loop handles data, optimiser,
gradient clipping, and saving.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from tasks.task4_bt_loss import bt_loss


def build_rm(
    base_name: str = "Qwen/Qwen2.5-0.5B",
    pad_token_id: int | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Build the reward model: AMFSC backbone + LoRA + scalar head.

    The structure you want:

    * ``AutoModelForSequenceClassification.from_pretrained(base_name,
      num_labels=1, dtype=torch.float32)``. ``num_labels=1`` gives the
      single scalar reward; ``dtype=fp32`` avoids the Qwen-0.5B bf16
      NaN cliff at the classifier head.
    * Set ``model.config.pad_token_id = pad_token_id``. Without this
      AMFSC silently pools the token at index 0 instead of the last
      non-pad token, and the reward signal collapses.
    * Wrap with ``peft.get_peft_model(model, LoraConfig(
      task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
      lora_dropout=lora_dropout, target_modules=...))``. ``SEQ_CLS``
      task type tells peft to keep the scalar head trainable while
      freezing the backbone except for the LoRA deltas.

    LoRA target-modules choice: ``"all-linear"`` is the easy default
    (peft auto-discovers every Linear layer and wraps it). Picking
    only the attention projections — for Qwen 2.5 that's
    ``["q_proj", "k_proj", "v_proj", "o_proj"]`` — is more
    parameter-efficient at the cost of capacity. Rank, alpha, and
    dropout are yours to tune.

    Args:
        base_name: HuggingFace model id for the base LM.
        pad_token_id: ``tokenizer.pad_token_id`` — required for correct
            last-non-pad-token pooling.
        lora_r, lora_alpha, lora_dropout: LoRA hyperparameters.

    Returns:
        A ``peft.PeftModel`` wrapping the AMFSC. The forward returns a
        ``SequenceClassifierOutput`` whose ``.logits`` has shape
        ``(batch, 1)`` (squeeze the last dim to get ``(batch,)`` scores).
    """
    # <YOUR CODE HERE>
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=1,
        dtype=torch.float32,
    )
    model.config.pad_token_id = pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
    )

    return get_peft_model(model, lora_config)


def rm_step(rm, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One RM training step's forward pass.

    Run the model on both the chosen and the rejected side of the
    preference batch, get scalar scores per row, and return the BT
    loss + the two score tensors.

    Args:
        rm: the model from ``build_rm``.
        batch: dict with keys ``chosen_ids``, ``chosen_attn``,
            ``rejected_ids``, ``rejected_attn``, all already on the
            right device.

    Returns:
        Tuple ``(loss, chosen_scores, rejected_scores)``:
        - ``loss``: scalar tensor — mean of ``bt_loss(chosen_scores,
          rejected_scores)``. ``train_rm.py`` calls ``.backward()``,
          gradient-clip, and ``optimiser.step()`` around this.
        - ``chosen_scores``: shape ``(batch,)`` — RM score per chosen row.
        - ``rejected_scores``: shape ``(batch,)`` — RM score per rejected row.

    Hint: AMFSC returns ``.logits`` of shape ``(batch, num_labels)``;
    with ``num_labels=1`` you ``.squeeze(-1)`` to get ``(batch,)``.
    Import ``bt_loss`` from ``tasks.task4_bt_loss`` (already imported
    at the top of this file).
    """
    # <YOUR CODE HERE>
    chosen_scores = rm(
        input_ids=batch["chosen_ids"],
        attention_mask=batch["chosen_attn"],
    ).logits.squeeze(-1)

    rejected_scores = rm(
        input_ids=batch["rejected_ids"],
        attention_mask=batch["rejected_attn"],
    ).logits.squeeze(-1)

    loss = bt_loss(chosen_scores, rejected_scores).mean()

    return loss, chosen_scores, rejected_scores
