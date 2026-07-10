"""DPO trainer — imports your ``dpo_loss`` from ``tasks/task2_dpo_loss.py``.

Two-model architecture: a frozen SFT-merged reference, and a fresh
LoRA-adapter-on-top-of-SFT policy. Each preference row becomes a pair
of tokenised sequences (chosen and rejected). Loss is masked to the
completion half so the DPO term sees only completion log-probs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.io import read_jsonl
from src.detox_hw.train_sft import IGNORE_INDEX, LORA_TARGETS_ALL, chat_prompt_ids, cosine_lr
from tasks.task2_dpo_loss import dpo_loss


# --------------------------------------------------------------------------- #
# Dataset.                                                                    #
# --------------------------------------------------------------------------- #


class DpoDataset(Dataset):
    """Each row → two tokenised sequences (chosen and rejected).
    Collator interleaves them into a (2*batch, T) tensor."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 384):
        self.items: list[dict] = []
        for p in pairs:
            prompt_ids = chat_prompt_ids(tokenizer, p["prompt"])
            def one_side(text: str) -> dict:
                resp_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    resp_ids = resp_ids + [tokenizer.eos_token_id]
                full = prompt_ids + resp_ids
                if len(full) > max_length:
                    cut = len(full) - max_length
                    full = full[cut:]
                    prompt_kept = max(0, len(prompt_ids) - cut)
                else:
                    prompt_kept = len(prompt_ids)
                labels = [IGNORE_INDEX] * prompt_kept + list(full[prompt_kept:])
                return {"input_ids": full, "labels": labels}
            self.items.append({
                "chosen":   one_side(p["chosen"]),
                "rejected": one_side(p["rejected"]),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def dpo_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    seqs: list[dict] = []
    for item in batch:
        # Even-indexed = chosen, odd = rejected. The trainer slices
        # log-probs back into the two halves.
        seqs.append(item["chosen"])
        seqs.append(item["rejected"])
    max_len = max(len(s["input_ids"]) for s in seqs)
    n = len(seqs)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s["input_ids"])
        ids[i, :L] = torch.tensor(s["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(s["labels"], dtype=torch.long)
        attn[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def per_example_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probs of the labelled tokens per example. Ignored
    tokens contribute zero."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask.float()
    return gathered.sum(dim=-1)


# --------------------------------------------------------------------------- #
# Train.                                                                      #
# --------------------------------------------------------------------------- #


def train(
    pairs: list[dict],
    sft_dir: Path,
    out_dir: Path,
    base_name: str = "Qwen/Qwen2.5-0.5B",
    beta: float = 0.1,
    lr: float = 5e-6,
    batch_size: int = 2,
    grad_accum: int = 8,
    epochs: int = 1,
    lora_r: int = 32,
    log_every: int = 50,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference = SFT (no LoRA trainable on this copy).
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.float32, device_map=device,
    )
    reference = PeftModel.from_pretrained(ref_base, str(sft_dir))
    reference = reference.merge_and_unload().eval()
    for p in reference.parameters():
        p.requires_grad = False

    # Policy = SFT + new LoRA adapter (trainable).
    pol_base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.float32, device_map=device,
    )
    policy = PeftModel.from_pretrained(pol_base, str(sft_dir))
    policy = policy.merge_and_unload()
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=list(LORA_TARGETS_ALL),
        bias="none", task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()
    policy.train()

    ds = DpoDataset(pairs, tokenizer)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: dpo_collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )
    print(f"dpo train: {len(ds)} pairs, {len(loader)} batches/epoch")
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)
    total_micro = len(loader) * epochs
    total_steps = total_micro // grad_accum
    warmup = max(1, int(total_steps * 0.03))

    step = micro = 0
    optim.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            pol_out = policy(input_ids=ids, attention_mask=attn)
            with torch.no_grad():
                ref_out = reference(input_ids=ids, attention_mask=attn)
            # ===== TASK 2 (part 2) — wire your dpo_loss into the trainer =====
            #
            # The collator interleaves preference pairs so even rows are
            # chosen and odd rows are rejected. Using the provided helper
            # ``per_example_logps(logits, labels)``, compute per-example
            # log-probs for the policy and the reference, slice each
            # resulting ``(batch,)`` tensor into chosen/rejected halves,
            # call your ``dpo_loss``, and set ``loss = losses.mean()``.
            #
            # Names the rest of this loop expects after the block:
            #   - ``loss``      — scalar tensor, fed into ``.backward()`` below
            #   - ``chosen_r``  — shape ``(batch/2,)``, for the log line further down
            #   - ``rejected_r``— shape ``(batch/2,)``, same
            # <YOUR CODE HERE>

                        policy_logps = per_example_logps(pol_out.logits, labels)
            reference_logps = per_example_logps(ref_out.logits, labels)

            policy_chosen_logps = policy_logps[0::2]
            policy_rejected_logps = policy_logps[1::2]
            reference_chosen_logps = reference_logps[0::2]
            reference_rejected_logps = reference_logps[1::2]

            losses, chosen_r, rejected_r = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=beta,
            )
            loss = losses.mean()
            
            # ==================================================================
            (loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum == 0:
                cur_lr = cosine_lr(step, total_steps, warmup, lr)
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad()
                step += 1
                if step % log_every == 0:
                    margin = (chosen_r - rejected_r).mean().item()
                    print(json.dumps({
                        "step": step, "of": total_steps,
                        "loss": float(loss.item()),
                        "chosen_r": float(chosen_r.mean().item()),
                        "rejected_r": float(rejected_r.mean().item()),
                        "margin": margin, "lr": cur_lr,
                    }))
    policy.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved DPO adapter to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSONL of {prompt, chosen, rejected} rows")
    p.add_argument("--sft-dir", required=True, help="path to SFT adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=32)
    a = p.parse_args()
    pairs = list(read_jsonl(a.train))
    train(
        pairs, Path(a.sft_dir), Path(a.out), base_name=a.base,
        beta=a.beta, lr=a.lr, batch_size=a.batch_size,
        grad_accum=a.grad_accum, epochs=a.epochs, lora_r=a.lora_r,
    )


if __name__ == "__main__":
    main()
