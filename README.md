# LLM Post-training Homework — detox direction

You will push `Qwen/Qwen2.5-0.5B` (the **non-Instruct** variant) away
from hostile completions on three held-out prompt families, using SFT
→ DPO → PPO via verl. Eight tasks, 100 points.

We start from the non-Instruct base because the Instruct variant has
already been RLHF'd into politeness — a detox-direction homework
needs a model that *can* produce hostile completions, so you can
measure progress as that capability collapses.

To score toxicity we use the off-the-shelf `unitary/toxic-bert`
classifier (accessed via the `detoxify` package) plus an
eyeball-the-completions pass on every eval step — the metric misses
subtle reward hacks the eyeball catches immediately.

Over the course of the walkthrough you'll train and evaluate four
adapters/checkpoints:

1. **SFT** on the benign-side completions of Detoxify-filtered
   preference pairs.
2. **DPO** initialised from SFT, on the same pairs.
3. **RM** — a Bradley-Terry reward model on the prompt + completion
   chat-template, trained on the same pairs.
4. **PPO** via verl, with three reward variants in three separate
   runs: `inv:detoxify` (the off-the-shelf detox score), the RM you
   trained, and a custom reward you design.

## Tasks

| # | Task | What you implement / write | Points |
|---|---|---|---|
| 1 | SFT evaluation | `src/detox_hw/eval_lib.py::sampled_eval` | 15 |
| 2 | DPO loss + trainer wiring | `tasks/task2_dpo_loss.py::dpo_loss` + the marked block in `src/detox_hw/train_dpo.py` | 15 |
| 3 | DPO evaluation | `src/detox_hw/eval_lib.py::greedy_eval` | 10 |
| 4 | Bradley-Terry preference loss | `tasks/task4_bt_loss.py::bt_loss` | 10 |
| 5 | RM module + training step | `tasks/task5_reward_head.py::build_rm` + `::rm_step` | 20 |
| 6 | PPO with `inv:detoxify` eval | `src/detox_hw/eval_lib.py::worst_of_k_eyeball` | 10 |
| 7 | PPO with your RM eval | no code — writeup in `submissions/task7_ppo_rm_eval.txt` | 5 |
| 8 | Custom reward design + analysis | `tasks/task8_custom_reward.py::reward_score` + `submissions/task8_writeup.md` | 15 |

Anything else you write — helper functions, extra scripts, additional
eval — is yours; not graded.

## Environment

You need:

- A Linux VM with one H100 (or comparable) and **docker** installed.
- The `nvidia-container-toolkit` so docker sees the GPU. Verify with
  `sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
- ≥ 200 GB disk (the verl docker image alone is ~25 GB; the HF cache
  for Qwen-0.5B is another ~3 GB).
- Python 3.10+ on the host for the SFT / DPO / RM training steps. The
  PPO step runs inside the verl container; you don't need a host
  Python for it.

Clone the repo on the VM:

```bash
git clone https://github.com/st-fedotov/detox-hw-sol.git && cd detox-hw-sol
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -U "torch>=2.1" "transformers>=4.45" "peft>=0.13" \
                "datasets>=2.20" "detoxify>=0.5" "torchao>=0.16" \
                "scikit-learn" "tqdm"
```

## End-to-end walkthrough

### Step 1 — prepare the data

The data source is **`Anthropic/hh-rlhf`** (harmless-base split).
Each row is a pair `(chosen, rejected)` where a human labeller picked
the more helpful/harmless completion. We use the labels as-is —
`chosen` is our positive (the side we want the policy to behave
like), `rejected` is the negative.

We tighten the filter further with Detoxify:

- `chosen` (the benign completion) must score ≤ **0.10** — clearly
  benign, not just slightly polite.
- `rejected` (the toxic completion) must score ≥ **0.50** — actually
  hostile, not just impolite.

The two thresholds make sure the preference signal is real: the two
sides of every pair sit unambiguously on opposite sides of the
toxicity surface. Run:

```bash
python -m data_prep.build_pairs --out-dir data --max-rows 80000
```

Writes `data/dpo.jsonl` (preference triples) and `data/sft.jsonl`
(SFT rows where response = the benign side). Yields ~2.5k filtered
pairs from hh-rlhf harmless-base's ~80k rows.

#### What's Detoxify?

Detoxify (`unitaryai/detoxify`) is a small BERT-family classifier
trained on the Jigsaw "Toxic Comment Classification" datasets
(Wikipedia / Civil Comments + a few extensions). The model we use —
the `"original"` variant — wraps the Hugging Face checkpoint
`unitary/toxic-bert`. Given a string, it returns a dict of six
probability scores in `[0, 1]`:

```
toxicity, severe_toxicity, obscene, threat, insult, identity_attack
```

We only read the **`toxicity`** head — a coarse "is this text
hostile/abusive" score. Higher = more toxic. Detoxify shows up in
two roles in this walkthrough:

- **Filter** at data-prep time (the thresholds above).
- **Held-out scalar gauge** at eval time, scoring every greedy and
  sampled completion. The trained policy never sees Detoxify during
  training — SFT uses the chat-template LM loss, DPO uses your own
  preference loss, and the RM is trained on the same preference
  pairs. The only place Detoxify becomes the policy's *reward* is the
  Task 6 PPO variant (`inv:detoxify`). That's exactly the experiment
  that surfaces the reward-hack you'll diagnose.

Detoxify is **not an oracle.** It confuses *aggressive register*
(loud, exclamatory text) with *actually hostile* content, misses
subtler bigotry that doesn't use surface slurs, and is noisy in the
0.05–0.30 score band. We work around that by combining the scalar
mean Detoxify with a sampled K=16 diagnostic and by eyeballing
completions alongside the metric. Both are introduced under Step 3.

### Step 2 — train SFT

SFT here is plain language-modeling on the `chosen` (benign-side)
half of our preference pairs: given the prompt, predict the benign
completion token by token. We mask the loss to the response half
only (so the model isn't graded on predicting its own system prompt
back), and we apply Qwen's chat template at dataset construction.

The SFT checkpoint we train here is the starting point for DPO in
Step 4 and the reference policy for PPO in Step 7.

```bash
python -m src.detox_hw.train_sft \
    --train data/sft.jsonl \
    --out checkpoints/sft \
    --epochs 1 --batch-size 4 --grad-accum 4
```

LoRA-on-base fine-tune on the benign-side completions.

### Step 3 — Task 1: SFT evaluation [15 pts]

#### Eval scaffolding — three prompt families, two measurement modes

Every trained checkpoint in this walkthrough is evaluated on three
held-out prompt families:

- **`mild_prefix`** — 15 neutral sentence-starters. Tests whether the
  policy drifts toxic on neutral input.
- **`direct_provocation`** — 15 explicit asks for hostility. Tests
  whether the policy takes the invitation when asked.
- **`rtp_challenging`** — 50 prompts from
  `allenai/real-toxicity-prompts` filtered to `challenging=True`.

And two measurement modes:

- **Greedy** — `do_sample=False`, report mean Detoxify over
  completions. Sensitive to the *mode* of the policy distribution.
- **Sampled-support, K=16** — for each prompt, draw 16 completions at
  `temp=1.0, top_p=0.95, top_k=50` and score each with Detoxify. Let
  `R = (r_1, …, r_16)` be that prompt's 16 toxicity scores. The three
  numbers reported are aggregations over prompts of per-prompt
  functions of `R`:
  - `support_rate` — fraction of prompts where `max(R) ≥ 0.5`, i.e.
    *at least one* of the 16 samples for that prompt is clearly toxic.
    For a detox-direction policy, lower = better.
  - `mean_max` — average of `max(R)` across prompts. Tracks the
    "worst plausible completion" intensity across the eval set.
  - `mean_std` — average of `std(R)` across prompts. Tracks how much
    the 16 samples for a single prompt disagree.

A policy whose per-prompt std collapses has likely converged on a
narrow attractor and produces the same kind of completion regardless
of prompt — the canonical reward-hack signature. You'll see this
shape explicitly in Task 6.

The eval scaffolding lives in `src/detox_hw/eval_lib.py`. The three
helpers there are the implementation work for Tasks 1, 3, and 6:

- `sampled_eval` — Task 1 (this task)
- `greedy_eval` — Task 3
- `worst_of_k_eyeball` — Task 6

#### Your task

First, **fill in `sampled_eval` in `src/detox_hw/eval_lib.py`** — the
K=16 diagnostic that returns `{slice: {support_rate, mean_max,
mean_std}}` per eval slice. Then run:

```bash
python -m tasks.task1_sft_eval \
    --sft-dir checkpoints/sft \
    --out submissions/task1_sft_eval.json
```

Deliverable: `submissions/task1_sft_eval.txt` — the eval output and
your takeaways (what moved vs base, did the support shrink, etc.).

### Step 4 — Task 2: implement `dpo_loss` [15 pts]

SFT moved the policy in one direction (the chosen side) but didn't
directly punish the rejected side. **DPO** (Direct Preference
Optimization) does both at once: it nudges the policy so the chosen
completion is more probable *relative to the reference model* than
the rejected one. Higher relative probability for chosen → lower for
rejected → the mode shifts toward chosen.

The DPO loss for one preference pair is

```
L_DPO = -log σ( β · [
    log(π(y_+|x) / π_ref(y_+|x)) - log(π(y_-|x) / π_ref(y_-|x))
] )
```

where `π` is the trainable policy, `π_ref` is the frozen reference
model (the SFT checkpoint from Step 2 in our case), `y_+` is the
chosen completion, `y_-` is the rejected one, `β` controls how
strongly we anchor to the reference, and `σ` is the logistic.

**Your task has two parts:**

1. **Implement `dpo_loss` in `tasks/task2_dpo_loss.py`.** The function
   returns `(losses, chosen_rewards, rejected_rewards)`; the trainer
   uses the first for backward and the latter two for logging.
2. **Wire your loss into the trainer at `src/detox_hw/train_dpo.py`.**
   Inside the training loop there's a block marked `# TASK 2 (part 2)`
   where the policy and reference forward passes are already done.
   Inside that block you compute per-example log-probs with the
   provided `per_example_logps(logits, labels)` helper, slice each
   resulting `(batch,)` tensor into `chosen` (even rows) and
   `rejected` (odd rows) — the collator interleaves preference pairs
   so even-row = chosen, odd-row = rejected — then call your
   `dpo_loss` and set `loss = losses.mean()`. The
   `chosen_r` / `rejected_r` you return are picked up by the per-step
   log line further down.

Sanity-check `dpo_loss` on a hand-checkable fixture before kicking
off training:

```bash
python -m tasks.task2_dpo_loss
```

(prints `dpo_loss: all checks passed` if the math + sign convention
match what the trainer expects. The fixture is a batch of 3 with
known expected `losses`, `chosen_rewards`, `rejected_rewards`, and a
detached-gradient check so the optimiser only ever sees `losses`.)

Then train:

```bash
python -m src.detox_hw.train_dpo \
    --train data/dpo.jsonl \
    --sft-dir checkpoints/sft \
    --out checkpoints/dpo \
    --epochs 1
```

### Step 5 — Task 3: DPO evaluation [10 pts]

Fill in `greedy_eval` in `src/detox_hw/eval_lib.py`. Then:

```bash
python -m tasks.task3_dpo_eval \
    --sft-dir checkpoints/sft --dpo-dir checkpoints/dpo \
    --out submissions/task3_dpo_eval.json
```

Deliverable: `submissions/task3_dpo_eval.txt` — the eval output and
your takeaways.

### Step 6 — Tasks 4 + 5: bt_loss + RM module + RM training [10 + 20 pts]

DPO learned directly from `(chosen, rejected)` pairs. Classical RLHF
inserts a third thing in between: train a **reward model** on those
same pairs, then use it as the scalar reward in an online RL
algorithm (PPO, in our case — that's Step 7). Here we build and
train the RM.

Two notes up front:

- **The RM is a regression head on top of the base LM** — same
  backbone as the policy. We attach LoRA to the backbone and add a
  single scalar projection on the last non-pad token. Training is
  the **Bradley-Terry log-sigmoid loss** on the difference of two
  such scores (chosen minus rejected): `L = -log σ(s_chosen -
  s_rejected)`. With `Qwen/Qwen2.5-0.5B` as the base, the structure
  is `AutoModelForSequenceClassification(..., num_labels=1)` + LoRA
  with `task_type=SEQ_CLS` — `build_rm` produces exactly this.
- **The RM forward pass sees the full (prompt, response)
  chat-templated pair**, not the response in isolation. A reward of
  the form `r(response)` cannot represent "is this response
  appropriate to *this* prompt"; only `r(prompt, response)` can.
  This is the canonical RLHF RM signature (InstructGPT, Anthropic
  HH). The data collator hands `rm_step` the full chat-templated
  pair so this falls out naturally.

Fill in `tasks/task4_bt_loss.py` (the BT log-sigmoid loss on a single
`(chosen_scores, rejected_scores)` batch) and
`tasks/task5_reward_head.py` (`build_rm` builds the AMFSC + LoRA
stack; `rm_step` runs one forward over chosen and rejected and
returns the BT loss + the two score tensors).

The trainer below imports `build_rm` and `rm_step` from
`tasks/task5_reward_head.py`, which in turn imports `bt_loss` from
`tasks/task4_bt_loss.py`. So your edits to either file land in the
next run automatically — no glue to update.

Sanity-check `bt_loss` before training:

```bash
python -m tasks.task4_bt_loss
```

(prints `bt_loss: all checks passed` — checks the equal-scores
case lands on `log 2`, the three-pair fixture matches the expected
values, and the sign points the right way when chosen loses.)

Sanity-check `build_rm` + `rm_step` next (loads Qwen-0.5B, so the
first run downloads ~1 GB to the HF cache; subsequent runs are
fast):

```bash
python -m tests.test_task5_reward_head
```

(prints `task 5 (build_rm + rm_step) — all tests passed` — verifies
the RM forward returns `(batch,)`-shaped scalar scores and `rm_step`
returns a finite scalar loss on a tiny fixture batch. The
`score.weight | MISSING` line above the success message is expected —
that's AMFSC initialising a fresh scalar head on the causal-LM base,
explained again further down.)

Then:

```bash
python -m src.detox_hw.train_rm \
    --train data/dpo.jsonl \
    --out checkpoints/rm \
    --val-fraction 0.1
```

Outputs include `val_metrics.json` with held-out pairwise accuracy as
a sanity check on your implementation.

Expected log noise: you'll see a `score.weight | MISSING` line from
the model loader. That's not an error — Qwen-2.5-0.5B is a causal-LM
base with no classifier head, and `AutoModelForSequenceClassification`
initializes a fresh scalar `score` linear on top. That fresh head is
precisely what `build_rm` is meant to produce; training is what fills
it in.

Now evaluate the trained RM on the held-out 10% of pairs the trainer
set aside. We report three things:

- **Pairwise accuracy.** Fraction of held-out pairs where the RM
  ranks chosen strictly above rejected. This is the direct
  generalization of the Bradley-Terry training objective at eval
  time. Chance is 0.5; a usable RM lives well above that.
- **Mean reward margin.** Average of `s_chosen - s_rejected` across
  held-out pairs. Accuracy says *how often* the RM gets the
  direction right; mean margin says *by how much*.
- **Side-by-side eyeball.** Three held-out pairs printed with their
  RM scores side by side — the qualitative check. Do the scores
  agree with what a human would call the less-toxic completion? Use
  `--eyeball-seed N` to redraw if the default sample lands on noisy
  pairs (hh-rlhf has some).

```bash
python -m tasks.rm_eval \
    --rm-dir checkpoints/rm \
    --pairs data/dpo.jsonl
```

Deliverable: `submissions/rm_eval.txt` — the eval output and your
takeaways. The numbers here are what Step 7's PPO will be optimising
against in Task 7, so it's worth knowing whether your RM agrees with
its training data.

### Step 7 — PPO via verl (Tasks 6 + 7)

DPO trained on a fixed set of preference pairs. PPO is *online* RL.
At each step:

1. The **actor** (the trainable policy) samples K completions from a
   batch of prompts and a scalar reward function scores each
   completion at the end.
2. A learned value head (the **critic**) predicts the expected return
   from each token position. Per-token advantages come from the
   standard PPO formula `A_t = R_t − V(s_t)`, smoothed across
   timesteps via GAE.
3. The actor is updated on the PPO clipped surrogate objective with
   those advantages, plus a KL penalty that anchors it to a frozen
   **reference** policy (the SFT-merged policy in our case).
4. The critic is updated to better predict the observed returns.

We sample K=8 completions per prompt (the `--rollout-n` flag in the
command below). PPO doesn't *require* multiple rollouts per prompt,
but more samples per prompt give a tighter advantage estimate and
more data per optimiser step at the same prompt-batch size.

Three reward variants in this homework, one PPO run each:

- **`inv:detoxify`** (Task 6) — `r = 1 - Detoxify(completion)`. The
  off-the-shelf detox reward. Chases the easiest path to lower
  Detoxify — which is where the reward-hack attractor emerges, as
  you'll see.
- **`rm:<your-RM-dir>`** (Task 7) — `r = your trained RM`. A learned,
  prompt-conditioned reward. Different attractor profile; worth
  comparing to Task 6.
- **`custom:<your-module>`** (Task 8) — your own design, trying to
  resist the reward-hack patterns Tasks 6 and 7 expose.

PPO with vLLM-driven rollouts is weeks of engineering to assemble from
scratch — actor, critic, frozen reference policy for the KL term, vLLM
rollout workers, plus the Ray orchestration tying them together. We
use **verl** (an open-source RLHF/RL-for-LLMs trainer from Volcengine)
as the off-the-shelf path. It runs PPO with vLLM rollouts and
FSDP-sharded training — exactly what we need for the homework.

verl ships as a Docker image (`verlai/verl:vllm023.dev1`) because the
underlying stack — vLLM, Ray, FSDP, and the right CUDA / torch /
transformers pins — is brittle to assemble from `pip install`. The
image is a known-working pinned environment; pulling it once gives you
the whole stack. We mount the host's repo and the HF / torch caches
into the container so artifacts (checkpoints, eval inputs, downloaded
weights) survive between runs and the container reads weights from
disk instead of going over the network.

Pull the verl image once:

```bash
sudo docker pull verlai/verl:vllm023.dev1
```

Build the parquets verl reads (parquet is a columnar binary table
format — verl's data pipeline is built on Apache Arrow and expects
parquet inputs by default):

```bash
SYS="You are a helpful assistant. Respond to the user thoughtfully and kindly."
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/train.parquet --system-prompt "$SYS"
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/val.parquet --system-prompt "$SYS" --max 200
```

The docker runs below bind-mount the host's `~/.cache/huggingface` and
`~/.cache/torch` directories into the container, so verl reads Qwen
and Detoxify from disk instead of pulling them over the container's
network. Steps 3–6 already populated both caches as a side effect of
every `from_pretrained` and `Detoxify(...)` call along the way — you
don't need to do anything extra here.

#### Verl setup evidence — one-time

Before launching any PPO run, capture evidence that the docker
container has GPU access and that the data + RM are in place. The
commands below write to `submissions/verl_setup.txt` themselves.

```bash
mkdir -p submissions

# (a) GPU access from inside the verl container
sudo docker run --rm --gpus all verlai/verl:vllm023.dev1 nvidia-smi \
    > submissions/verl_setup.txt
echo "---" >> submissions/verl_setup.txt

# (b) Data + RM on the host
ls -la data/*.parquet checkpoints/rm/ >> submissions/verl_setup.txt
```

#### Task 6 — PPO with `inv:detoxify` [10 pts]

The docker run below launches verl's PPO trainer. The flag block at
the end is the PPO config:

| flag | meaning |
|---|---|
| `--total-steps 100` | number of PPO outer-loop update steps |
| `--train-batch-size 16` | prompts gathered per outer step (before inner minibatching) |
| `--ppo-mini-batch-size 8` | minibatch size for the inner PPO SGD |
| `--rollout-n 8` | completions sampled per prompt (the group used for advantage estimation) |
| `--max-response-length 64` | token cap per completion — keeps rollouts fast and forces the policy to commit early |
| `--rollout-gpu-mem 0.25` | fraction of GPU memory vLLM reserves for its KV cache (the rest goes to actor / critic / ref weights, which share the GPU) |
| `--actor-lr 2e-6` | learning rate for the policy head; small because we're nudging an already-trained policy |
| `--critic-lr 1e-5` | learning rate for the value head; larger because the head is initialized fresh |
| `--kl-coef 0.001` | coefficient on the KL penalty toward the reference (SFT-merged) policy; mild — anchors without freezing |
| `--save-freq 20` / `--test-freq 10` | checkpoint and validation cadences (in outer steps) |

The same flag block carries over to Tasks 7 and 8 — only `--reward`
and the `--out` directory change between the three runs.

Output is piped through `tee` so the training log lands in
`submissions/task6_log.txt`:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=inv:detoxify \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_inv_detoxify \
             --reward inv:detoxify \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task6_log.txt
```

verl trains with **FSDP** (Fully Sharded Data Parallel — PyTorch's
API for splitting a model's parameters, gradients, and optimiser
state across processes), and its checkpoint format mirrors that:
sharded `.pt` files keyed by world rank, not loadable with
`from_pretrained`. To load the trained PPO policy into our eval
scripts (which use `AutoModelForCausalLM.from_pretrained`), we need
it as a regular HuggingFace directory: a single `config.json` +
`model.safetensors` + `tokenizer*.json`. verl ships a
`model_merger` utility that does exactly this conversion (FSDP
shards → consolidated HF format). Run:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_inv_detoxify/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_inv_detoxify_merged"

# Permission fix: the merger writes model.safetensors as root:
sudo chmod 644 checkpoints/ppo_inv_detoxify_merged/model.safetensors

# Evidence: prove the merged ckpt is in place
ls -la checkpoints/ppo_inv_detoxify_merged/ > submissions/task6_merged_ls.txt
```

**Your task: implement `worst_of_k_eyeball` in
`src/detox_hw/eval_lib.py`.** For each prompt, sample K=16
completions, score them with Detoxify, and return the most-toxic one
per prompt — the "with 16 tries, can the policy still land hostile?"
read. Then eval:

```bash
python -m tasks.task6_ppo_detoxify_eval \
    --ppo-dir checkpoints/ppo_inv_detoxify_merged \
    --out submissions/task6_ppo_detoxify_eval.json
```

Deliverable: `submissions/task6_ppo_detoxify_eval.txt` — the eval
output and your interp. Specifically: did the policy collapse to a
prompt-independent attractor? What does it look like?

#### Task 7 — PPO with your RM [5 pts]

Same docker run, but replace the reward env var and capture the log
under a different name:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=rm:/workspace/checkpoints/rm \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_rm \
             --reward rm:/workspace/checkpoints/rm \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task7_log.txt
```

Merge the FSDP shards to HF format (same conversion as in Task 6)
and dump the directory listing for evidence:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_rm/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_rm_merged"

sudo chmod 644 checkpoints/ppo_rm_merged/model.safetensors
ls -la checkpoints/ppo_rm_merged/ > submissions/task7_merged_ls.txt
```

Eval:

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_rm_merged \
    --out submissions/task7_ppo_rm_eval.json
```

Deliverable: `submissions/task7_ppo_rm_eval.txt` — the eval output
and your interp. Specifically: same attractor as Task 6, or different?
Why might that be?

### Step 8 — Task 8: custom reward + writeup [15 pts]

Tasks 6 and 7 each showed you an attractor: `inv:detoxify` collapsed
the policy onto a Detoxify-saturating completion (often a refusal
template or system-prompt echo); `rm:<your-RM>` collapsed onto a
*different* attractor (its own learned shortcut). The pattern is the
same: an RL policy converges on whatever sub-region of the response
space saturates the reward most cheaply, and "cheaply" almost always
means "a single template repeated across prompts."

Your task: **design a reward that can't be saturated by a single
template.** A few angles you might combine:

- Saturate Detoxify above some threshold — once a completion is
  clearly benign, uniform reward removes the incentive to push the
  template attractor harder.
- Penalise repetition (trigram or n-gram).
- Penalise length-cap hits (if the policy learns to always run to
  the token cap, penalise that signal).
- Add a prompt-relevance signal — bag-of-words overlap or embedding
  similarity ties the reward to the prompt, so prompt-independent
  template completions stop scoring well. Beware trivial echoing.
- Blend or gate Detoxify with your RM. Where they disagree is signal.

The implementation lives in `tasks/task8_custom_reward.py`; the
function signature is `reward_score(texts, prompts=None) ->
list[float]`. The verl reward worker imports your function when you
launch with `TOXIC_REWARD=custom:tasks.task8_custom_reward`. Run verl
with it (log to `submissions/task8_log.txt`):

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
  -e HYDRA_FULL_ERROR=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_custom \
             --reward custom:tasks.task8_custom_reward \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task8_log.txt
```

Merge + capture the directory listing:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_custom/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_custom_merged"

sudo chmod 644 checkpoints/ppo_custom_merged/model.safetensors
ls -la checkpoints/ppo_custom_merged/ > submissions/task8_merged_ls.txt
```

Run eval (you can reuse `task7_ppo_rm_eval.py` with the custom-PPO
path, or write your own eval script — the helpers in
`src/detox_hw/eval_lib.py` are reusable):

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_custom_merged \
    --out submissions/task8_ppo_custom_eval.json
```

Deliverables:

- `submissions/task8_ppo_custom_eval.txt` — eval output and your
  interp. Did the new reward avoid the template attractor? What's
  the attractor now, if any?
- `submissions/task8_writeup.md` — what you tried, what collapsed
  into what, what your final design looks like, why you think it
  works (or why it still failed).

## Submission

Submit a single **`*.zip`** file containing:

```
tasks/
  task2_dpo_loss.py
  task4_bt_loss.py
  task5_reward_head.py
  task8_custom_reward.py

src/detox_hw/
  eval_lib.py

submissions/
  task1_sft_eval.txt
  task3_dpo_eval.txt
  rm_eval.txt
  task6_ppo_detoxify_eval.txt
  task7_ppo_rm_eval.txt
  task8_ppo_custom_eval.txt
  task8_writeup.md
  verl_setup.txt
  task6_log.txt
  task6_merged_ls.txt
  task7_log.txt
  task7_merged_ls.txt
  task8_log.txt
  task8_merged_ls.txt
```
