"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

from typing import Sequence

import re
from collections import Counter

import torch
from detoxify import Detoxify


_DETOX = None

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "with", "you",
    "your",
}

_GENERIC_REFUSALS = (
    "i don't understand",
    "i do not understand",
    "i cannot help",
    "i can't help",
    "i cannot assist",
    "i can't assist",
    "i'm sorry",
    "provide more context",
    "not programmed to",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content_words(text: str) -> set[str]:
    return {word for word in _words(text) if word not in _STOPWORDS}


def _repetition_ratio(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def _detox_scores(texts: Sequence[str]) -> list[float]:
    global _DETOX

    if not texts:
        return []

    if _DETOX is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETOX = Detoxify("original", device=device)

    predictions = _DETOX.predict(list(texts))["toxicity"]
    return [float(value) for value in predictions]


def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Reward benign, relevant, varied completions without template collapse."""
    if prompts is None:
        prompts = [""] * len(texts)
    if len(prompts) != len(texts):
        raise ValueError("texts and prompts must have the same length")

    toxicities = _detox_scores(texts)
    normalized = [" ".join(_words(text)) for text in texts]
    frequencies = Counter(normalized)

    rewards: list[float] = []

    for text, prompt, normalized_text, toxicity in zip(
        texts, prompts, normalized, toxicities
    ):
        words = _words(text)

        # Saturate once the response is clearly benign.
        detox_reward = (
            1.0
            if toxicity <= 0.10
            else max(0.0, 1.0 - (toxicity - 0.10) / 0.90)
        )

        prompt_words = _content_words(prompt)
        response_words = _content_words(text)
        if prompt_words:
            overlap = len(prompt_words & response_words) / len(prompt_words)
            relevance_bonus = 0.15 * min(overlap / 0.25, 1.0)
        else:
            relevance_bonus = 0.0

        repetition_penalty = 0.60 * _repetition_ratio(words)

        # Approximation of hitting the 64-token response cap.
        token_count = len(re.findall(r"\w+|[^\w\s]", text))
        length_penalty = 0.25 if token_count >= 60 else 0.0

        # Penalise one completion template repeated across the rollout batch.
        duplicate_penalty = 0.0
        if len(texts) > 1 and normalized_text:
            duplicate_penalty = 0.35 * (
                (frequencies[normalized_text] - 1) / (len(texts) - 1)
            )

        empty_penalty = 0.50 if len(words) < 3 else 0.0

        generic_refusal_penalty = (
            0.35
            if len(words) <= 12
            and any(phrase in normalized_text for phrase in _GENERIC_REFUSALS)
            else 0.0
        )

        reward = (
            detox_reward
            + relevance_bonus
            - repetition_penalty
            - length_penalty
            - duplicate_penalty
            - empty_penalty
            - generic_refusal_penalty
        )
        rewards.append(float(reward))

    return rewards


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
