"""llm_project_068.py

Tiny self-contained Python example related to building an LLM project.
Each file focuses on one small concept so the repo has lots of runnable code.
"""

from __future__ import annotations

import math
import random


def set_seed(seed: int = 68) -> None:
    random.seed(seed)


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]


def sample_from_probs(probs: list[float]) -> int:
    r = random.random()
    c = 0.0
    for idx, p in enumerate(probs):
        c += p
        if r <= c:
            return idx
    return len(probs) - 1


def tiny_next_token_predictor(prompt: str = "ai", vocab: str = "abcdefghijklmnopqrstuvwxyz .") -> str:
    """Toy predictor: scores next characters using simple heuristics."""
    set_seed(68)
    # Heuristic score based on prompt length and character position
    base = len(prompt) % max(1, len(vocab))
    scores = []
    for j, ch in enumerate(vocab):
        # deterministic-ish pseudo score
        scores.append(-abs((j - base) % len(vocab)) / 7.0 + (random.random() - 0.5) * 0.15)
    probs = softmax(scores)
    nxt = vocab[sample_from_probs(probs)]
    return nxt


def main() -> None:
    prompt = "ai"
    out = prompt
    for _ in range(40):
        out += tiny_next_token_predictor(out)
    print(out)


if __name__ == "__main__":
    main()
