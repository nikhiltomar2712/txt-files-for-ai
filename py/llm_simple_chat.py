"""Simple LLM-like chat demo (toy implementation).

This repo is a learning collection (txt files + starter code).
This script demonstrates the *shape* of an LLM pipeline:
- tokenization (toy)
- context window
- generation loop

It is NOT a real LLM and does not require heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict


def simple_tokenize(text: str) -> List[str]:
    # Toy tokenizer: split by whitespace, keep punctuation as part of tokens.
    return [t for t in text.strip().split() if t]


def simple_detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


@dataclass
class ToyLLM:
    vocab: List[str]

    def __post_init__(self) -> None:
        self.v2i: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    def next_token_logits(self, context_tokens: List[str]) -> List[float]:
        """Toy logits: encourage tokens that appear in context.

        This makes generation feel "context-aware" without actually doing ML.
        """
        logits = [-10.0] * len(self.vocab)
        for tok in context_tokens[-20:]:
            if tok in self.v2i:
                logits[self.v2i[tok]] = 3.0

        # Add a few generic useful tokens if missing.
        for special in ["the", "a", "to", "and", "I", "you", "AI", "build", "model", "data", "train", "learn"]:
            if special in self.v2i:
                logits[self.v2i[special]] = max(logits[self.v2i[special]], 1.0)

        return logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.9,
        context_window: int = 120,
    ) -> str:
        tokens = simple_tokenize(prompt)
        context = tokens[-context_window:]

        generated: List[str] = []
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(context)

            # Softmax with temperature (safe for toy values)
            t = max(1e-6, temperature)
            exps = [math.exp(l / t) for l in logits]
            s = sum(exps)
            probs = [e / s for e in exps]

            # Greedy sample (still uses probs to pick best)
            idx = max(range(len(probs)), key=lambda i: probs[i])
            next_tok = self.vocab[idx]
            generated.append(next_tok)

            context.append(next_tok)
            if len(context) > context_window:
                context = context[-context_window:]

            # Stop if we hit a sentence-like token
            if next_tok in [".", "?", "!"]:
                break

        return simple_detokenize(tokens + generated)


def build_demo_vocab() -> List[str]:
    base = [
        "I",
        "you",
        "we",
        "AI",
        "artificial",
        "intelligence",
        "learn",
        "learning",
        "model",
        "data",
        "train",
        "training",
        "loss",
        "optimize",
        "build",
        "pipeline",
        "predict",
        "token",
        "context",
        "generate",
        "response",
        "the",
        "a",
        "to",
        "and",
        "for",
        "with",
        "in",
        "on",
        "of",
        "it",
        "is",
        "are",
        "can",
        "will",
        "should",
        ". ",
        ".",
        "?",
        "!",
        "begin",
        "steps",
        "first",
        "next",
        "then",
        "finally",
        "example",
    ]

    # Normalize: remove any token with spaces
    base = [t.strip() for t in base if t.strip()]
    # Add a handful of common words
    base += [
        "how",
        "does",
        "work",
        "explain",
        "create",
        "system",
        "dataset",
        "evaluation",
        "validate",
        "deploy",
        "code",
        "python",
        "js",
        "web",
        "app",
    ]

    # Ensure unique while preserving order
    seen = set()
    vocab = []
    for w in base:
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    return vocab


def main() -> None:
    vocab = build_demo_vocab()
    llm = ToyLLM(vocab=vocab)

    print("Toy LLM Chat Demo (not a real LLM). Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit", "q"}:
            break

        prompt = (
            "You are learning how AI works. "
            "Explain clearly and give a small python-code idea. "
            "User request: "
            f"{user}"
        )

        out = llm.generate(prompt, max_new_tokens=70, temperature=0.8)
        # Print only the continuation portion (toy)
        print("Assistant:", out)
        print()


if __name__ == "__main__":
    main()

