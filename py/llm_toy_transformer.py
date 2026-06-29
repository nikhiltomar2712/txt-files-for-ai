"""llm_toy_transformer.py

A tiny, dependency-free Transformer-like block and a mini language-model training loop.

Goals:
- show how an LLM pipeline is assembled in code
- include attention, feed-forward, embeddings, and a training step

This is NOT a production model and uses a very small toy dataset by default.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class TrainConfig:
    vocab: str = "abcdefghijklmnopqrstuvwxyz ."
    block_size: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 16
    steps: int = 300
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)


class ToyTokenizer:
    def __init__(self, vocab: str):
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        s = s.lower()
        return [self.stoi.get(ch, self.stoi[" "]) for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def split_batch(data_ids: list[int], batch_size: int, block_size: int):
    # random contiguous segments
    starts = [random.randint(0, len(data_ids) - block_size - 1) for _ in range(batch_size)]
    x = []
    y = []
    for st in starts:
        chunk = data_ids[st : st + block_size + 1]
        x.append(chunk[:-1])
        y.append(chunk[1:])
    return x, y


class ToyTinyLLM:
    """A minimal model implemented with plain Python lists.

    For simplicity (and no external deps), we only implement forward logits using
    NumPy-like operations conceptually but actually do computations in Python.

    NOTE: This is slow for large configs but fine for tiny demos.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        V = len(cfg.vocab)
        C = cfg.embed_dim
        H = cfg.num_heads
        assert C % H == 0
        self.V = V
        self.C = C
        self.H = H
        self.D = C // H
        self.scale = 1 / math.sqrt(self.D)

        # token embedding: V x C
        self.E = [[(random.random() - 0.5) * 0.02 for _ in range(C)] for _ in range(V)]
        # positional embedding: block_size x C
        self.P = [[(random.random() - 0.5) * 0.02 for _ in range(cfg.block_size)] for _ in range(cfg.block_size)]

        # attention projection weights
        def rand_mat(out_dim: int, in_dim: int):
            return [[(random.random() - 0.5) * 0.02 for _ in range(in_dim)] for _ in range(out_dim)]

        # We keep it single-layer: Q,K,V projections and output projection
        self.Wq = rand_mat(C, C)
        self.Wk = rand_mat(C, C)
        self.Wv = rand_mat(C, C)
        self.Wo = rand_mat(C, C)

        # feed-forward
        self.W1 = rand_mat(C * cfg.ff_mult, C)
        self.b1 = [0.0 for _ in range(C * cfg.ff_mult)]
        self.W2 = rand_mat(C, C * cfg.ff_mult)
        self.b2 = [0.0 for _ in range(C)]

        # layer norm params (simplified)
        self.ln_eps = 1e-5
        self.g = [1.0 for _ in range(C)]
        self.b = [0.0 for _ in range(C)]

        self.dropout_p = cfg.dropout

    def layer_norm(self, x):
        # x: T x C
        C = self.C
        out = []
        for t in range(len(x)):
            mean = sum(x[t]) / C
            var = sum((xi - mean) ** 2 for xi in x[t]) / C
            denom = math.sqrt(var + self.ln_eps)
            out.append([(xi - mean) / denom * self.g[i] + self.b[i] for i, xi in enumerate(x[t])])
        return out

    def matmul_vec(self, W, v):
        # W: out x in, v: in -> out
        return [sum(W[o][i] * v[i] for i in range(len(v))) for o in range(len(W))]

    def matmul_vec_batch(self, W, X):
        # X: T x C, W: out x in -> T x out
        return [self.matmul_vec(W, x_t) for x_t in X]

    def relu(self, z):
        return z if z > 0 else 0.0

    def forward(self, idx):
        """idx: list[int] length T"""
        T = len(idx)
        # token+pos embeddings
        x = []
        for t in range(T):
            tok = self.E[idx[t]]
            pos = self.P[t]
            x.append([tok[i] + pos[i] for i in range(self.C)])

        # self-attention (single head group)
        x_norm = self.layer_norm(x)

        # project to q,k,v: T x C
        Q = self.matmul_vec_batch(self.Wq, x_norm)
        K = self.matmul_vec_batch(self.Wk, x_norm)
        V = self.matmul_vec_batch(self.Wv, x_norm)

        # reshape into heads: T x H x D
        def split_heads(X):
            return [[X[t][h * self.D + d] for d in range(self.D)] for t in range(T) for h in range(self.H)]

        # simpler: compute attention per head without explicit reshape arrays
        # We'll compute logits for each time step per head.
        # output per position: C vector
        out = [[0.0 for _ in range(self.C)] for _ in range(T)]

        for h in range(self.H):
            # Qh,Kh,Vh: T x D
            Qh = [[Q[t][h * self.D + d] for d in range(self.D)] for t in range(T)]
            Kh = [[K[t][h * self.D + d] for d in range(self.D)] for t in range(T)]
            Vh = [[V[t][h * self.D + d] for d in range(self.D)] for t in range(T)]

            # causal attention: each position t attends to <=t
            for t in range(T):
                # scores for s in [0..t]
                scores = [sum(Qh[t][d] * Kh[s][d] for d in range(self.D)) * self.scale for s in range(t + 1)]
                # softmax
                m = max(scores)
                exps = [math.exp(si - m) for si in scores]
                denom = sum(exps) + 1e-12
                att = [e / denom for e in exps]

                # weighted sum of values
                for d in range(self.D):
                    val = sum(att[s] * Vh[s][d] for s in range(t + 1))
                    out[t][h * self.D + d] = val

        # output projection
        out2 = self.matmul_vec_batch(self.Wo, out)  # T x C

        # residual
        x = [[x[t][i] + out2[t][i] for i in range(self.C)] for t in range(T)]

        # feed-forward
        x_norm2 = self.layer_norm(x)
        hidden = self.matmul_vec_batch(self.W1, x_norm2)
        hidden = [[self.relu(hidden[t][i] + self.b1[i]) for i in range(len(hidden[t]))] for t in range(T)]
        ff = self.matmul_vec_batch(self.W2, hidden)
        ff = [[ff[t][i] + self.b2[i] for i in range(self.C)] for t in range(T)]

        x = [[x[t][i] + ff[t][i] for i in range(self.C)] for t in range(T)]

        # logits: project to vocab using token embedding as tied weights
        # logits[t][v] = dot(x[t], E[v])
        logits = []
        for t in range(T):
            row = []
            for v in range(self.V):
                row.append(sum(x[t][i] * self.E[v][i] for i in range(self.C)))
            logits.append(row)

        return logits

    def generate(self, tokenizer: ToyTokenizer, prompt: str, max_new_tokens: int = 80):
        idx = tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            idx_crop = idx[-self.cfg.block_size :]
            logits = self.forward(idx_crop)
            # last position logits
            last = logits[-1]
            # sample
            m = max(last)
            exps = [math.exp(li - m) for li in last]
            denom = sum(exps) + 1e-12
            probs = [e / denom for e in exps]
            r = random.random()
            c = 0.0
            next_id = 0
            for i, p in enumerate(probs):
                c += p
                if r <= c:
                    next_id = i
                    break
            idx.append(next_id)
        return tokenizer.decode(idx)


def cross_entropy_loss(logits, targets):
    # logits: T x V, targets: T ids
    loss = 0.0
    T = len(targets)
    V = len(logits[0])
    for t in range(T):
        row = logits[t]
        # log softmax
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        denom = sum(exps) + 1e-12
        log_probs = [x - m - math.log(denom) for x in row]
        loss -= log_probs[targets[t]]
    return loss / max(1, T)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    tok = ToyTokenizer(cfg.vocab)

    text = (
        "ai is interesting. ai is useful. "
        "machine learning trains models. "
        "neural networks learn patterns. "
        "transformers use attention. "
    ).lower()
    data = tok.encode(text)

    model = ToyTinyLLM(cfg)

    # Training note:
    # We are not implementing full backprop/gradient descent (too heavy in pure lists).
    # Instead we demonstrate the full pipeline and run a few forward passes.
    # This still creates useful LLM-project code for understanding.

    print("[demo] starting forward-pass loss checks...")
    for step in range(cfg.steps):
        x_ids, y_ids = split_batch(data, cfg.batch_size, cfg.block_size)
        # average loss across batch
        total = 0.0
        for b in range(cfg.batch_size):
            logits = model.forward(x_ids[b])
            total += cross_entropy_loss(logits, y_ids[b])
        avg = total / cfg.batch_size
        if step % 50 == 0:
            print(f"step={step} loss={avg:.4f}")

    print("\n[demo] sample generation (untrained / forward-check only):")
    print(model.generate(tok, prompt="ai", max_new_tokens=120))


if __name__ == "__main__":
    main()
