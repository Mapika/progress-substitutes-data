"""Dyck-4: well-formed nested bracket sequences, match-distance labels, held-out evaluation.

Tokens: id i < 4 opens a bracket of type i, id 4 + i closes it, 8 is the separator.

Sampler, per document: the length L is uniform over the even numbers from 2 to 256; the maximum nesting depth D
is uniform over 1..min(32, L / 2); the bracket skeleton is uniform over all balanced paths of length L whose
maximum depth is exactly D (each step goes up or down with probability proportional to its number of valid
completions, so nothing is rejected); every opening bracket draws its type uniformly.

Label classes (class of the prediction made at a token, i.e. of the target that follows it): 0 when the target
is not a closing bracket; for a closing bracket, by its match distance, the number of tokens back to the bracket
it closes,
    1  match distance 1
    2  match distance 2-7
    3  match distance 8-31
    4  match distance 32 or more
plus 4 (classes 5-8) when the nesting depth before the close is 16 or more.

Document i of a split is a function of (seed, split, i) only. The corpus of the paper:

    python -m thinning.domains.dyck build --out data/dyck --n-train 40000000 --n-eval 50000 --seed 0
"""

import argparse

import numpy as np
import torch
from numba import njit, prange

from thinning.data import read_rows, row_positions
from thinning.domains.othello import doc_seed, log_odds, next_u64, randbelow, write_splits

K, MAX_DEPTH, MIN_LEN, MAX_LEN = 4, 32, 2, 256
SEP_ID, VOCAB_SIZE = 2 * K, 2 * K + 1
SKILLS = {"closing_2_7": (2, 6), "closing_deep": (5, 6, 7, 8)}


@njit(cache=True)
def path_counts():
    """N[D, n, h]: number of paths of n steps from height h to height 0 that stay within [0, D];
    H[D, n, h]: number of those that touch height D. Float64 counts, built by additions only."""
    N = np.zeros((MAX_DEPTH + 1, MAX_LEN + 1, MAX_DEPTH + 2))
    H = np.zeros((MAX_DEPTH + 1, MAX_LEN + 1, MAX_DEPTH + 2))
    for D in range(MAX_DEPTH + 1):
        N[D, 0, 0] = 1.0
        if D == 0:
            H[D, 0, 0] = 1.0
        for n in range(1, MAX_LEN + 1):
            for h in range(D + 1):
                v = 0.0
                if h < D:
                    v += N[D, n - 1, h + 1]
                if h > 0:
                    v += N[D, n - 1, h - 1]
                N[D, n, h] = v
        for n in range(MAX_LEN + 1):
            H[D, n, D] = N[D, n, D]
        for n in range(1, MAX_LEN + 1):
            for h in range(D):
                v = H[D, n - 1, h + 1]
                if h > 0:
                    v += H[D, n - 1, h - 1]
                H[D, n, h] = v
    return N, H


@njit(cache=True)
def _sample_document(state, out, N, H):
    """Write one document and its separator into ``out``; return its length with the separator."""
    state, r = randbelow(state, (MAX_LEN - MIN_LEN) // 2 + 1)
    L = MIN_LEN + 2 * r
    state, r = randbelow(state, min(MAX_DEPTH, L // 2))
    D = 1 + r
    types = np.empty(MAX_DEPTH + 1, dtype=np.int64)
    h = 0
    touched = False
    for step in range(L):
        rem = L - step - 1
        up = 0.0
        down = 0.0
        if h < D:
            up = N[D, rem, h + 1] if (touched or h + 1 == D) else H[D, rem, h + 1]
        if h > 0:
            down = N[D, rem, h - 1] if touched else H[D, rem, h - 1]
        state, r = next_u64(state)
        if np.float64(r >> np.uint64(11)) / 2.0**53 * (up + down) < up:  # a uniform number in [0, 1) decides
            state, types[h] = randbelow(state, K)
            out[step] = types[h]
            h += 1
            touched = touched or h == D
        else:
            h -= 1
            out[step] = K + types[h]
    out[L] = SEP_ID
    return L + 1


@njit(cache=True, parallel=True)
def generate_documents(seed, split, indices, N, H):
    """The documents with the given indices of a split, as zero-padded rows, and their lengths with the
    separator."""
    buf = np.zeros((indices.shape[0], MAX_LEN + 1), dtype=np.uint8)
    lengths = np.empty(indices.shape[0], dtype=np.int64)
    for i in prange(indices.shape[0]):
        lengths[i] = _sample_document(doc_seed(seed, split, indices[i]), buf[i], N, H)
    return buf, lengths


@njit(cache=True)
def _label(tok):
    out = np.zeros(tok.shape[0], dtype=np.int8)
    opened = np.empty(tok.shape[0], dtype=np.int64)  # positions of the brackets that are open
    depth = 0
    for t in range(tok.shape[0] - 1):
        k = tok[t]
        if k == SEP_ID:
            depth = 0
        elif k < K:
            opened[depth] = t
            depth += 1
        else:
            depth -= 1
        if K <= tok[t + 1] < SEP_ID and depth > 0:
            distance = t + 1 - opened[depth - 1]
            out[t] = (1 if distance == 1 else 2 if distance < 8 else 3 if distance < 32 else 4) + 4 * (depth >= 16)
    return out


def label_stream(tokens: np.ndarray) -> np.ndarray:
    """Label class of the prediction made at every token of a chunk of well-formed documents that starts at a
    document start. The last token has no target in the chunk and gets class 0."""
    return _label(np.ascontiguousarray(tokens, dtype=np.int64))


class EvalSet:
    """The first ``n_rows`` rows of a held-out stream, cut as training rows are. The paper reads competence on
    2,000 rows of 2,048 tokens and the per-position odds on the first 500. A position is scored when its
    document starts inside its row. ``x`` holds the rows and ``scored`` (n_rows, seq_len) marks the scored
    positions; ``y`` (targets) and ``cls`` (label classes) run over the scored positions in row order."""

    def __init__(self, stream, n_rows: int, seq_len: int = 2048):
        self.x, y = read_rows(stream, 0, n_rows, seq_len)
        self.scored = y != -100
        self.y = y[self.scored]
        labels = label_stream(stream.window(0, n_rows * (seq_len + 1)))
        self.cls = labels[row_positions(0, n_rows, seq_len)][self.scored.reshape(-1).numpy()]


@torch.no_grad()
def measure(model, evalset: EvalSet, skill: str = "closing_2_7", device: str = "cpu", per_position=False) -> dict:
    """Competence and losses on the held-out rows.

    competence         mean over the skill's positions of p(true closing bracket) / p(any closing bracket):
                       the choice of the bracket type given that a bracket closes
    background_loss    cross-entropy (nats) at all scored positions outside the skill
    loss, skill_loss   cross-entropy at all scored positions and at the skill's positions
    per_position adds  skill_mass and other_mass: at each of the skill's positions, the shares of the
                       closing-bracket probability on the true bracket and on the three others"""
    lp = []
    for r0 in range(0, len(evalset.x), 64):
        x = evalset.x[r0:r0 + 64].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x)
        lp.append(torch.log_softmax(logits.float(), -1)[evalset.scored[r0:r0 + 64].to(device)].cpu())
    lp = torch.cat(lp)
    on = torch.from_numpy(np.isin(evalset.cls, SKILLS[skill]))
    log_p_true = lp.gather(1, evalset.y[:, None])[:, 0]
    nll = -log_p_true.double()
    p_close = lp[on][:, K:2 * K].exp().sum(1).double()
    out = {"competence": float((log_p_true[on].exp().double() / p_close.clamp(min=1e-30)).mean()),
           "background_loss": float(nll[~on].mean()),
           "loss": float(nll.mean()), "skill_loss": float(nll[on].mean())}
    if per_position:
        z = log_odds(lp[on][:, K:2 * K], evalset.y[on] - K)
        out.update(skill_mass=1 / (1 + np.exp(-z)), other_mass=1 / (1 + np.exp(z)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Dyck-4 corpus")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--out", required=True, help="output directory; the streams train/ and eval/ are written in it")
    ap.add_argument("--n-train", type=int, required=True, help="number of training documents")
    ap.add_argument("--n-eval", type=int, required=True, help="number of eval documents before duplicates are dropped")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    N, H = path_counts()
    write_splits(a.out, lambda split, indices: generate_documents(a.seed, split, indices, N, H), a.n_train, a.n_eval,
                 VOCAB_SIZE, SEP_ID, domain="dyck", seed=a.seed, k=K, max_depth=MAX_DEPTH, max_len=MAX_LEN)


if __name__ == "__main__":
    main()
