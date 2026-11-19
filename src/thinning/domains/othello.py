"""Othello: uniformly sampled legal games, forced-pass and forced-move labels, held-out evaluation.

Tokens: ids 0-59 are the 60 playable squares in increasing square index (square = 8 * (row - 1) + column, so
a1 = 0 and h8 = 63; the four centre squares are never played), 60 is the pass token and 61 the separator. Black
moves first. A side without a legal square emits the pass token when its opponent can move, and the game ends
when neither side can move. Boards are 64-bit masks, one bit per square.

Label classes (class of the prediction made at a token, i.e. of the position reached after it):
    0  the side to move has two or more legal squares
    1  exactly one legal square (forced move)
    2  no legal square while the opponent can move: the target is the pass token (forced pass)
    3  the game is over: the target is the separator

Game i of a split is a function of (seed, split, i) only. The corpus of the paper:

    python -m thinning.domains.othello build --out data/othello --n-train 700000000 --n-eval 1000000 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from numba import njit, prange

from thinning.data import write_stream

PASS_ID, SEP_ID, VOCAB_SIZE = 60, 61, 62
SKILLS = {"forced_pass": (2,), "forced_move": (1,)}
PLAYABLE = np.array([sq for sq in range(64) if sq not in (27, 28, 35, 36)], dtype=np.int64)
TOK2SQ = np.concatenate((PLAYABLE, [-1, -1]))
SQ2TOK = np.full(64, -1, dtype=np.int64)
SQ2TOK[PLAYABLE] = np.arange(60)
MAX_GAME_TOKENS = 128  # 60 moves, at most 60 passes and the separator

U0, U1 = np.uint64(0), np.uint64(1)
INIT_BLACK = np.uint64((1 << 28) | (1 << 35))  # e4, d5
INIT_WHITE = np.uint64((1 << 27) | (1 << 36))  # d4, e5
_NOT_A, _NOT_H = np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F)
_S1, _S7, _S8, _S9, _S27, _S30, _S31, _S32 = (np.uint64(s) for s in (1, 7, 8, 9, 27, 30, 31, 32))
_GOLDEN, _M1, _M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_FNV_OFFSET, _FNV_PRIME = np.uint64(0xCBF29CE484222325), np.uint64(0x100000001B3)


@njit(cache=True, inline="always")
def _mix64(z):
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


@njit(cache=True)
def doc_seed(base_seed, split, index):
    """State of the random stream of document ``index`` of a split (0 train, 1 eval)."""
    h = _mix64(np.uint64(base_seed) + _GOLDEN)
    h = _mix64(h ^ (np.uint64(split) * _M1 + _GOLDEN))
    return _mix64(h ^ (np.uint64(index) * _M2 + _GOLDEN))


@njit(cache=True, inline="always")
def next_u64(state):
    """One splitmix64 step: the new state and a random 64-bit integer."""
    state = state + _GOLDEN
    return state, _mix64(state)


@njit(cache=True, inline="always")
def randbelow(state, n):
    """The new state and a uniform integer in [0, n), for n < 2**32."""
    state, r = next_u64(state)
    return state, np.int64(((r >> _S32) * np.uint64(n)) >> _S32)


@njit(cache=True, inline="always")
def _shift(x, d):
    """Move every disc one square in direction ``d``; discs that leave the board are dropped."""
    if d < 2:
        return (x << _S1) & _NOT_A if d == 0 else (x >> _S1) & _NOT_H
    if d < 4:
        return x << _S8 if d == 2 else x >> _S8
    if d < 6:
        return (x << _S9) & _NOT_A if d == 4 else (x << _S7) & _NOT_H
    return (x >> _S7) & _NOT_A if d == 6 else (x >> _S9) & _NOT_H


@njit(cache=True)
def legal_moves(p, o):
    """Squares on which the side with discs ``p`` may move against discs ``o``: each flips at least one disc."""
    empty = ~(p | o)
    moves = U0
    for d in range(8):
        t = _shift(p, d) & o
        for _ in range(5):
            t |= _shift(t, d) & o
        moves |= _shift(t, d) & empty
    return moves


@njit(cache=True)
def flips(p, o, sq):
    """Discs of ``o`` bracketed, and so flipped, when the side with discs ``p`` moves on square ``sq``."""
    m = U1 << np.uint64(sq)
    if (p | o) & m:
        return U0
    out = U0
    for d in range(8):
        f = U0
        x = _shift(m, d)
        while x & o:
            f |= x
            x = _shift(x, d)
        if x & p:
            out |= f
    return out


@njit(cache=True, inline="always")
def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, inline="always")
def _kth_square(x, k):
    """Index of the k-th lowest set bit of ``x`` (k = 0 is the lowest)."""
    for _ in range(k):
        x &= x - U1
    return popcount((x & (~x + U1)) - U1)


@njit(cache=True)
def _play_random_game(state, out):
    """Write one game of uniformly sampled legal moves, with its separator, into ``out``; return its length."""
    p, o = INIT_BLACK, INIT_WHITE
    n = 0
    while True:
        moves = legal_moves(p, o)
        if moves == U0:
            if legal_moves(o, p) == U0:
                break
            out[n] = PASS_ID
        else:
            state, k = randbelow(state, popcount(moves))
            sq = _kth_square(moves, k)
            f = flips(p, o, sq)
            p |= f | (U1 << np.uint64(sq))
            o &= ~f
            out[n] = SQ2TOK[sq]
        n += 1
        p, o = o, p
    out[n] = SEP_ID
    return n + 1


@njit(cache=True, parallel=True)
def generate_games(seed, split, indices):
    """The games with the given indices of a split, as zero-padded rows, and their lengths with the separator."""
    buf = np.zeros((indices.shape[0], MAX_GAME_TOKENS), dtype=np.uint8)
    lengths = np.empty(indices.shape[0], dtype=np.int64)
    for i in prange(indices.shape[0]):
        lengths[i] = _play_random_game(doc_seed(seed, split, indices[i]), buf[i])
    return buf, lengths


@njit(cache=True, parallel=True)
def _doc_hashes(buf, lengths):
    """64-bit FNV-1a hash of every document ``buf[i, :lengths[i]]``."""
    out = np.empty(lengths.shape[0], dtype=np.uint64)
    for i in prange(lengths.shape[0]):
        h = _FNV_OFFSET
        for j in range(lengths[i]):
            h = (h ^ np.uint64(buf[i, j])) * _FNV_PRIME
        out[i] = h
    return out


def write_splits(out, generate, n_train, n_eval, vocab_size, sep_id, docs_per_shard=1 << 24, **provenance):
    """Write the streams ``out/train`` and ``out/eval`` of a generated corpus.

    ``generate(split, indices)`` returns the documents with the given indices of split 0 (train) or 1 (eval) as
    zero-padded rows, and their lengths with the separator. An eval document that equals a training document is
    left out: its hash is looked up among the training hashes and the training document is generated again for
    an exact comparison."""
    out = Path(out)
    hashes = []

    def flatten(buf, lengths):
        return buf[np.arange(buf.shape[1]) < lengths[:, None]]

    def train_shards():
        for a in range(0, n_train, docs_per_shard):
            buf, lengths = generate(0, np.arange(a, min(a + docs_per_shard, n_train)))
            hashes.append(_doc_hashes(buf, lengths))
            yield flatten(buf, lengths)

    def eval_shards():
        known, first = np.unique(np.concatenate(hashes), return_index=True)
        for a in range(0, n_eval, docs_per_shard):
            buf, lengths = generate(1, np.arange(a, min(a + docs_per_shard, n_eval)))
            h = _doc_hashes(buf, lengths)
            at = np.minimum(np.searchsorted(known, h), known.size - 1)
            hit = np.flatnonzero(known[at] == h)
            keep = np.ones(len(lengths), dtype=bool)
            if hit.size:
                train_buf, train_len = generate(0, first[at[hit]])
                for r, i in enumerate(hit):
                    keep[i] = not np.array_equal(train_buf[r, :train_len[r]], buf[i, :lengths[i]])
            yield flatten(buf[keep], lengths[keep])

    write_stream(out / "train", train_shards(), vocab_size, sep_id, split="train", **provenance)
    write_stream(out / "eval", eval_shards(), vocab_size, sep_id, split="eval", **provenance)


@njit(cache=True)
def _replay(tok):
    """Label class and legal squares of the side to move after every token. The stream starts at a game start."""
    cls = np.zeros(tok.shape[0], dtype=np.int8)
    legal = np.zeros(tok.shape[0], dtype=np.uint64)
    p, o = INIT_BLACK, INIT_WHITE
    for t in range(tok.shape[0]):
        k = tok[t]
        if k == SEP_ID:
            p, o = INIT_BLACK, INIT_WHITE
        elif k == PASS_ID:
            p, o = o, p
        else:
            sq = np.uint64(TOK2SQ[k])
            f = flips(p, o, sq)
            p |= f | (U1 << sq)
            o &= ~f
            p, o = o, p
        legal[t] = legal_moves(p, o)
        if legal[t] != U0:
            cls[t] = 1 if popcount(legal[t]) == 1 else 0
        elif legal_moves(o, p) != U0:
            cls[t] = 2
        else:
            cls[t] = 3
    return cls, legal


def label_stream(tokens: np.ndarray) -> np.ndarray:
    """Label class of the prediction made at every token of a chunk that starts at a game start."""
    return _replay(np.ascontiguousarray(tokens, dtype=np.int64))[0]


def first_documents(stream, n_docs: int) -> list[np.ndarray]:
    """The first ``n_docs`` documents of a stream, without separators, for documents of at most 80 tokens on average."""
    w = stream.window(0, min(stream.total_tokens, 80 * n_docs + 100_000)).astype(np.int64)
    cuts = np.flatnonzero(w == stream.sep_id)[:n_docs]
    return [w[a:b] for a, b in zip(np.concatenate(([0], cuts[:-1] + 1)), cuts)]


@torch.no_grad()
def log_probs(model, docs, sep_id: int, device, batch_docs: int) -> torch.Tensor:
    """Next-token log-probabilities (N, vocabulary) at every prediction of every document, in order. A document
    is scored as [sep] + document + [sep], so it contributes one row per token and one for its closing separator."""
    out = []
    for b0 in range(0, len(docs), batch_docs):
        batch = docs[b0:b0 + batch_docs]
        width = max(len(d) for d in batch) + 1
        x = torch.full((len(batch), width), sep_id, dtype=torch.long)
        scored = torch.zeros(len(batch), width, dtype=torch.bool)
        for r, d in enumerate(batch):
            x[r, 1:1 + len(d)] = torch.as_tensor(d)
            scored[r, :len(d) + 1] = True
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x)
        out.append(torch.log_softmax(logits.float()[scored.to(device)], -1).cpu())
    return torch.cat(out)


def log_odds(lp: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """ln p(target) - ln p(every other token) for rows of log-probabilities ``lp`` and targets ``y``."""
    rest = lp.masked_fill(torch.nn.functional.one_hot(y, lp.shape[1]).bool(), float("-inf"))
    return (lp.gather(1, y[:, None])[:, 0] - torch.logsumexp(rest, 1)).double().numpy()


class EvalSet:
    """The first ``n_docs`` games of a held-out stream; the paper scores snapshots on 8,000 games and returns on
    2,000. Arrays over all predictions, in stream order: ``cls`` label class, ``legal`` (N, 60) legal squares of
    the side to move, ``y`` target token."""

    def __init__(self, stream, n_docs: int):
        self.docs = first_documents(stream, n_docs)
        inputs = np.concatenate([np.concatenate(([SEP_ID], g)) for g in self.docs])
        self.y = np.concatenate([np.concatenate((g, [SEP_ID])) for g in self.docs])
        self.cls, legal = _replay(inputs)
        bits = np.unpackbits(legal.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
        self.legal = bits[:, PLAYABLE].astype(bool)


def measure(model, evalset: EvalSet, skill: str = "forced_pass", device: str = "cpu", per_position=False) -> dict:
    """Competence and losses on the held-out games.

    competence             mean probability of the target at the skill's positions (the pass token at forced
                           passes, the only legal square at forced moves)
    background_loss        cross-entropy (nats) at positions with two or more legal squares
    loss, skill_loss       cross-entropy at all positions and at the skill's positions
    forced_move_legality   at forced moves, the share of the probability on squares that is on the legal square
    legal_mass_two         probability on the legal squares at positions with exactly two
    per_position adds      skill_mass and other_mass (probability of the target and of all other tokens at each
                           of the skill's positions) and background_loss_positions (cross-entropy at all
                           positions outside the skill)"""
    lp = log_probs(model, evalset.docs, SEP_ID, device, batch_docs=500)
    y = torch.from_numpy(evalset.y)
    nll = -lp.gather(1, y[:, None])[:, 0].double().numpy()
    p = lp.exp().double().numpy()
    cls, legal = evalset.cls, evalset.legal
    on = np.isin(cls, SKILLS[skill])
    legal_mass = (p[:, :60] * legal).sum(1)
    out = {"competence": float(p[on, evalset.y[on]].mean()),
           "background_loss": float(nll[cls == 0].mean()),
           "loss": float(nll.mean()), "skill_loss": float(nll[on].mean()),
           "forced_move_legality": float((legal_mass / p[:, :60].sum(1))[cls == 1].mean()),
           "legal_mass_two": float(legal_mass[(cls == 0) & (legal.sum(1) == 2)].mean())}
    if per_position:
        z = log_odds(lp[torch.from_numpy(on)], y[torch.from_numpy(on)])
        out.update(skill_mass=1 / (1 + np.exp(-z)), other_mass=1 / (1 + np.exp(z)),
                   background_loss_positions=float(nll[~on].mean()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Othello corpus: uniformly sampled legal games")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--out", required=True, help="output directory; the streams train/ and eval/ are written in it")
    ap.add_argument("--n-train", type=int, required=True, help="number of training games")
    ap.add_argument("--n-eval", type=int, required=True, help="number of eval games before duplicates are dropped")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    write_splits(a.out, lambda split, indices: generate_games(a.seed, split, indices), a.n_train, a.n_eval,
                 VOCAB_SIZE, SEP_ID, domain="othello", seed=a.seed)


if __name__ == "__main__":
    main()
