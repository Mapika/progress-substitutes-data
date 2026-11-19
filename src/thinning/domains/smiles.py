"""SMILES: character-level molecular strings, ring-closure and branch-closing labels, held-out evaluation.

Tokens: one character per token, with the ids of ``STOI`` in a vocabulary of 79 ids (the other ids are unused);
1 is the separator. Strings are used as given.

Label classes (class of the prediction made at a token, i.e. of the target that follows it), read from the
string alone:
    0  any other target
    1  ring-opening digit: a ring-bond digit whose ring number is not open
    2  ring-closing digit: a ring-bond digit whose ring number is open (ring closure)
    3  ")" (branch closing)
    4  the separator (end of the molecule)
    5  "%" or the first digit of a two-digit ring label %nn, whose ring event is on its second digit
    6  "(" (branch opening)
Digits inside [...] are isotopes, hydrogen counts or charges, not ring bonds.

The corpus of the paper is read from the Hugging Face dataset chandar-lab/ZINC_22, revision
f13c8b4132037e38102499e07741dc9a71431ea5, split train (1,212 parquet files), column SMILES, in file order and
then row order. The training stream is the first 300,000,000 molecules and the held-out stream the last 3,000,000
of the split, without the strings that occur in the training stream:

    python -m thinning.domains.smiles download --out raw/zinc22
    python -m thinning.domains.smiles build --parquet raw/zinc22 --out data/smiles

The stream of the paper lacks 830 of the first 300,000,000 molecules, which a check that is not applied here
excluded. The first of them is row 419,362, so later stream positions are shifted relative to the paper's.

Any other corpus is built from text files with one string per line; the held-out molecules then follow the
training molecules:

    python -m thinning.domains.smiles build --smiles a.smi b.smi --out data/smiles --n-train 1000000 --n-eval 10000
"""

import argparse
import urllib.request
from collections import deque
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from numba import njit

from thinning.data import write_stream
from thinning.domains.othello import first_documents, log_odds, log_probs

VOCAB_SIZE, SEP_ID = 79, 1
STOI = {"#": 2, "%": 3, "(": 4, ")": 5, "+": 7, "-": 8, ".": 9, "/": 10, "0": 11, "1": 12, "2": 13, "3": 14,
        "4": 15, "5": 16, "6": 17, "7": 18, "8": 19, "9": 20, "=": 22, "@": 23, "B": 25, "C": 26, "F": 29, "H": 31,
        "I": 32, "N": 37, "O": 38, "P": 39, "S": 42, "[": 50, "\\": 51, "]": 52, "c": 55, "e": 57, "i": 61,
        "l": 64, "n": 66, "o": 67, "r": 70, "s": 71}
SKILLS = {"ring_closure": (2,), "branch_closing": (3,)}
D0, PERCENT, LEFT_PAREN, RIGHT_PAREN = STOI["0"], STOI["%"], STOI["("], STOI[")"]
LEFT_BRACKET, RIGHT_BRACKET = STOI["["], STOI["]"]
ATOM_IDS = [STOI[c] for c in "BCFINOPScnos["]  # first character of an atom; "[" opens a bracket atom
REVISION = "f13c8b4132037e38102499e07741dc9a71431ea5"
PARQUET_URL = f"https://huggingface.co/datasets/chandar-lab/ZINC_22/resolve/{REVISION}/data/"
PARQUET_NAMES = [f"train-{i:05d}-of-01212.parquet" for i in range(1212)]

_ENCODE = bytes({**STOI, "\n": SEP_ID}.get(chr(b), 255) for b in range(256))  # character code -> id, 255 if none
_DECODE = bytes(max(_ENCODE.find(i), 0) for i in range(256))  # id -> character code
_DELETE_VOCABULARY = str.maketrans("", "", "".join(STOI))


def encode(text: str) -> np.ndarray:
    """Token ids of a SMILES string. A line break becomes the separator, so "CCO\\nC1CC1\\n" is two documents."""
    ids = np.frombuffer(text.encode("ascii").translate(_ENCODE), dtype=np.uint8)
    if (ids == 255).any():
        raise ValueError("character outside the vocabulary")
    return ids


def decode(ids) -> str:
    """The string of a token array; separators become line breaks."""
    return np.asarray(ids, dtype=np.uint8).tobytes().translate(_DECODE).decode("ascii")


@njit(cache=True)
def _scan(tok):
    """For every token: its class as a target, and the bit mask of the ring numbers 0-9 that are open after it.
    The stream starts at a molecule start."""
    cls = np.zeros(tok.shape[0], dtype=np.int8)
    open_after = np.zeros(tok.shape[0], dtype=np.uint16)
    is_open = np.zeros(100, dtype=np.bool_)
    mask = 0
    in_bracket = False
    percent = 0  # 1 after "%", 2 after the first digit of %nn
    tens = 0
    for t in range(tok.shape[0]):
        k = tok[t]
        ring = -1
        if k == SEP_ID:
            cls[t] = 4
            is_open[:] = False
            mask = 0
            in_bracket = False
            percent = 0
        elif in_bracket:
            in_bracket = k != RIGHT_BRACKET
        elif k == LEFT_BRACKET:
            in_bracket = True
        elif percent == 1:
            cls[t] = 5
            tens = (k - D0) * 10
            percent = 2
        elif percent == 2:
            ring = tens + (k - D0)
            percent = 0
        elif k == PERCENT:
            cls[t] = 5
            percent = 1
        elif D0 <= k <= D0 + 9:
            ring = k - D0
        elif k == RIGHT_PAREN:
            cls[t] = 3
        elif k == LEFT_PAREN:
            cls[t] = 6
        if ring >= 0:
            cls[t] = 2 if is_open[ring] else 1
            is_open[ring] = not is_open[ring]
            if ring < 10:
                mask ^= 1 << ring
        open_after[t] = mask
    return cls, open_after


def label_stream(tokens: np.ndarray) -> np.ndarray:
    """Label class of the prediction made at every token of a chunk that starts at a molecule start. The last
    token has no target in the chunk and gets class 0."""
    target_class, _ = _scan(np.ascontiguousarray(tokens, dtype=np.int64))
    return np.append(target_class[1:], np.int8(0))


class EvalSet:
    """The first ``n_docs`` molecules of a held-out stream (8,000 in the paper). Arrays over all predictions, in
    stream order: ``cls`` label class, ``y`` target token, ``open_digits`` (N, 10) the ring numbers that are
    open when the prediction is made."""

    def __init__(self, stream, n_docs: int):
        self.docs = first_documents(stream, n_docs)
        self.y = np.concatenate([np.concatenate((m, [SEP_ID])) for m in self.docs])
        target_class, open_after = _scan(np.concatenate(([SEP_ID], self.y)))
        self.cls = target_class[1:]
        self.open_digits = (open_after[:-1, None] >> np.arange(10)) & 1 > 0


def measure(model, evalset: EvalSet, skill: str = "ring_closure", device: str = "cpu", per_position=False) -> dict:
    """Competence and losses on the held-out molecules.

    competence            mean probability of the target at the skill's positions (the true closing digit at
                          ring closures, ")" at branch closings)
    background_loss       cross-entropy (nats) at all positions outside the skill
    loss, skill_loss      cross-entropy at all positions and at the skill's positions
    ring_opening_choice   at ring-opening digits, p(true digit) / p(any digit)
    per_position adds     skill_mass and other_mass: the probability of the target and of all other tokens at
                          each of the skill's positions"""
    lp = log_probs(model, evalset.docs, SEP_ID, device, batch_docs=1000)
    y = torch.from_numpy(evalset.y)
    nll = -lp.gather(1, y[:, None])[:, 0].double().numpy()
    on = np.isin(evalset.cls, SKILLS[skill])
    opening = evalset.cls == 1
    p_digit = lp[torch.from_numpy(opening), D0:D0 + 10].exp().double().numpy().sum(1)
    out = {"competence": float(np.exp(-nll[on]).mean()),
           "background_loss": float(nll[~on].mean()),
           "loss": float(nll.mean()), "skill_loss": float(nll[on].mean()),
           "ring_opening_choice": float((np.exp(-nll[opening]) / np.maximum(p_digit, 1e-30)).mean())}
    if per_position:
        z = log_odds(lp[torch.from_numpy(on)], y[torch.from_numpy(on)])
        out.update(skill_mass=1 / (1 + np.exp(-z)), other_mass=1 / (1 + np.exp(z)))
    return out


def output_classes(model, evalset: EvalSet, device: str = "cpu") -> dict:
    """Mean probability of each class of next token at the ring-closure positions: the true closing digit, the
    digit of another open ring, a digit whose ring is not open, an atom, a branch opening, any other token."""
    closure = evalset.cls == 2
    p = log_probs(model, evalset.docs, SEP_ID, device, batch_docs=1000)[torch.from_numpy(closure)].exp()
    p = p.double().numpy()
    digits, is_open = p[:, D0:D0 + 10], evalset.open_digits[closure]
    true = digits[np.arange(len(p)), evalset.y[closure] - D0]
    out = {"true_digit": true.mean(),
           "other_open_digit": ((digits * is_open).sum(1) - true).mean(),
           "digit_not_open": (digits * ~is_open).sum(1).mean(),
           "atom": p[:, ATOM_IDS].sum(1).mean(),
           "branch_opening": p[:, LEFT_PAREN].mean()}
    other = np.setdiff1d(np.arange(VOCAB_SIZE), [*range(D0, D0 + 10), *ATOM_IDS, LEFT_PAREN])
    out["other"] = p[:, other].sum(1).mean()
    return {k: float(v) for k, v in out.items()}


def _rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def download(out, n_train: int, n_eval: int) -> None:
    """Fetch the leading parquet files that hold ``n_train`` molecules and the trailing files that hold
    ``n_eval``. A file that is already in ``out`` is not fetched again."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for names, needed in ((PARQUET_NAMES, n_train), (PARQUET_NAMES[::-1], n_eval)):
        rows = 0
        for name in names:
            if rows >= needed:
                break
            if not (out / name).exists():
                urllib.request.urlretrieve(PARQUET_URL + name, out / (name + ".part"))
                (out / (name + ".part")).replace(out / name)
            rows += _rows(out / name)


def _strings(path: Path):
    """The strings of one file, in order: the SMILES column of a parquet file or the lines of a text file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(columns=["SMILES"]):
            yield from batch.column(0).to_pylist()
    else:
        with open(path) as fh:
            yield from (line.strip() for line in fh)


def molecules(files):
    """The usable strings of the files, in order: not empty and written with vocabulary characters only."""
    for path in files:
        for s in _strings(Path(path)):
            if s and not s.translate(_DELETE_VOCABULARY):
                yield s


def _last_molecules(files, n: int) -> list[str]:
    """The last ``n`` molecules of parquet files, read from the trailing files that hold ``n`` rows."""
    first, rows = len(files), 0
    while first > 0 and rows < n:
        first -= 1
        rows += _rows(files[first])
    return list(deque(molecules(files[first:]), maxlen=n))


def build(files, out, n_train: int, n_eval: int, shard_molecules: int = 20_000_000) -> None:
    """Write ``out/train``, the first ``n_train`` molecules, and ``out/eval``: the last ``n_eval`` molecules of
    parquet files, or the ``n_eval`` molecules that follow the training molecules in text files, in both cases
    without the strings that occur in train."""
    out, files = Path(out), [Path(f) for f in files]
    if files[0].suffix == ".parquet":
        held_out = _last_molecules(files, n_eval)
    else:
        held_out = list(islice(molecules(files), n_train, n_train + n_eval))
    candidates, in_train = set(held_out), set()

    def train_shards():
        source = islice(molecules(files), n_train)
        while shard := list(islice(source, shard_molecules)):
            in_train.update(candidates.intersection(shard))
            yield encode("".join(s + "\n" for s in shard))

    write_stream(out / "train", train_shards(), VOCAB_SIZE, SEP_ID, split="train", domain="smiles")
    eval_tokens = encode("".join(s + "\n" for s in held_out if s not in in_train))
    write_stream(out / "eval", [eval_tokens], VOCAB_SIZE, SEP_ID, split="eval", domain="smiles")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download the ZINC22 files of the paper and build a SMILES corpus")
    ap.add_argument("command", choices=["download", "build"])
    ap.add_argument("--out", required=True, help="download: directory for the parquet files; build: output "
                    "directory, the streams train/ and eval/ are written in it")
    ap.add_argument("--parquet", help="build from the parquet files of this directory")
    ap.add_argument("--smiles", nargs="+", help="build from text files with one SMILES string per line")
    ap.add_argument("--n-train", type=int, default=300_000_000, help="number of training molecules")
    ap.add_argument("--n-eval", type=int, default=3_000_000, help="number of eval molecules before duplicates "
                    "are dropped")
    a = ap.parse_args(argv)
    if a.command == "download":
        download(a.out, a.n_train, a.n_eval)
    else:
        files = sorted(Path(a.parquet).glob("train-*.parquet")) if a.parquet else a.smiles
        build(files, a.out, a.n_train, a.n_eval)


if __name__ == "__main__":
    main()
