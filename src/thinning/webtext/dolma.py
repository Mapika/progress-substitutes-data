"""Dolma 1.7 as tokenized for DataDecide: staging, the instance pool, the data order of a seed, the repetition filter.

Source: the dataset ``allenai/DataDecide-data-recipes`` at a pinned revision. Its files are headerless uint16 token
streams (GPT-NeoX tokenizer of OLMo, vocabulary 50,304, documents separated by token 50279), and a training instance
is one of the consecutive 2,048-token blocks of a file. The files of each of the 15 sources are listed in
``olmo/data/named_data_mixes.py`` on the ``DataDecide`` branch of github.com/allenai/OLMo (``--mixes``).

    python -m thinning.webtext.dolma stage --mixes named_data_mixes.py --out data/dolma

reads runs of instances with HTTP range requests and writes rows of 2,048 uint16 tokens to ``<out>/<pool>/<source>
.bin``, with every staged chunk [file, first instance, instances] and the source weights in ``<out>/plan.json``:
  heldout  33.5M tokens that no training pool contains; sources with at least 20 files take them from files (5%)
           that the training pools leave out
  train    6.6B tokens
  extra    18B tokens, disjoint from the other two
A pool holds every source in proportion to its share of the instances of the recipe, and a training pool spreads a
source over its files in proportion to their sizes. Chunk offsets are random; ``make_plan`` replays the draws that
gave the pools of the paper, in order, with two sets of regions that no pool uses (a size-weighted draw of
training chunks and 600,000 further StarCoder instances). A rerun continues after the last whole chunk of a file.

A run reads the instances of ``Pool.order``: every source contributes a seeded sample without replacement of its
rows (train, then extra) in proportion to its weight, and the union is shuffled once.
"""

import argparse
import ast
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = "allenai/DataDecide-data-recipes"
REVISION = "3baf34baf5b636f0943401b5c6a2ccb7e5cf3bb9"
MIXES_SHA256 = "c1da2be413b7f70c9bd2728d0256a8231a33462a17e63ec5f81215e52c4d4234"
SOURCES = ["gutenberg_books", "pes20_stem_papers", "wikipedia_wikibooks", "megawika", "stackexchange", "arxiv",
           "algebraic_stack", "openwebmath", "tulu", "cc_news", "starcoder", "c4", "reddit", "falcon", "web_rest"]
ROW, ITEM = 2048, 2
HELDOUT_TOKENS, TRAIN_TOKENS, EXTRA_TOKENS = 33.5e6, 6.6e9, 18e9
MAX_CHUNK = 8192                                  # instances per range request (32 MiB)


def load_sources(mixes_path) -> dict:
    """{source: token files} from the DATA_SOURCES literal of the mixes file; a file listed twice counts twice."""
    text = Path(mixes_path).read_bytes()
    if hashlib.sha256(text).hexdigest() != MIXES_SHA256:
        print("warning: the mixes file differs from the one the pools of the paper were planned with", flush=True)
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "DATA_SOURCES")
    lists = {ast.literal_eval(k): v for k, v in zip(node.value.keys, node.value.values)}
    return {s: list(ast.literal_eval(lists[s])) for s in SOURCES}


def hub_sizes() -> dict:
    """{file: bytes} of the pinned revision."""
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        return {f["rfilename"]: f["size"] for f in json.load(r)["siblings"]}


def make_plan(src: dict, sizes: dict) -> dict:
    """{"weights": {source: share}, "plan": {pool: {source: [[file, first instance, instances], ...]}}}."""
    ninst = {p: sizes[p] // (ROW * ITEM) for s in SOURCES for p in src[s]}
    total = {s: sum(ninst[p] for p in src[s]) for s in SOURCES}
    weights = {s: total[s] / sum(total.values()) for s in SOURCES}
    files = {s: sorted(set(src[s])) for s in SOURCES}
    rows = lambda s, tokens: int(round(weights[s] * tokens / ROW))  # noqa: E731

    used = {}                                     # file -> [first, end) instance intervals that are taken
    rng = np.random.default_rng(0)
    for s in SOURCES:                             # regions that no pool uses: 4 of 4 MiB in each of 8 files
        for p in [files[s][j] for j in rng.choice(len(files[s]), min(8, len(files[s])), replace=False)]:
            for _ in range(4):
                off = int(rng.integers(0, max(1, (sizes[p] - 2**22) // 2))) * 2
                used.setdefault(p, []).append((off // (ROW * ITEM), (off + 2**22) // (ROW * ITEM) + 1))

    def place(rng, used, pick, k):
        """A free run of k instances at a random offset of the file that ``pick()`` names."""
        for _ in range(4000):
            p = pick()
            if ninst[p] < k:
                continue
            lo = int(rng.integers(0, ninst[p] - k + 1))
            if not any(lo < b and a < lo + k for a, b in used.get(p, ())):
                used.setdefault(p, []).append((lo, lo + k))
                return [p, lo, k]
        raise RuntimeError(f"no free run of {k} instances")

    def weighted(rng, used, fs, n, c):
        """n instances in chunks of at most c, each from a file drawn in proportion to its size."""
        fs, w = sorted(fs), np.array([ninst[p] for p in sorted(fs)], float)
        pick = lambda: fs[rng.choice(len(fs), p=w / w.sum())]  # noqa: E731
        return [place(rng, used, pick, k) for k in [c] * (n // c) + [n % c] * (n % c > 0)]

    def stratified(rng, fs, n):
        """n instances with every file contributing in proportion to its size (largest remainder)."""
        raw = n * np.array([ninst[p] for p in fs], float) / sum(ninst[p] for p in fs)
        share = np.floor(raw).astype(int)
        share[np.argsort(-(raw - share), kind="stable")[:n - int(share.sum())]] += 1
        c = max(1, min(MAX_CHUNK, int(np.ceil(n / len(fs) / 2)), max(1, n // 32)))
        return [place(rng, used, lambda: p, k)
                for p, m in zip(fs, share.tolist()) for k in [c] * (m // c) + [m % c] * (m % c > 0)]

    rng, scratch, heldout = np.random.default_rng(20261016), {p: list(v) for p, v in used.items()}, {}
    for s in SOURCES:                             # held-out chunks, then a size-weighted training draw (unused)
        ev = tr = files[s]
        if len(ev) >= 20:
            perm = rng.permutation(len(files[s]))
            k_ev = max(1, int(round(0.05 * len(perm))))
            ev, tr = [files[s][i] for i in perm[:k_ev]], [files[s][i] for i in perm[k_ev:]]
        n_ev = max(16, rows(s, HELDOUT_TOKENS))
        heldout[s] = weighted(rng, scratch, ev, n_ev, max(1, min(MAX_CHUNK, n_ev // 8)))
        weighted(rng, scratch, tr, rows(s, TRAIN_TOKENS), max(1, min(MAX_CHUNK, rows(s, TRAIN_TOKENS) // 32)))
    train_files = {}
    for s in SOURCES:
        ev = {p for p, _, _ in heldout[s]}
        train_files[s] = [p for p in files[s] if p not in ev] if len(files[s]) >= 20 else files[s]
        for p, lo, k in heldout[s]:
            used.setdefault(p, []).append((lo, lo + k))
    rng = np.random.default_rng(20261017)
    train = {s: stratified(rng, train_files[s], rows(s, TRAIN_TOKENS)) for s in SOURCES}
    stratified(np.random.default_rng(20261018), train_files["starcoder"], 600_000)
    rng = np.random.default_rng(20261018)
    extra = {s: stratified(rng, train_files[s], rows(s, EXTRA_TOKENS)) for s in SOURCES}
    return dict(weights=weights, plan=dict(heldout=heldout, train=train, extra=extra))


def fetch(chunk) -> bytes:
    """The bytes of one planned chunk, read with a range request."""
    path, lo, k = chunk
    start, nbytes = lo * ROW * ITEM, k * ROW * ITEM
    req = urllib.request.Request(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}",
                                 headers={"Range": f"bytes={start}-{start + nbytes - 1}"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                buf = r.read()
            if len(buf) == nbytes:
                return buf
        except OSError:
            pass
        time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"range read failed: {path} at {start}, {nbytes} bytes")


def stage(out, mixes, pools=("heldout", "train", "extra"), workers=4):
    """Write the plan and the pool files. A file that is partly written continues after its last whole chunk."""
    plan_path = Path(out) / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        plan_path.write_text(json.dumps(dict(repo=REPO, revision=REVISION, **make_plan(load_sources(mixes),
                                                                                       hub_sizes()))))
    plan = json.loads(plan_path.read_text())["plan"]
    with ThreadPoolExecutor(workers) as ex:
        for pool in pools:
            for s, chunks in plan[pool].items():
                f = Path(out) / pool / f"{s}.bin"
                f.parent.mkdir(exist_ok=True)
                ends = np.cumsum([k for _, _, k in chunks]) * ROW * ITEM
                done = int(np.searchsorted(ends, f.stat().st_size if f.exists() else 0, side="right"))
                with open(f, "ab") as fh:
                    fh.truncate(int(ends[done - 1]) if done else 0)
                    for buf in ex.map(fetch, chunks[done:]):
                        fh.write(buf)
                print(f"{pool}/{s}: {int(ends[-1]) // (ROW * ITEM)} rows", flush=True)


class Pool:
    """Staged training rows per source: the rows of ``train`` followed by those of ``extra``."""

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.weights = json.loads((data_dir / "plan.json").read_text())["weights"]
        self.keys = sorted(self.weights)
        self.parts = {}
        for k in self.keys:
            paths = [p for p in (data_dir / "train" / f"{k}.bin", data_dir / "extra" / f"{k}.bin") if p.exists()]
            self.parts[k] = [np.memmap(p, np.uint16, "r").reshape(-1, ROW) for p in paths]
        self.n_rows = {k: sum(len(p) for p in self.parts[k]) for k in self.keys}

    def row(self, source: int, r: int) -> np.ndarray:
        for part in self.parts[self.keys[source]]:
            if r < len(part):
                return part[r]
            r -= len(part)
        raise IndexError(r)

    def counts(self, n_inst: int) -> dict:
        """Instances per source for a run of n_inst instances (largest remainder)."""
        raw = np.array([self.weights[k] for k in self.keys]) / sum(self.weights.values()) * n_inst
        base = np.floor(raw).astype(np.int64)
        base[np.argsort(-(raw - base), kind="stable")[:n_inst - int(base.sum())]] += 1
        if any(c > self.n_rows[k] for k, c in zip(self.keys, base)):
            raise ValueError(f"the pools have too few rows for a run of {n_inst} instances")
        return dict(zip(self.keys, map(int, base)))

    def order(self, n_inst: int, seed: int) -> np.ndarray:
        """(n_inst, 2) int64 [source index, row]: the instances of a run in the order they are read."""
        rng = np.random.Generator(np.random.PCG64(seed))
        counts = self.counts(n_inst)
        parts = []
        for i, k in enumerate(self.keys):
            r = rng.permutation(self.n_rows[k])[:counts[k]].astype(np.int64)
            parts.append(np.stack([np.full(counts[k], i, np.int64), r], 1))
        o = np.concatenate(parts)
        return o[rng.permutation(len(o))]


def instance_ok(x: np.ndarray, max_period: int = 13, max_count: int = 32) -> bool:
    """The repetition filter of OLMo (olmo/data/util.py, Apache-2.0): False for an instance in which a sequence
    of period 1-13 repeats at least 32 times; all targets of such an instance are removed from the loss."""
    lead = lambda a: len(a) if a.all() else int(np.argmin(a))  # noqa: E731  length of the leading run of True
    for period in range(1, min(max_period, len(x) // 3) + 1):
        shaped = np.pad(x, (0, period - len(x) % period), constant_values=-1).reshape(-1, period)
        eq = shaped == np.roll(shaped, shift=1, axis=0)       # a block of the period equals the block before it
        hits = np.flatnonzero(eq.all(axis=1))
        for run in np.split(hits, np.flatnonzero(np.diff(hits) != 1) + 1) if len(hits) else ():
            before = lead(eq[run[0] - 1][::-1])
            start = (run[0] - 1) * period - (before if before < period else 0)
            end = (run[-1] + 1) * period + lead(eq[run[-1] + 1])
            if (end - start) // period >= max_count:
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage the Dolma 1.7 pools")
    ap.add_argument("command", choices=["stage"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixes", required=True, help="olmo/data/named_data_mixes.py of the OLMo DataDecide branch")
    ap.add_argument("--pools", default="heldout,train,extra")
    a = ap.parse_args(argv)
    stage(a.out, a.mixes, a.pools.split(","))


if __name__ == "__main__":
    main()
