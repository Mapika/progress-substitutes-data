"""Held-out evaluation of web-text models: the evaluation sets, competence and background loss.

    python -m thinning.webtext.evalsets build --data data/dolma --tokenizer tokenizer.json --out data/eval
    python -m thinning.webtext.evalsets score-capitals --snapshot run/step8192 --eval data/eval \
        --tokenizer tokenizer.json --out capitals_step8192.npz

``build`` (which needs spaCy with ``en_core_web_sm``) writes two files from the held-out pool. heldout.npz has 768
rows stratified over the sources with their structural classes, the auxiliaries among them whose parsed subject is
a preceding noun, proper noun or numeral, 32 rows of repeated random word sequences and 6 rows of counting
sequences (comma lists and numbered lines). closing.npz has 1,000 windows of 256 tokens from other held-out rows
that end at a closing bracket whose opener is 16-255 tokens back, each in three variants that differ only in the
opener token, one per bracket type, with the bare closing bracket of that type as the last token.

The prompts and answer groups of the two factual relations come from ``thinning.webtext.entities``: 816 capital
prompts over 102 groups and 864 city prompts over 54 groups.

Measures of ``Evaluator.measure``: ``loss`` is the cross-entropy, in nats, at every target of the held-out rows,
``background_loss_structural`` at the targets outside the four structural classes and ``background_loss`` at
those outside the structural and the factual classes. ``competence_agreement`` is p(true form) / [p(true form) +
p(other-number form)] at the auxiliaries, ``competence_repetition`` p(true token) in the second copy of a random
sequence, ``competence_counting`` p(v) / [p(v-1) + p(v) + p(v+1) + p(v+2)] at the numbers of a count and
``competence_closing`` the mass on the tokens that start with the true closing bracket over the mass on those that
start with any closing bracket (1/3 without reading the opener; ``competence_closing_nested`` over the windows
with another bracket pair inside). ``competence_capitals`` and ``competence_city_countries`` are the mass on the
true answer group over the mass on all groups at the last prompt token. ``score-capitals`` writes, for a snapshot,
the mass on the true group (``skill_mass``) and on the other 101 groups (``other_mass``) of every capital prompt,
and ``background_loss``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from thinning.webtext import labels as L
from thinning.webtext.dolma import ROW
from thinning.webtext.entities import build_prompts

KEYS = ("loss", "background_loss_structural", "background_loss", "competence_agreement", "competence_repetition",
        "competence_counting", "competence_closing", "competence_closing_nested")


def _subject(t):
    """Subject head of a finite verb in a spaCy parse; an auxiliary or a conjunct takes that of its head."""
    for _ in range(6):
        for c in t.children:
            if c.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                return c
        if t.dep_ not in ("aux", "auxpass", "conj") or t.head is t:
            return None
        t = t.head
    return None


def agreement_items(X, T) -> np.ndarray:
    """(row, column, token, other-number token) of the auxiliaries of rows X whose subject, in the spaCy parse of
    their document, is a preceding noun, proper noun or numeral."""
    import spacy
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
    nlp.max_length = 201_000
    items = []
    for r, x in enumerate(X):
        cut = [-1] + list(np.flatnonzero(x == L.EOS)) + [len(x)]
        for a, b in zip(cut[:-1], cut[1:]):
            seg = x[a + 1:b]
            strs = [T.strings[int(v)] for v in seg]
            text = "".join(strs)
            if len(seg) < 8 or not text.strip():
                continue
            end = np.cumsum([len(s) for s in strs])                  # character at which each token ends
            hit = np.zeros(len(seg), bool)
            for t in nlp(text[:200_000]):
                if t.tag_ in ("VBZ", "VBP") or (t.tag_ == "VBD" and t.lower_ in ("was", "were")):
                    s = _subject(t)
                    j = int(np.searchsorted(end, t.idx, side="right"))     # first token that overlaps the word
                    if s is not None and s.i < t.i and s.pos_ in ("NOUN", "PROPN", "NUM") and j < len(seg):
                        hit[j] |= (end[j - 1] if j else 0) < t.idx + len(t.text)
            items += [(r, a + 1 + j, seg[j], T.partner[seg[j]]) for j in np.flatnonzero(hit) if T.agr[seg[j]]]
    return np.array(items, np.int64).reshape(-1, 4)


def build_heldout(data_dir, T, n_rows=768, seed=20261018) -> dict:
    """The arrays of heldout.npz."""
    rng = np.random.default_rng(seed)
    weights = json.loads((Path(data_dir) / "plan.json").read_text())["weights"]
    raw = np.array([weights[k] for k in sorted(weights)]) * n_rows
    take = np.maximum(4, np.floor(raw).astype(int))
    while take.sum() > n_rows:
        take[np.argmax(take - raw)] -= 1
    while take.sum() < n_rows:
        take[np.argmax(raw - take)] += 1
    X = []
    for k, t in zip(sorted(weights), take):
        pool = np.memmap(Path(data_dir) / "heldout" / f"{k}.bin", np.uint16, "r").reshape(-1, ROW)
        X.append(np.asarray(pool[np.sort(rng.choice(len(pool), min(int(t), len(pool)), replace=False))]))
    X = np.concatenate(X).astype(np.int64)
    words = np.array([v for v, s in enumerate(T.strings[:L.N_TEXT])
                      if len(s) >= 3 and s[0] == " " and s[1:].isalpha() and 1000 <= v < 40000])
    n, pad = 255, lambda row: row + [L.EOS] * (ROW - len(row))  # noqa: E731
    copy_rows, copy_pos = [], []
    for r in range(32):                           # a row holds [separator, sequence, sequence] four times
        row = []
        while len(row) + 2 * n + 1 <= ROW:
            seq = list(rng.choice(words, n, replace=False))
            copy_pos += [(r, len(row) + 1 + n + i) for i in range(3, n)]
            row += [L.EOS] + seq + seq
        copy_rows.append(pad(row))
    spaced, bare = {}, {}                         # value -> token of a number with and without a leading space
    for v, s in enumerate(T.strings):
        if T.num[v] >= 0:
            (spaced if s.startswith(" ") else bare)[int(T.num[v])] = v
    fillers = rng.choice(words, 4096)
    succ_rows, succ_pos, row = [], [], []
    for it in range(480):                         # eight numbers; the third to the eighth are scored
        table, v0 = bare if it % 2 else spaced, int(rng.integers(1, 480))
        toks, where = [], []
        for k in range(8):
            toks += [187] if it % 2 else [L.COMMA] * (k > 0)       # "\n17. word" lines or " 17, 18" lists
            where.append(len(toks))
            toks += [table[v0 + k], L.DOT, int(fillers[(it * 8 + k) % 4096])] if it % 2 else [table[v0 + k]]
        cands = [[table.get(u, -1) for u in range(v0 + k - 1, v0 + k + 3)] for k in range(8)]
        if min(min(c) for c in cands) < 0:
            continue
        if len(row) + 1 + len(toks) > ROW:
            succ_rows, row = succ_rows + [pad(row)], []
        succ_pos += [(len(succ_rows), len(row) + 1 + where[k], *cands[k]) for k in range(2, 8)]
        row += [L.EOS] + toks
    return dict(rows=X.astype(np.uint16), cls=L.label(X.astype(np.uint16), T), agr=agreement_items(X, T),
                copy_rows=np.array(copy_rows, np.uint16), copy_pos=np.array(copy_pos, np.int64),
                succ_rows=np.array(succ_rows + [pad(row)], np.uint16), succ_pos=np.array(succ_pos, np.int64))


def build_closing(data_dir, T, vocab, exclude, n_items=1000, seed=20261023, window=256) -> dict:
    """The arrays of closing.npz: seqs (3 per window), the bracket type of each and whether the window is nested."""
    text_id = {}
    for v, s in enumerate(T.strings[:L.N_TEXT]):
        if s and vocab.encode(s) == [v]:
            text_id.setdefault(s, v)
    swap = np.full((L.VOCAB, 3), -1, np.int64)    # an opener token with the bracket of each type
    for v, s in enumerate(T.strings[:L.N_TEXT]):
        br = [ch for ch in s if ch in L.OPENERS or ch in L.CLOSERS]
        if len(br) == 1 and br[0] in L.OPENERS:
            ids = [text_id.get(s.replace(br[0], o)) for o in L.OPENERS]
            if None not in ids and ids[L.OPENERS.index(br[0])] == v:
                swap[v] = ids
    closer = [T.strings.index(c) for c in L.CLOSERS]
    rng = np.random.default_rng(seed)
    used = {r.astype(np.uint16).tobytes() for r in exclude}
    cands = []
    for path in sorted((Path(data_dir) / "heldout").glob("*.bin")):
        for x in np.memmap(path, np.uint16, "r").reshape(-1, ROW):
            if x.tobytes() in used:
                continue
            o, _, ds = L.label_detail(x, T)
            got = []
            for j in np.flatnonzero((o == L.CLOSE) & (ds >= 16) & (ds < window)):
                win = x[max(j - window + 1, 0):j + 1].astype(np.int64)
                oi = window - 1 - int(ds[j])
                if j >= window - 1 and swap[win[oi], 0] >= 0 and not (win == L.EOS).any():
                    got.append((win, oi, int(ds[j]), bool(np.isin(T.brk[win[oi + 1:-1]], (1, 2, 3)).any())))
            cands += [got[i] for i in rng.permutation(len(got))[:2]] if got else []
    seqs, kind, nested = [], [], []
    for i in rng.permutation(len(cands)):
        win, oi, d, nest = cands[i]
        var = []
        for t in range(3):
            y = win.copy()
            y[oi], y[-1] = swap[win[oi], t], closer[t]
            o, kd, ds = L.label_detail(y, T)
            var += [y] if o[-1] == L.CLOSE and kd[-1] == t and ds[-1] == d else []
        if len(var) == 3 and len(seqs) < 3 * n_items:
            seqs, kind, nested = seqs + var, kind + [0, 1, 2], nested + [nest] * 3
    return dict(seqs=np.array(seqs, np.uint16), kind=np.array(kind, np.int64), nested=np.array(nested, bool))


def _logits(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        return model(input_ids=x).logits


def _competence(name, lp):
    """Competence at each item from the log-probabilities (items, candidates) of its candidate tokens."""
    if name == "agreement":                       # the true form against the other-number form
        return torch.sigmoid(lp[:, 0] - lp[:, 1])
    if name == "repetition":
        return torch.exp(lp[:, 0])
    return torch.exp(lp[:, 1] - torch.logsumexp(lp, -1))          # counting: v among v - 1, v, v + 1, v + 2


class Evaluator:
    """The evaluation sets on a device and the measures of a model. Under DDP every rank scores its share of the
    batches and the sums are reduced over the ranks."""

    def __init__(self, eval_dir, vocab, tables, device):
        t = lambda a: torch.from_numpy(np.asarray(a).astype(np.int64))  # noqa: E731
        z, c = np.load(Path(eval_dir) / "heldout.npz"), np.load(Path(eval_dir) / "closing.npz")
        self.dev = torch.device(device)
        self.rows, self.cls, self.copy_rows, self.succ_rows = (t(z[k]) for k in ("rows", "cls", "copy_rows",
                                                                                 "succ_rows"))
        self.fcls = t(L.label_facts(z["rows"].astype(np.uint16), tables)[0])
        copied = self.copy_rows[tuple(t(z["copy_pos"]).T)]            # the true token of every repetition item
        self.items = {}
        for name, arr in (("agreement", t(z["agr"])), ("counting", t(z["succ_pos"])),
                          ("repetition", torch.cat([t(z["copy_pos"]), copied[:, None]], 1))):
            self.items[name] = {int(r): arr[arr[:, 0] == r, 1:].to(self.dev) for r in torch.unique(arr[:, 0])}
        self.closers = [torch.nonzero(t(tables.first) == k)[:, 0].to(self.dev) for k in range(3)]
        self.closing, self.kind, self.nested = t(c["seqs"]), t(c["kind"]), torch.from_numpy(c["nested"])
        self.prompts = {}
        for rel, p in build_prompts(vocab.encode).items():
            ids = torch.full((len(p["ids"]), max(len(i) for i in p["ids"])), L.EOS, dtype=torch.long)
            G = torch.zeros(len(p["groups"]), L.VOCAB)
            for j, i in enumerate(p["ids"]):
                ids[j, :len(i)] = torch.tensor(i)
            for g, grp in enumerate(p["groups"]):
                G[g, grp] = 1.0
            self.prompts[rel] = dict(ids=ids, last=t([len(i) - 1 for i in p["ids"]]), g=t(p["g"]), G=G.to(self.dev))

    def group_mass(self, model, rel, b):
        """(prompts of slice b, groups) probability mass on every answer group at the last prompt token, and the
        group of the true answer."""
        p = self.prompts[rel]
        x = p["ids"][b].to(self.dev)
        lg = _logits(model, x)[torch.arange(len(x), device=self.dev), p["last"][b].to(self.dev)]
        return torch.softmax(lg.float(), -1) @ p["G"].T, p["g"][b].to(self.dev)

    @torch.no_grad()
    def prompt_competence(self, model, rel, bs=256, rank=0, world=1, assign=None) -> float:
        """Mean over the prompts of the mass on the true answer group, or on the group that ``assign`` maps it
        to, over the mass on all groups."""
        n = len(self.prompts[rel]["ids"])
        q = torch.zeros(n, dtype=torch.float64, device=self.dev)     # a rank fills the prompts of its batches
        for i in range(rank * bs, n, bs * world):
            gm, g = self.group_mass(model, rel, slice(i, min(i + bs, n)))
            g = g if assign is None else assign.to(self.dev)[g]
            q[i:i + bs] = (gm.gather(1, g[:, None])[:, 0] / gm.sum(1)).double()
        if world > 1:
            torch.distributed.all_reduce(q)
        return float(q.mean())

    @torch.no_grad()
    def measure(self, model, bs=8, rank=0, world=1) -> dict:
        was, dev = model.training, self.dev
        model.eval()
        acc = {k: torch.zeros(2, dtype=torch.float64, device=dev) for k in KEYS}
        shard = lambda n, bs: [slice(i, min(i + bs, n)) for i in range(rank * bs, n, bs * world)]  # noqa: E731

        def add(k, v):
            acc[k][0] += v.double().sum()
            acc[k][1] += v.numel()

        for name, rows in (("agreement", self.rows), ("repetition", self.copy_rows), ("counting", self.succ_rows)):
            for b in shard(len(rows), bs):
                x = rows[b].to(dev)
                lp = torch.log_softmax(_logits(model, x[:, :-1]).float(), -1)
                if rows is self.rows:
                    nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
                    c, fc = self.cls[b, 1:].to(dev), self.fcls[b, 1:].to(dev)
                    add("loss", nll)
                    add("background_loss_structural", nll[c == 0])
                    add("background_loss", nll[(c == 0) & (fc == 0)])
                for j in range(len(x)):
                    it = self.items[name].get(b.start + j)        # columns: position, candidate tokens
                    if it is not None:
                        add("competence_" + name, _competence(name, lp[j, it[:, 0] - 1].gather(-1, it[:, 1:])))
        for b in shard(len(self.closing), 8 * bs):
            x = self.closing[b].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                lg = model.lm_head(model.model(input_ids=x[:, :-1]).last_hidden_state[:, -1])
            lp = torch.log_softmax(lg.float(), -1)
            g = torch.stack([torch.logsumexp(lp[:, ix], -1) for ix in self.closers], -1)
            q = torch.exp(g.gather(-1, self.kind[b].to(dev)[:, None])[:, 0] - torch.logsumexp(g, -1))
            add("competence_closing", q)
            add("competence_closing_nested", q[self.nested[b].to(dev)])
        if world > 1:
            for v in acc.values():
                torch.distributed.all_reduce(v)
        out = {k: float(v[0] / v[1]) if v[1] > 0 else math.nan for k, v in acc.items()}
        out["competence_capitals"] = self.prompt_competence(model, "cap", 32 * bs, rank, world)
        out["competence_city_countries"] = self.prompt_competence(model, "ctry", 32 * bs, rank, world)
        model.train(was)
        return out


@torch.no_grad()
def score_capitals(snapshot, eval_dir, vocab, tables, device, bs=256, loss_bs=4) -> dict:
    """Per-prompt masses of the capital prompts and the background loss of a stored snapshot. On CUDA the model
    is evaluated with bf16 weights."""
    from transformers import LlamaForCausalLM
    dev = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = LlamaForCausalLM.from_pretrained(snapshot, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
    E = Evaluator(eval_dir, vocab, tables, dev)
    masses = [E.group_mass(model, "cap", slice(i, i + bs)) for i in range(0, len(E.prompts["cap"]["ids"]), bs)]
    skill = torch.cat([gm.gather(1, g[:, None])[:, 0].double() for gm, g in masses]).cpu().numpy()
    total = torch.cat([gm.sum(1).double() for gm, _ in masses]).cpu().numpy()
    return dict(skill_mass=skill, other_mass=total - skill,
                background_loss=np.float64(E.measure(model, bs=loss_bs)["background_loss"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the web-text evaluation sets, or score capital prompts")
    ap.add_argument("command", choices=["build", "score-capitals"])
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--data", help="build: the staged pools")
    ap.add_argument("--eval", help="score-capitals: directory of the evaluation sets")
    ap.add_argument("--snapshot", help="score-capitals: snapshot directory of a run")
    ap.add_argument("--out", required=True, help="output directory (build) or .npz file (score-capitals)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    vocab = L.load_vocab(a.tokenizer)
    T = L.Tables(vocab)
    if a.command == "build":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        heldout = build_heldout(a.data, T)
        np.savez_compressed(Path(a.out) / "heldout.npz", **heldout)
        np.savez_compressed(Path(a.out) / "closing.npz", **build_closing(a.data, T, vocab, heldout["rows"]))
    else:
        np.savez(a.out, **score_capitals(a.snapshot, a.eval, vocab, T, a.device))


if __name__ == "__main__":
    main()
