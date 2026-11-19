"""Skill labels and thinning masks for web text, on rows of 2,048 GPT-NeoX tokens.

Every label is a function of one training row. State resets at the start of the row and at every document separator.

Structural classes (at most one per token, by priority CLOSE > AGR > SUCC > COPY):
  CLOSE  closing bracket: a token whose text starts with ``)``, ``]`` or ``}`` while the top of the document's
         bracket stack is the opener of the same type (every bracket character of every token updates the stack);
  AGR    number-marked auxiliary: `` is``, `` are``, `` was``, `` were``, `` has`` or `` have``, every occurrence;
  SUCC   counting sequence: a whole number v >= 1 written as one token (1-4 digits, not part of a longer number,
         a decimal or a thousands group) whose most recent whole number, at most 64 tokens back, is v - 1;
  COPY   repeated 4-gram completion: the last token of a 4-gram that occurred earlier in the document, its first
         occurrence ending at least 64 tokens before.

Factual classes (``entities``): CAP marks every token of a capital name whose country was named at most 128 tokens
earlier in the document, CTRY every token of a country name whose city was named at most 128 tokens earlier. Names
match as token sequences with or without a leading space and must end at a word boundary.

Thinning: the target at a labelled token stays in the loss iff hash(p, salt of the class) < f, where p is the stream
position of the token, (source index << 40) + row * 2048 + column, for structural classes, and the position of the
first token of the answer mention for factual classes, so a mention is kept or removed as a whole. The sets retained
at different fractions are nested, and a row has the same masks in every run that reads it.

The vocabulary tables are built from a tokenizer file (``tokenizer.json`` of ``EleutherAI/pythia-14m``; the
tokenizer of the DataDecide models gives the same tables).
"""

import re

import numba as nb
import numpy as np

from thinning.masks import position_hash
from thinning.webtext import entities

ROW, VOCAB, EOS = 2048, 50304, 50279
N_TEXT = 50277                                   # ids 1..N_TEXT-1 carry text; 0 and the rest are special tokens
CLOSE, AGR, SUCC, COPY = 1, 2, 3, 4
CAP, CTRY = 1, 2
STRUCTURAL = {"close": CLOSE, "agr": AGR, "succ": SUCC, "copy": COPY}
FACTUAL = {"cap": CAP, "ctry": CTRY}
SALT = {CLOSE: 101, AGR: 202, SUCC: 303, COPY: 404}
FACT_SALT = {CAP: 505, CTRY: 606}
AUX_IDS = (310, 403, 369, 497, 556, 452)         # is, are, was, were, has, have; number partners are adjacent
AUX_WORDS = (" is", " are", " was", " were", " has", " have")
SUCC_WINDOW, COPY_MIN_DIST = 64, 64
DOT, COMMA = 15, 13
OPENERS, CLOSERS = "([{", ")]}"
MAX_NAME = 10                                    # longest name, in tokens


class Vocab:
    """Token strings (one per id) and the text encoder of the tokenizer."""

    def __init__(self, strings, encode):
        self.strings, self.encode = list(strings), encode


def load_vocab(tokenizer_json) -> Vocab:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    strings = [tok.decode([v]) if 0 < v < N_TEXT else "" for v in range(VOCAB)]
    return Vocab(strings, lambda text: tok.encode(text).ids)


class Tables:
    """Vocabulary tables of the two labellers."""

    def __init__(self, vocab: Vocab):
        S = self.strings = vocab.strings
        self.brk = np.zeros((VOCAB, 8), np.int8)          # bracket characters: 1..3 open ( [ {, 4..6 close ) ] }
        self.brk_n = np.zeros(VOCAB, np.int8)
        self.first = np.full(VOCAB, -1, np.int8)          # closer type of a token that starts with ) ] }
        self.num = np.full(VOCAB, -1, np.int32)           # value of a whole-number token
        self.dig_start = np.zeros(VOCAB, np.bool_)
        self.dig_end = np.zeros(VOCAB, np.bool_)
        for v, s in enumerate(S):
            chars = [ch for ch in s if ch in OPENERS or ch in CLOSERS]
            for k, ch in enumerate(chars):
                self.brk[v, k] = OPENERS.index(ch) + 1 if ch in OPENERS else CLOSERS.index(ch) + 4
            self.brk_n[v] = len(chars)
            if s and s[0] in CLOSERS:
                self.first[v] = CLOSERS.index(s[0])
            if re.fullmatch(r" ?[0-9]{1,4}", s):
                self.num[v] = int(s)
            self.dig_start[v] = bool(s) and s[0].isdigit()
            self.dig_end[v] = bool(s) and s[-1].isdigit()
        if any(S[i] != w for w, i in zip(AUX_WORDS, AUX_IDS)):
            raise ValueError("the tokenizer does not have the GPT-NeoX vocabulary")
        self.agr = np.zeros(VOCAB, np.bool_)
        self.agr[list(AUX_IDS)] = True
        self.partner = np.arange(VOCAB, dtype=np.int64)   # the other-number form of an auxiliary
        for a, b in zip(AUX_IDS[0::2], AUX_IDS[1::2]):
            self.partner[a], self.partner[b] = b, a
        self.alnum_start = np.array([bool(s) and s[0].isalnum() for s in S], np.bool_)
        self.alnum_end = np.array([bool(s) and s[-1].isalnum() for s in S], np.bool_)
        self._name_patterns(vocab)

    def _name_patterns(self, vocab):
        """Token sequences of every name, with (relation, role, unit, without leading space); role 0 is a subject,
        1 an answer. Patterns are grouped by first token, longest first."""
        units = entities.units()
        pats = {}
        for name, rel in FACTUAL.items():
            for ui, u in enumerate(units[name]):
                for role, names in ((0, u["subj"]), (1, u["ans"])):
                    for nm in names:
                        nm = nm[4:] if nm.startswith("the ") else nm
                        for nospace in (0, 1):
                            ids = tuple(vocab.encode(nm if nospace else " " + nm))
                            if not 0 < len(ids) <= MAX_NAME:
                                raise ValueError(f"name {nm!r} has {len(ids)} tokens")
                            pats[(ids, (rel, role, ui, nospace))] = None
        pats = sorted(pats, key=lambda p: (p[0][0], -len(p[0])))
        self.pat = np.full((len(pats), MAX_NAME), -1, np.int64)
        self.pat_len = np.array([len(ids) for ids, _ in pats], np.int64)
        self.pat_meta = np.array([meta for _, meta in pats], np.int64)
        self.pat_start = np.full(VOCAB, -1, np.int64)     # first pattern that begins with a token, and how many
        self.pat_count = np.zeros(VOCAB, np.int64)
        for i, (ids, _) in enumerate(pats):
            self.pat[i, :len(ids)] = ids
            if self.pat_start[ids[0]] < 0:
                self.pat_start[ids[0]] = i
            self.pat_count[ids[0]] += 1
        self.max_units = max(len(u) for u in units.values())


@nb.njit(cache=True)
def _label_row(x, brk, brk_n, first, agr, num, dig_start, dig_end, out, kind, dist):
    """One row. out: class per token; kind, dist: bracket type and opener distance of the closing brackets."""
    n = x.shape[0]
    for i in range(n):
        out[i] = 0
        kind[i] = -1
        dist[i] = 0
    st_t = np.empty(8 * n, np.int8)
    st_p = np.empty(8 * n, np.int64)
    sp = 0
    for i in range(n):
        v = x[i]
        if v == EOS:
            sp = 0
            continue
        for k in range(brk_n[v]):
            c = brk[v, k]
            if c <= 3:
                st_t[sp] = c
                st_p[sp] = i
                sp += 1
            elif sp > 0 and st_t[sp - 1] == c - 3:
                if k == 0 and first[v] >= 0:
                    out[i] = CLOSE
                    kind[i] = first[v]
                    dist[i] = i - st_p[sp - 1]
                sp -= 1
    for i in range(n):
        if out[i] == 0 and agr[x[i]]:
            out[i] = AGR
    last_v = -1
    last_i = -1000000
    for i in range(n):
        v = x[i]
        if v == EOS:
            last_v = -1
            last_i = -1000000
            continue
        val = num[v]
        if val < 0:
            continue
        if i > 0 and dig_end[x[i - 1]] and x[i - 1] != EOS:
            continue                                      # inside a longer number
        if i > 1 and (x[i - 1] == DOT or x[i - 1] == COMMA) and dig_end[x[i - 2]]:
            continue                                      # decimal or thousands part
        if i + 1 < n and dig_start[x[i + 1]]:
            continue
        if val >= 1 and last_v == val - 1 and i - last_i <= SUCC_WINDOW and out[i] == 0:
            out[i] = SUCC
        last_v = val
        last_i = i
    seg = np.empty(n, np.int64)
    s = 0
    for i in range(n):
        if x[i] == EOS:
            s += 1
        seg[i] = s
    h = np.empty(n, np.int64)                             # key of the 4-gram that ends at i, within its document
    for i in range(n):
        if i < 3 or x[i] == EOS or x[i - 1] == EOS or x[i - 2] == EOS or x[i - 3] == EOS:
            h[i] = -1 - i
        else:
            z = ((x[i - 3] * 50331653 + x[i - 2]) * 50331653 + x[i - 1]) * 50331653 + x[i]
            z = z ^ (seg[i] << 47)
            h[i] = z & 0x3FFFFFFFFFFFFFFF
    order = np.argsort(h, kind="mergesort")
    j = 0
    while j < n:
        k = j
        while k + 1 < n and h[order[k + 1]] == h[order[j]]:
            k += 1
        if k > j and h[order[j]] >= 0:
            p0 = order[j]
            for q in range(j + 1, k + 1):
                t = order[q]
                if t - p0 >= COPY_MIN_DIST and out[t] == 0:
                    out[t] = COPY
        j = k + 1


@nb.njit(cache=True, parallel=True)
def _label_rows(X, brk, brk_n, first, agr, num, dig_start, dig_end, out):
    for r in nb.prange(X.shape[0]):
        o = np.empty(X.shape[1], np.uint8)
        kd = np.empty(X.shape[1], np.int8)
        ds = np.empty(X.shape[1], np.int32)
        _label_row(X[r].astype(np.int64), brk, brk_n, first, agr, num, dig_start, dig_end, o, kd, ds)
        out[r] = o


def label(X: np.ndarray, T: Tables) -> np.ndarray:
    """Structural class of every token of rows X (R, n) uint16."""
    out = np.empty(X.shape, np.uint8)
    _label_rows(np.ascontiguousarray(X), T.brk, T.brk_n, T.first, T.agr, T.num, T.dig_start, T.dig_end, out)
    return out


def label_detail(x: np.ndarray, T: Tables):
    """One row -> (class, bracket type of a closing bracket, distance to its opener)."""
    o, kd, ds = np.empty(len(x), np.uint8), np.empty(len(x), np.int8), np.empty(len(x), np.int32)
    _label_row(x.astype(np.int64), T.brk, T.brk_n, T.first, T.agr, T.num, T.dig_start, T.dig_end, o, kd, ds)
    return o, kd, ds


@nb.njit(cache=True)
def _fact_row(x, pat, pat_len, meta, start, count, an_start, an_end, max_units, window, cls, span):
    """One row. cls: factual class per token; span: column of the first token of its answer mention (-1 none)."""
    n = x.shape[0]
    for i in range(n):
        cls[i] = 0
        span[i] = -1
    last = np.full((3, max_units), -1000000, np.int64)    # end of the latest subject mention per (relation, unit)
    for i in range(n):
        v = x[i]
        if v == EOS:
            last[:, :] = -1000000
            continue
        s0 = start[v]
        if s0 < 0:
            continue
        for j in range(s0, s0 + count[v]):
            L = pat_len[j]
            if i + L > n:
                continue
            ok = True
            for k in range(1, L):
                if x[i + k] != pat[j, k] or x[i + k] == EOS:
                    ok = False
                    break
            if not ok:
                continue
            if i + L < n and an_start[x[i + L]]:
                continue                                  # not at a word end
            rel, role, u, nospace = meta[j, 0], meta[j, 1], meta[j, 2], meta[j, 3]
            if nospace == 1 and i > 0 and an_end[x[i - 1]] and x[i - 1] != EOS:
                continue                                  # inside a word
            if role == 0:
                last[rel, u] = i + L - 1
            elif i - last[rel, u] <= window and cls[i] == 0:
                for k in range(L):
                    cls[i + k] = rel
                    span[i + k] = i


@nb.njit(cache=True, parallel=True)
def _fact_rows(X, pat, pat_len, meta, start, count, an_start, an_end, max_units, window, cls, span):
    for r in nb.prange(X.shape[0]):
        c = np.empty(X.shape[1], np.uint8)
        s = np.empty(X.shape[1], np.int32)
        _fact_row(X[r].astype(np.int64), pat, pat_len, meta, start, count, an_start, an_end, max_units, window, c, s)
        cls[r] = c
        span[r] = s


def label_facts(X: np.ndarray, T: Tables, window: int = entities.WINDOW):
    """Factual class of every token of rows X (R, n) uint16, and the first column of its answer mention."""
    cls, span = np.empty(X.shape, np.uint8), np.empty(X.shape, np.int32)
    _fact_rows(np.ascontiguousarray(X), T.pat, T.pat_len, T.pat_meta, T.pat_start, T.pat_count, T.alnum_start,
               T.alnum_end, T.max_units, window, cls, span)
    return cls, span


def stream_positions(src: np.ndarray, row: np.ndarray) -> np.ndarray:
    """(n, 2048) stream position of every token of pool rows (source index, row)."""
    base = (src.astype(np.int64) << 40) + row.astype(np.int64) * ROW
    return base[:, None] + np.arange(ROW, dtype=np.int64)[None, :]


def thinned(lab, fcls, fspan, pos, keep: dict, fkeep: dict) -> np.ndarray:
    """True where the target at a token is removed from the loss. lab, fcls, fspan: labels of the rows; pos: the
    stream position of every token; keep {structural class: f}; fkeep {factual class: f}."""
    drop = np.zeros(lab.shape, bool)
    for k, f in keep.items():
        sel = lab == k
        if f < 1.0 and sel.any():
            drop[sel] = position_hash(pos[sel], SALT[k]) >= f
    for k, f in fkeep.items():
        ri, ci = np.nonzero(fcls == k)
        if f < 1.0 and len(ri):
            drop[ri, ci] |= position_hash(pos[ri, fspan[ri, ci].astype(np.int64)], FACT_SALT[k]) >= f
    return drop


def parse_fractions(spec: str, classes: dict) -> dict:
    """``"close=0.3,agr=0.05"`` -> {class: retained fraction}, 1 for the classes that are not named."""
    out = {code: 1.0 for code in classes.values()}
    for part in filter(None, spec.split(",")):
        name, f = part.split("=")
        out[classes[name]] = float(f)
    return out
