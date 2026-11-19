"""Document filtering of three domains of web text (astronomy, Rust, Chinese): labels, classifier, reader, measure.

A document is a run of tokens between separators (50279) in a pool file of ``thinning.webtext.dolma``. Its
reference labels are functions of the decoded text:
  astro  astronomy by lexicon density: (strong terms + 0.3 x weak terms) per word of at least 0.01 with 4 distinct
         strong terms, or of at least 0.004 with 8, in a document of at least 50 words;
  rust   Rust by exclusive syntax: at least 4 distinct patterns of ``RUST_SYNTAX``;
  zh     Chinese by CJK share: Han, kana and hangul are at least 20% of the characters, with at least 50 Han
         characters, kana at most 10% of Han plus kana, and no more hangul than Han.
The filter is one linear classifier per domain over the hashed token unigrams and bigrams of a document's first 512
tokens, fitted to the reference labels of documents that no run reads. The retained fraction f of a domain is the
share of its reference-labelled tokens that remains in the stream: a run removes every document that scores at or
above the threshold of f, the largest one that leaves at most f.

    python -m thinning.webtext.docfilter index --data data/dolma --tokenizer tokenizer.json --out data/docfilter
    python -m thinning.webtext.docfilter fit --data data/dolma --out data/docfilter
    python -m thinning.webtext.docfilter terms --data data/dolma --tokenizer tokenizer.json --out data/docfilter

``index`` decodes every document of the staged pools (``tokenizer.json`` of ``EleutherAI/pythia-14m``) and writes
the start, length and labels of the documents of each pool file to ``<source>.<part>.npz``. ``fit`` trains the
classifiers on documents outside the rows that the runs read and writes classifier.npz, the scores of every document
(``<source>.<part>.scores.npy``) and thresholds.json. ``terms`` writes terms.npz: 96 rows per domain of unread
reference-labelled documents with the targets that require the domain skill (tokens of strong astronomy terms, Rust
idiom tokens, tokens at least 20 times as frequent in Chinese as in English documents).

``FilteredReader`` reads the rows of the data order, cuts out the removed documents and packs the remaining tokens
into rows of 2,048. Shares are 1 (nothing removed), 0.3, 0.1, 0.03, 0.01 or 0. Below the lowest share that a
domain's classifier sets alone (``LOWEST``), a run removes what that threshold removes and every reference-labelled
document except those whose hash is below f. Competence is term accuracy (``TermEvaluator``).
"""

import argparse
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np
import torch

from thinning.webtext import labels as L
from thinning.webtext.dolma import ROW, Pool
from thinning.webtext.evalsets import Evaluator, _logits
from thinning.webtext.train import Reader

DOMAINS = ("astro", "rust", "zh")                 # bits 1, 2, 4 of a document's labels; bit 8 marks English prose
NAMES, SALT = {"astro": "astronomy", "rust": "rust", "zh": "chinese"}, {"astro": 712, "rust": 723, "zh": 734}
LOWEST = {"astro": 0.1, "rust": 0.03, "zh": 0.01}  # lowest share set by the classifier threshold alone
LEVELS, PREFIX, HASH_SIZE = (0.3, 0.1, 0.03, 0.01), 512, 1 << 22
HORIZON = int(round(8e9 / (128 * ROW))) * 128     # instances of the data order of a run
READ_ROWS = int(4.7e9 // ROW)                     # rows of an order that count as read by the runs

AST_STRONG = frozenset("""astronomy astronomer astronomers astronomical astrophysics astrophysical astrophysicist
astrophysicists telescope telescopes galaxy galaxies galactic extragalactic nebula nebulae nebular supernova
supernovae supernovas quasar quasars pulsar pulsars exoplanet exoplanets exoplanetary interstellar intergalactic
cosmology cosmological redshift redshifts constellation constellations parsec parsecs kiloparsec kiloparsecs
megaparsec megaparsecs kpc mpc luminosity luminosities spectroscopic photometric photometry asteroid asteroids comet
comets meteorite meteorites meteoroid perihelion aphelion planetesimal planetesimals protoplanetary protostar
protostars protostellar circumstellar magnetar magnetars blazar blazars heliosphere heliocentric ecliptic
declination sidereal messier arcsecond arcseconds arcsec arcmin solstice equinox exomoon kuiper oort hubble jwst
spitzer chandrasekhar hertzsprung jupiter saturn uranus neptune pluto jovian saturnian brightest-star starlight
stargazing stargazers lightyear lightyears planetarium observatory observatories cepheid cepheids""".split())
AST_MULTI = """black hole,black holes,neutron star,neutron stars,white dwarf,white dwarfs,red giant,red dwarf,brown
dwarf,main sequence,milky way,light-year,light-years,light year,light years,dark matter,dark energy,solar system,solar
wind,solar flare,big bang,gravitational lensing,gravitational waves,cosmic microwave,event horizon,andromeda
galaxy,kepler space,space telescope,orbital period,star formation,star cluster,globular cluster,open cluster,dwarf
planet,binary star,variable star,accretion disk,accretion disc,right ascension,apparent magnitude,absolute
magnitude""".replace("\n", " ").split(",")
AST_WEAK = frozenset("""star stars stellar planet planets planetary orbit orbits orbital orbiting moon moons lunar
solar sun eclipse eclipses meteor meteors cosmic celestial nasa esa spacecraft venus mars mercury satellite""".split())
ENGLISH = "the of and to is that for with was on are this be by from".split()
RUST_SYNTAX = [r"\blet\s+mut\s+\w", r"\bfn\s+\w+\s*(?:<[^>\n]*>)?\s*\(", r"&mut\s+self\b", r"&self\b",
               r"\b(?:println|eprintln|format|vec|panic|assert_eq|assert_ne|writeln|unimplemented|todo)!\s*[\(\[]",
               r"#\[(?:derive|cfg|test|allow|inline|macro_use)\b", r"\buse\s+(?:std|crate|super|self|core)::",
               r"'static\b", r"\.unwrap\(\)", r"\bimpl(?:<[^>\n]*>)?\s+[\w:<>]+\s+for\s+\w",
               r"\bpub\s+(?:fn|struct|enum|trait|mod|crate)\b", r"\bpub\(crate\)", r"\bBox<dyn\b",
               r"\b(?:u8|u16|u32|u64|i32|i64|usize|isize|f64)\b\s*[,;)>=]", r"\bimpl\s+\w+\s*\{", r"\bmod\s+\w+\s*;",
               r"\bSome\(\w", r"\bOk\(\(\)\)", r"\bResult<", r"\bOption<", r"\bcargo\b", r"\.rs\b"]
RUST_IDIOM = frozenset("""fn mut impl let pub Some None Ok Err unwrap expect Vec String usize isize u8 u16 u32 u64 i8
i16 i32 i64 f32 f64 Self self match println format vec struct enum trait use mod crate derive Option Result Box Rc
Arc iter collect clone into as_ref where dyn ref move loop unsafe macro_rules :: => -> &mut &self 'a 'static""".split())
_WORD = re.compile(r"[^\W\d_]+")
_MULTI = re.compile("|".join(r"\b" + re.escape(m) + r"\b" for m in AST_MULTI))
_RUST = [re.compile(p) for p in RUST_SYNTAX]
_HAN, _KANA, _HANGUL = (re.compile(p) for p in ("[\u4e00-\u9fff]", "[\u3040-\u30ff]", "[\uac00-\ud7af]"))


def text_features(text: str) -> list:
    """Counts of a document: words, English stopwords, strong astronomy terms, distinct strong terms, weak terms,
    distinct Rust patterns, Han, kana and hangul characters, characters."""
    low = text.lower()
    words = Counter(_WORD.findall(low))
    strong, multi = [w for w in words if w in AST_STRONG], _MULTI.findall(low)
    code = "fn" in words or "let" in words or "impl" in words or "::" in text or "!(" in text
    return [sum(words.values()), sum(words[w] for w in ENGLISH), sum(words[w] for w in strong) + len(multi),
            len(strong) + len(set(multi)), sum(words[w] for w in AST_WEAK),
            sum(1 for p in _RUST if p.search(text)) if code else 0,
            len(_HAN.findall(text)), len(_KANA.findall(text)), len(_HANGUL.findall(text)), len(text)]


def reference_labels(feats) -> np.ndarray:
    """Label bits of documents from their ``text_features`` (n, 10), in float32: 1 astronomy, 2 Rust, 4 Chinese,
    8 English prose without CJK characters (the comparison documents of the Chinese terms)."""
    words, english, strong, distinct, weak, rust, han, kana, hangul, chars = np.asarray(feats, np.float32).T
    density = (strong + 0.3 * weak) / np.maximum(words, 1)
    astro = (((density >= 0.01) & (distinct >= 4)) | ((density >= 0.004) & (distinct >= 8))) & (words >= 50)
    cjk = han + kana + hangul
    zh = (cjk / np.maximum(chars, 1) >= 0.2) & (kana <= 0.1 * (han + kana)) & (hangul <= han) & (han >= 50)
    prose = (english >= 20) & (cjk == 0) & (words >= 100)
    return (astro * 1 + (rust >= 4) * 2 + zh * 4 + prose * 8).astype(np.uint8)


def pool_files(pool) -> list:
    """(source index, source, part, flat tokens, first pool row of the part) of every file of the pool."""
    return [(si, k, j, part.reshape(-1), sum(len(p) for p in pool.parts[k][:j]))
            for si, k in enumerate(pool.keys) for j, part in enumerate(pool.parts[k])]


def _label_chunk(job) -> np.ndarray:
    path, tokenizer, start, length = job
    from tokenizers import Tokenizer
    tokens = np.memmap(path, np.uint16, "r")
    texts = Tokenizer.from_file(tokenizer).decode_batch([tokens[s:s + n].tolist() for s, n in zip(start, length)])
    return reference_labels(np.array([text_features(t) for t in texts], np.float32).reshape(-1, 10))


def build_index(pool, tokenizer, out, workers=4, chunk=8192, block=1 << 28):
    """Write ``<out>/<source>.<part>.npz`` per pool file: start, len and labels of its non-empty documents."""
    with ProcessPoolExecutor(workers) as ex:
        for _, k, j, tokens, _ in pool_files(pool):
            eos = np.concatenate([np.flatnonzero(np.asarray(tokens[a:a + block]) == L.EOS) + a
                                  for a in range(0, len(tokens), block)])
            start, end = np.r_[0, eos + 1], np.r_[eos, len(tokens)]
            start, length = start[end > start], (end - start)[end > start]
            jobs = [(tokens.filename, str(tokenizer), start[a:a + chunk], length[a:a + chunk])
                    for a in range(0, len(start), chunk)]
            np.savez(Path(out) / f"{k}.{j}.npz", start=start, len=length.astype(np.int32),
                     labels=np.concatenate(list(ex.map(_label_chunk, jobs))))


def _mix(x: np.ndarray, salt: int = 0) -> np.ndarray:
    """splitmix64 of uint64 keys, salted."""
    with np.errstate(over="ignore"):
        golden = np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (np.uint64(salt) * golden)) + golden
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def unread_documents(pool, out) -> dict:
    """The documents that touch no row among the first 4.7B tokens of the 8B-token data orders of seeds 0 and 1 (another
    horizon gives another order). Flat arrays in the order of ``pool_files``: file, start, len, labels and split, a hash
    of the document's place in 0..9. Splits 0-5 fit the classifiers; 6-7 give the rows of terms.npz."""
    read = {k: np.zeros(pool.n_rows[k], bool) for k in pool.keys}
    for seed in (0, 1):
        order = pool.order(HORIZON, 1_000_003 * (seed + 1))[:READ_ROWS]
        for i, k in enumerate(pool.keys):
            read[k][order[order[:, 0] == i, 1]] = True
    cols = {c: [] for c in ("file", "start", "len", "labels", "split")}
    for n, (si, k, j, tokens, first) in enumerate(pool_files(pool)):
        z = np.load(Path(out) / f"{k}.{j}.npz")
        start, length, before = z["start"], z["len"].astype(np.int64), np.r_[0, np.cumsum(read[k])]
        last = first + np.minimum((start + length) // ROW, len(tokens) // ROW - 1)
        doc = np.flatnonzero(before[last + 1] == before[first + start // ROW])
        key = (np.uint64(si) << np.uint64(56)) ^ (np.uint64(j) << np.uint64(48)) ^ start[doc].astype(np.uint64)
        for c, v in zip(cols, (np.full(len(doc), n), start[doc], length[doc], z["labels"][doc], _mix(key) % 10)):
            cols[c].append(v)
    return {c: np.concatenate(v) for c, v in cols.items()}


def _tokens(files, docs, i, cap) -> np.ndarray:
    return np.asarray(files[docs["file"][i]][3][docs["start"][i]:][:min(docs["len"][i], cap)]).astype(np.int64)


@nb.njit(cache=True)
def hashed(x, n):
    """Feature ids of the tokens x[:n], n >= 1: the token ids and a hash in [vocabulary, HASH_SIZE) per bigram."""
    out = np.empty(2 * n - 1, np.int64)
    out[0] = x[0]
    for i in range(1, n):
        h = (x[i - 1] * 1000003 + x[i]) * 2654435761
        out[2 * i - 1], out[2 * i] = x[i], L.VOCAB + (h & 0x7FFFFFFFFFFF) % (HASH_SIZE - L.VOCAB)
    return out


@nb.njit(parallel=True, cache=True)
def score_documents(tokens, start, length, w, b, prefix):
    """Classifier score of every document: the bias plus the mean weight of the features of its first tokens."""
    out = np.empty(len(start), np.float32)
    for i in nb.prange(len(start)):
        n = min(length[i], prefix)
        f = hashed(tokens[start[i]:start[i] + n].astype(np.int64), n)
        s = 0.0
        for j in range(len(f)):
            s += w[f[j]]
        out[i] = b + s / len(f)
    return out


def fit_classifiers(pool, out, negatives=600_000) -> list:
    """Fit the classifiers on the unread documents of splits 0-5 (every reference-labelled one and a sample of the
    others; a fifth held out): logistic regression on the mean feature weight, positives weighted by sqrt(negatives
    / positives), Adam at 0.1, 6 epochs. Writes classifier.npz and returns the held-out average precisions."""
    files, docs = pool_files(pool), unread_documents(pool, out)
    sample = np.flatnonzero(docs["split"] < 6)
    ref = docs["labels"][sample] & 7 > 0
    rest = np.random.default_rng(1).choice(np.flatnonzero(~ref), min(int((~ref).sum()), negatives), replace=False)
    sample = sample[np.sort(np.concatenate([np.flatnonzero(ref), rest]))]
    bags = [torch.from_numpy(hashed(x, len(x))) for x in (_tokens(files, docs, i, PREFIX) for i in sample)]
    size, rng = torch.tensor([len(v) for v in bags]), np.random.default_rng(0)
    perm = rng.permutation(len(sample))
    held, train = perm[:len(sample) // 5], perm[len(sample) // 5:]
    w, b, ap = [], [], []
    for c in range(len(DOMAINS)):
        y = torch.from_numpy((docs["labels"][sample] >> c & 1).astype(np.float32))
        emb, bias = torch.nn.EmbeddingBag(HASH_SIZE, 1, mode="mean", sparse=True), torch.zeros(1, requires_grad=True)
        torch.nn.init.zeros_(emb.weight)
        score = lambda rows: emb(torch.cat([bags[r] for r in rows]),  # noqa: E731  one bag per document
                                 torch.cumsum(size[rows], 0) - size[rows])[:, 0] + bias
        opts = [torch.optim.SparseAdam(list(emb.parameters()), lr=0.1), torch.optim.Adam([bias], lr=0.1)]
        pos = float(y[train].sum())
        pos_weight = torch.tensor((len(train) - pos) / max(pos, 1.0)) ** 0.5
        for _ in range(6):
            order = rng.permutation(train)
            for a in range(0, len(order), 4096):
                rows = np.sort(order[a:a + 4096])
                torch.nn.functional.binary_cross_entropy_with_logits(score(rows), y[rows],
                                                                     pos_weight=pos_weight).backward()
                for o in opts:
                    o.step()
                    o.zero_grad()
        with torch.no_grad():
            chunks = [np.sort(held[a:a + 8192]) for a in range(0, len(held), 8192)]
            z = np.concatenate([score(rows).numpy() for rows in chunks])
        hit = (y.numpy()[np.concatenate(chunks)] > 0.5)[np.argsort(-z)]
        ap.append(float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit].mean()) if hit.any() else math.nan)
        w, b = w + [emb.weight.detach().numpy()[:, 0].astype(np.float32)], b + [np.float32(bias.item())]
    np.savez(Path(out) / "classifier.npz", w=np.stack(w), b=np.array(b))
    return ap


def thresholds(score, ref, weight) -> dict:
    """{f: threshold, share, removed_share_of_stream} of one domain from the score, the reference label and the
    expected stream tokens of every document. The threshold of f, midway between two neighbouring scores and
    rounded to 6 decimals, is the largest that leaves at most a share f of the reference-labelled tokens."""
    o = np.argsort(score)
    s, w = score[o], np.where(ref[o], weight[o], 0.0)
    left, out = np.cumsum(w) / w.sum(), {}
    for f in LEVELS:
        i = max(int(np.searchsorted(left, f, side="right")) - 1, 0)
        tau = round(float((s[i] + s[min(i + 1, len(s) - 1)]) / 2), 6)
        gone = score >= tau
        out[str(f)] = dict(threshold=tau, share=float(weight[ref & ~gone].sum() / weight[ref].sum()),
                           removed_share_of_stream=float(weight[gone].sum() / weight.sum()))
    return out


def score_pool(pool, out, average_precision=(math.nan,) * 3):
    """Score every document with classifier.npz and write the scores and thresholds.json. A document weighs its
    length times the share of the rows of its source that the data order of a run draws."""
    out, clf, counts = Path(out), np.load(Path(out) / "classifier.npz"), pool.counts(HORIZON)
    weight, labels, scores = [], [], []
    for _, k, j, tokens, _ in pool_files(pool):
        z = np.load(out / f"{k}.{j}.npz")
        scores.append(np.stack([score_documents(tokens, z["start"], z["len"].astype(np.int64), clf["w"][c],
                                                float(clf["b"][c]), PREFIX) for c in range(len(DOMAINS))], 1))
        np.save(out / f"{k}.{j}.scores.npy", scores[-1])
        weight.append(counts[k] / pool.n_rows[k] * z["len"].astype(np.float64))
        labels.append(z["labels"])
    weight, labels, scores = np.concatenate(weight), np.concatenate(labels), np.concatenate(scores)
    refs = [labels >> c & 1 > 0 for c in range(len(DOMAINS))]
    report = {d: dict(average_precision=average_precision[c], share_of_stream=float(weight[r].sum() / weight.sum()),
                      levels=thresholds(scores[:, c], r, weight)) for c, (d, r) in enumerate(zip(DOMAINS, refs))}
    (out / "thresholds.json").write_text(json.dumps(report, indent=1))


class FilteredReader(Reader):
    """``Reader`` on the filtered stream: the tokens of the data order without the removed documents and the
    separators after them, in batches. ``shares`` is ``"rust=0.03,astro=1,zh=0.3"`` (1 for a domain that is not
    named). Labels are computed on the packed rows and thinning uses the position of each token in the unfiltered
    stream. Batches come in order; ``rows_read`` counts the rows of the data order that have been consumed."""

    def __init__(self, *args, filter_dir, shares: str):
        super().__init__(*args)
        given = {k: float(v) for k, v in (p.split("=") for p in shares.split(",") if p)}
        if not set(given) <= set(DOMAINS) or not set(given.values()) <= {1.0, 0.0, *LEVELS}:
            raise SystemExit(f"--filter takes {', '.join(DOMAINS)} with shares 1, 0.3, 0.1, 0.03, 0.01 or 0")
        levels, self.docs = json.loads((Path(filter_dir) / "thresholds.json").read_text()), {}
        for si, k, j, tokens, first in pool_files(self.pool):
            z, scores = np.load(Path(filter_dir) / f"{k}.{j}.npz"), np.load(Path(filter_dir) / f"{k}.{j}.scores.npy")
            start, gone = z["start"].astype(np.int64), np.zeros(len(scores), bool)
            for c, d in enumerate(DOMAINS):
                f = given.get(d, 1.0)
                if f < 1.0:
                    flagged = scores[:, c] >= levels[d]["levels"][str(max(f, LOWEST[d]))]["threshold"]
                    ref = z["labels"] >> c & 1 > 0    # kept below LOWEST: a reference document with hash < f
                    h = (_mix(((si << 44) | (j << 43) | start).astype(np.uint64), SALT[d]) >> np.uint64(11)) / 2**53
                    gone |= flagged if f >= LOWEST[d] else (flagged | ref) & ~(ref & (h < f))
            spans = np.stack([start[gone], np.minimum(start + z["len"] + 1, len(tokens))[gone]], 1) + first * ROW
            self.docs[si] = np.concatenate([self.docs.get(si, spans[:0]), spans])
        self.next, self.rows_read, self.held = 0, 0, (np.zeros(0, np.int64),) * 2

    def _kept(self, s, r):
        """The tokens of pool row r of source s outside the removed documents, and their stream positions."""
        gone, keep, p0 = self.docs[s], np.ones(ROW, bool), r * ROW
        for a, b in gone[np.searchsorted(gone[:, 1], p0, side="right"):np.searchsorted(gone[:, 0], p0 + ROW)]:
            keep[max(a - p0, 0):b - p0] = False
        return self.pool.row(s, r).astype(np.int64)[keep], ((s << 40) + p0 + np.arange(ROW, dtype=np.int64))[keep]

    def rows(self, i):
        if i < self.next:
            raise ValueError("the filtered stream is read in order")
        n = self.batch * ROW
        while self.next <= i:
            held, size = [self.held], len(self.held[0])
            while size < n:
                held.append(self._kept(*(int(v) for v in self.order[self.rows_read])))
                self.rows_read, size = self.rows_read + 1, size + len(held[-1][0])
            x, pos = (np.concatenate(v) for v in zip(*held))
            self.held, self.next = (x[n:], pos[n:]), self.next + 1
        return tuple(v[:n].reshape(self.batch, ROW)[self.lo:self.hi] for v in (x, pos))


def word_marks(ids, strings, lexicon) -> np.ndarray:
    """True for the tokens that hold a letter of a word (a run of letters, lower-cased) of the lexicon."""
    marks, word, toks = np.zeros(len(ids), bool), "", []
    for t, v in enumerate(list(ids) + [-1]):
        for ch in strings[v] if v >= 0 else " ":
            if ch.isalpha():
                word += ch
                toks.append(t)
            else:
                if word.lower() in lexicon:
                    marks[toks] = True
                word, toks = "", []
    return marks


def build_terms(pool, out, strings, n_rows=96, cap=1024) -> dict:
    """The arrays of terms.npz: per domain ``rows_<d>`` (n_rows, 2048), a seeded sample of the unread
    reference-labelled documents of splits 6-7, the first ``cap`` tokens of each, with one separator between
    documents; and ``term_<d>`` (n_rows, 2047), the targets inside a document that require the domain skill."""
    files, docs = pool_files(pool), unread_documents(pool, out)
    fit, lab = docs["split"] < 6, docs["labels"]
    rng, counts = np.random.default_rng(3), []
    for sel in (fit & (lab & 4 > 0), fit & (lab & 8 > 0) & (lab & 6 == 0)):       # Chinese, then English prose
        ix = np.flatnonzero(sel)
        counts.append(sum(np.bincount(_tokens(files, docs, i, ROW), minlength=L.VOCAB)
                          for i in ix[rng.permutation(len(ix))[:20000]]))
    (zh, en), text = counts, np.array([s.strip() != "" for s in strings])
    vocab = {"zh": (zh >= 20) & (zh / zh.sum() >= 20 * ((en + 1) / en.sum())) & text,
             "rust": np.array([s.strip() in RUST_IDIOM for s in strings])}
    rng, arrays = np.random.default_rng(11), {}
    for c, d in enumerate(DOMAINS):
        ix = np.flatnonzero((docs["split"] >= 6) & (docs["split"] <= 7) & (lab >> c & 1 > 0))
        flat, inside, p = np.full(n_rows * ROW, L.EOS, np.int64), np.zeros(n_rows * ROW, bool), 0
        for i in ix[rng.permutation(len(ix))][:4 * n_rows * ROW // 200]:
            x = _tokens(files, docs, i, cap)[:len(flat) - p]
            flat[p:p + len(x)], inside[p:p + len(x)] = x, True
            p += len(x)
            p += p % ROW != 0                     # one separator unless the document ends a row
        X, inside = flat.reshape(n_rows, ROW), inside.reshape(n_rows, ROW)
        term = np.stack([word_marks(x, strings, AST_STRONG) for x in X]) if d == "astro" else vocab[d][X]
        arrays[f"rows_{d}"], arrays[f"term_{d}"] = X.astype(np.uint16), term[:, 1:] & inside[:, 1:] & inside[:, :-1]
    return arrays


@torch.no_grad()
def term_accuracy(model, rows, term, device, bs=8, rank=0, world=1) -> float:
    """Top-1 next-token accuracy at the marked targets; term[r, c] marks the target rows[r, c + 1]."""
    was, acc = model.training, torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    for i in range(rank * bs, len(rows), bs * world):
        x, m = rows[i:i + bs].to(device), term[i:i + bs].to(device)
        hit = _logits(model, x[:, :-1]).float().argmax(-1) == x[:, 1:]
        acc += torch.stack([hit[m].sum(), m.sum()])
    if world > 1:
        torch.distributed.all_reduce(acc)
    model.train(was)
    return float(acc[0] / acc[1]) if acc[1] > 0 else math.nan


class TermEvaluator(Evaluator):
    """The measures of ``Evaluator`` and ``competence_astronomy``, ``competence_rust`` and ``competence_chinese``,
    the term accuracy on the rows of terms.npz (``terms``)."""

    def __init__(self, terms, *args):
        super().__init__(*args)
        self.terms = {k: torch.from_numpy(v.astype(np.int64) if k[0] == "r" else v) for k, v in np.load(terms).items()}

    def measure(self, model, bs=8, rank=0, world=1) -> dict:
        accuracy = lambda d: term_accuracy(model, self.terms["rows_" + d], self.terms["term_" + d],  # noqa: E731
                                           self.dev, bs, rank, world)
        return dict(super().measure(model, bs, rank, world), **{"competence_" + NAMES[d]: accuracy(d) for d in DOMAINS})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Document filter of web text: index, classifiers and evaluation rows")
    ap.add_argument("command", choices=["index", "fit", "terms"])
    ap.add_argument("--data", required=True, help="staged pools (thinning.webtext.dolma)")
    ap.add_argument("--out", required=True, help="directory of the filter files")
    ap.add_argument("--tokenizer", help="tokenizer.json (index, terms)")
    ap.add_argument("--workers", type=int, default=4, help="index: processes that decode documents")
    a = ap.parse_args(argv)
    pool = Pool(a.data)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.command == "index":
        build_index(pool, a.tokenizer, a.out, a.workers)
    elif a.command == "fit":
        score_pool(pool, a.out, fit_classifiers(pool, a.out))
    else:
        np.savez_compressed(Path(a.out) / "terms.npz", **build_terms(pool, a.out, L.load_vocab(a.tokenizer).strings))


if __name__ == "__main__":
    main()
