"""Web text: FineWeb-Edu, with subject-verb agreement as the skill.

Corpus. The ``sample/10BT`` subset of ``HuggingFaceFW/fineweb-edu``: files 000-003, in order, are the training
stream, and the first 6,000 documents of file 013 are held out. Text is tokenized with the Pythia BPE tokenizer
(``EleutherAI/pythia-14m``; the vocabulary is padded to 50,304) and packed, one file at a time, into rows of 2,049
tokens that each start with the separator ``<|endoftext|>`` (id 0). A document that does not fit in the rest of a
row continues in the next row after a new separator, and the incomplete last row of a file is dropped.

Skill. spaCy (``en_core_web_sm``) tags and parses each document. A number-marked finite verb is a word tagged VBZ
or VBP, or "was"/"were" tagged VBD. Its class is 1 with a preceding noun subject (NOUN, PROPN or NUM), 2 with any
other preceding subject (mostly pronouns) and 3 without a preceding subject. Every BPE token that overlaps the
word takes its class; all other tokens have class 0. Target thinning of ``agreement`` covers the classes 1-3.

Labels are aligned to predictions: entry p of ``labels.npy`` is the class of the target token at stream position
p + 1, so the array is indexed by the position of the input token, as in ``thinning.masks``.

Evaluation. Competence is read on the first 384 held-out rows, at the targets of class 1 that are one of
is/are, was/were, has/have: the mean over these items of p(true form) / (p(true form) + p(other-number form)).
Background loss is the cross-entropy, in nats, over the held-out targets of class 0.

    python -m thinning.webtext.fineweb download --out RAW_DIR
    python -m thinning.webtext.fineweb prepare --raw RAW_DIR --tokenizer RAW_DIR/tokenizer.json --out DIR

``prepare`` writes the streams DIR/train and DIR/eval, each with its ``labels.npy``, and needs pyarrow, tokenizers
and spaCy; the labels of the paper come from spaCy 3.8.7, en_core_web_sm 3.8.0 and tokenizers 0.23.3.
"""

import argparse
import urllib.request
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from thinning.data import TokenStream, read_rows, row_positions, write_stream

SKILLS = {"agreement": (1, 2, 3)}
SEQ_LEN, ROW, SEP_ID, VOCAB_SIZE = 2048, 2049, 0, 50304
VERB_FORMS = (" is", " are", " was", " were", " has", " have")  # consecutive pairs differ in number only
TRAIN_FILES = ("000_00000.parquet", "001_00000.parquet", "002_00000.parquet", "003_00000.parquet")
EVAL_FILE, EVAL_DOCS, EVAL_ROWS = "013_00000.parquet", 6000, 384
CORPUS_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
              "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/")
TOKENIZER_URL = ("https://huggingface.co/EleutherAI/pythia-14m/resolve/"
                 "cf967c0a9a04383db6f7b1108d86b2962634b4ac/tokenizer.json")
_NLP = _BPE = None


def download(out) -> None:
    """Fetch the tokenizer and the five corpus files (8.6 GB) at fixed revisions; existing files are kept."""
    urls = {"tokenizer.json": TOKENIZER_URL, **{name: CORPUS_URL + name for name in (*TRAIN_FILES, EVAL_FILE)}}
    Path(out).mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        path = Path(out) / name
        if not path.exists():
            urllib.request.urlretrieve(url, f"{path}.part")
            Path(f"{path}.part").replace(path)


def _load_tools(tokenizer_path) -> None:
    """Load the tagger and the tokenizer once per process."""
    global _NLP, _BPE
    import spacy
    from tokenizers import Tokenizer
    _NLP = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
    _NLP.max_length = 2_000_000
    _BPE = Tokenizer.from_file(str(tokenizer_path))


def _subject(word):
    """Subject of a finite verb, or None: its own, or that of the head of an auxiliary or a conjunct."""
    for _ in range(6):
        for child in word.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                return child
        if word.dep_ not in ("aux", "auxpass", "conj") or word.head.i == word.i:
            return None
        word = word.head
    return None


def verb_spans(doc) -> list:
    """(first character, end character, class) of each number-marked finite verb of a parsed document."""
    spans = []
    for word in doc:
        if not (word.tag_ in ("VBZ", "VBP") or (word.tag_ == "VBD" and word.lower_ in ("was", "were"))):
            continue
        subject = _subject(word)
        preceding = subject is not None and subject.i < word.i
        cls = 3 if not preceding else 1 if subject.pos_ in ("NOUN", "PROPN", "NUM") else 2
        spans.append((word.idx, word.idx + len(word.text), cls))
    return spans


def token_classes(starts, ends, spans, n_chars: int) -> np.ndarray:
    """Class of each token with the character offsets [start, end): the largest class among its characters."""
    chars = np.zeros(n_chars + 1, np.int8)
    for first, end, cls in spans:
        chars[first:end] = cls
    marked = np.concatenate(([0], np.cumsum(chars > 0)))
    classes = np.zeros(len(starts), np.int8)
    for i in np.flatnonzero(marked[ends] > marked[starts]):
        classes[i] = chars[starts[i]:ends[i]].max()
    return classes


def label_document(text: str):
    """Token ids of a document (uint16) and the class of each token (int8)."""
    enc = _BPE.encode(text, add_special_tokens=False)
    offsets = np.asarray(enc.offsets, np.int64).reshape(-1, 2)
    spans = verb_spans(_NLP(text)) if len(offsets) else []
    return np.asarray(enc.ids, np.uint16), token_classes(offsets[:, 0], offsets[:, 1], spans, len(text))


def pack_rows(docs, row: int = ROW, sep_id: int = SEP_ID):
    """Concatenate (token ids, classes) documents into whole rows of ``row`` tokens that each start with the
    separator. A separator precedes every document and every continuation of a document in a new row; it has
    class 0. The incomplete last row is dropped."""
    tokens, classes, fill = [], [], 0
    sep = (np.array([sep_id], np.uint16), np.zeros(1, np.int8))

    def put(ids, cls):
        nonlocal fill
        tokens.append(ids)
        classes.append(cls)
        fill = (fill + len(ids)) % row

    for ids, cls in docs:
        if fill:
            put(*sep)
        done = 0
        while done < len(ids):
            if fill == 0:
                put(*sep)
            take = min(row - fill, len(ids) - done)
            put(ids[done:done + take], cls[done:done + take])
            done += take
    n = sum(len(t) for t in tokens) // row * row
    return np.concatenate(tokens + [sep[0]])[:n], np.concatenate(classes + [sep[1]])[:n]


def align_to_predictions(classes: np.ndarray) -> np.ndarray:
    """Labels indexed by the input position: entry p is the class of the token at p + 1; the last entry is 0."""
    return np.append(classes[1:], 0).astype(np.int8)


def _texts(path):
    """Document texts of a parquet file, in file order."""
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    for group in range(table.metadata.num_row_groups):
        yield from table.read_row_group(group, columns=["text"]).column("text").to_pylist()


def prepare(raw, tokenizer, out, workers: int = 8) -> None:
    """Tokenize, label and pack the raw files into the streams ``out/train`` and ``out/eval``."""
    from tokenizers import Tokenizer
    bpe = Tokenizer.from_file(str(tokenizer))
    form_ids = [bpe.encode(form, add_special_tokens=False).ids for form in VERB_FORMS]
    if any(len(ids) != 1 for ids in form_ids):
        raise ValueError("every verb form must be a single token")
    with Pool(workers, initializer=_load_tools, initargs=(tokenizer,)) as pool:
        for split, files, max_docs in (("eval", (EVAL_FILE,), EVAL_DOCS), ("train", TRAIN_FILES, None)):
            labels = []

            def shards():
                for name in files:
                    texts = islice(_texts(Path(raw) / name), max_docs)
                    tokens, classes = pack_rows(pool.imap(label_document, texts, chunksize=64))
                    labels.append(align_to_predictions(classes))
                    yield tokens

            write_stream(Path(out) / split, shards(), VOCAB_SIZE, SEP_ID, corpus="fineweb-edu sample/10BT",
                         files=list(files), row_len=ROW, verb_form_ids=[ids[0] for ids in form_ids])
            np.save(Path(out) / split / "labels.npy", np.concatenate(labels))


class EvalSet:
    """The first ``n_rows`` rows of the held-out stream: inputs ``x``, targets ``y`` and the class of every
    target, with the token ids of the verb forms read from the stream's meta.json."""

    def __init__(self, stream_or_dir, n_rows: int = EVAL_ROWS, seq_len: int = SEQ_LEN):
        stream = stream_or_dir if isinstance(stream_or_dir, TokenStream) else TokenStream(stream_or_dir)
        self.x, self.y = read_rows(stream, 0, n_rows, seq_len)
        labels = np.load(stream.dir / "labels.npy", mmap_mode="r")
        classes = np.asarray(labels[row_positions(0, n_rows, seq_len)]).astype(np.int64)
        self.classes = torch.from_numpy(classes).view(n_rows, seq_len)
        forms = torch.tensor(stream.meta["verb_form_ids"])
        self.partner = torch.arange(stream.vocab_size)
        self.is_form = torch.isin(self.partner, forms)
        self.partner[forms[0::2]], self.partner[forms[1::2]] = forms[1::2], forms[0::2]


@torch.no_grad()
def measure(model, evalset: EvalSet, skill: str, device, per_position: bool = False, batch_size: int = 16):
    """Competence and losses of ``model`` on the held-out rows.

    A test item is a target of class 1 that is one of the six verb forms. ``competence`` is the mean over the
    ``n_items`` items of p(true form) / (p(true form) + p(other-number form)). ``background_loss``, ``skill_loss``
    and ``loss`` are the cross-entropies over the targets outside the skill classes, of the skill classes and
    over all. ``per_position`` adds ``skill_mass`` = p(true form) and ``other_mass`` = p(other-number form) at
    each item, in stream order."""
    skill_classes = torch.tensor(SKILLS[skill], device=device)
    is_form, partner = evalset.is_form.to(device), evalset.partner.to(device)
    nll_sum, n_targets, logp_true, logp_other = np.zeros(2), np.zeros(2), [], []
    for i in range(0, len(evalset.x), batch_size):
        x, y, classes = (t[i:i + batch_size].to(device) for t in (evalset.x, evalset.y, evalset.classes))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logp = F.log_softmax(model(x).float(), -1)
        nll = -logp.gather(-1, y.clamp(min=0)[..., None])[..., 0]
        valid = y != -100
        in_skill = valid & torch.isin(classes, skill_classes)
        for k, sel in enumerate((valid & ~in_skill, in_skill)):
            nll_sum[k] += float(nll[sel].sum())
            n_targets[k] += float(sel.sum())
        item = valid & (classes == 1) & is_form[y.clamp(min=0)]
        target, logp_item = y[item], logp[item]
        logp_true.append(logp_item.gather(-1, target[:, None])[:, 0].cpu())
        logp_other.append(logp_item.gather(-1, partner[target][:, None])[:, 0].cpu())
    logp_true, logp_other = torch.cat(logp_true).double(), torch.cat(logp_other).double()
    out = {"competence": float(torch.sigmoid(logp_true - logp_other).mean()), "n_items": len(logp_true),
           "background_loss": nll_sum[0] / n_targets[0], "skill_loss": nll_sum[1] / n_targets[1],
           "loss": nll_sum.sum() / n_targets.sum()}
    if per_position:
        out["skill_mass"], out["other_mass"] = logp_true.exp().numpy(), logp_other.exp().numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download FineWeb-Edu and prepare its token streams and labels")
    sub = ap.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch the raw corpus files and the tokenizer")
    dl.add_argument("--out", required=True, help="directory for the raw files")
    pr = sub.add_parser("prepare", help="tokenize, label and pack the raw files")
    pr.add_argument("--raw", required=True, help="directory with the raw parquet files")
    pr.add_argument("--tokenizer", required=True, help="tokenizer.json of the Pythia BPE tokenizer")
    pr.add_argument("--out", required=True, help="output directory; receives train/ and eval/")
    pr.add_argument("--workers", type=int, default=8, help="labelling processes")
    a = ap.parse_args(argv)
    if a.command == "download":
        download(a.out)
    else:
        prepare(a.raw, a.tokenizer, a.out, a.workers)


if __name__ == "__main__":
    main()
