"""Domains: corpus builders, skill labels and held-out evaluation.

Every domain module defines

    SKILLS                         {skill name: tuple of label classes}
    label_stream(tokens)           int8 class of the prediction made at each token of a chunk of whole documents
    EvalSet(stream, ...)           a fixed set of held-out documents
    measure(model, evalset, skill, device, per_position=False)
                                   {"competence", "background_loss", ...}; with per_position also the arrays
                                   "skill_mass" and "other_mass" over the skill's test positions, whose ratio
                                   is the odds of the skill at each position

and a command line that builds its corpus. Labels for a training stream are written by

    python -m thinning.domains label --domain chess --data data/chess/train --n-tokens 2e9 --out labels.npy

The web-text module labels its stream from the raw text when it prepares the corpus and has no ``label_stream``.
"""

import argparse
import importlib
from multiprocessing import get_context

import numpy as np

from thinning.data import TokenStream

MODULES = {"chess": "thinning.domains.chess", "othello": "thinning.domains.othello",
           "smiles": "thinning.domains.smiles", "dyck": "thinning.domains.dyck",
           "webtext": "thinning.webtext.fineweb"}


def get(name: str):
    return importlib.import_module(MODULES[name])


def _label_span(args) -> np.ndarray:
    name, path, start, length = args
    return get(name).label_stream(TokenStream(path).window(start, length).astype(np.int64))


def label_tokens(name: str, path, n_tokens: int, chunk: int = 8_000_000, workers: int = 1) -> np.ndarray:
    """Labels of the first ``n_tokens`` positions of the stream at ``path``. The stream is labelled in chunks
    that end at a document separator, so the document that contains the last position is labelled whole."""
    stream = TokenStream(path)
    n_tokens = min(int(n_tokens), stream.total_tokens)
    spans, pos = [], 0
    while pos < n_tokens:
        buf = stream.window(pos, min(chunk, stream.total_tokens - pos))
        seps = np.flatnonzero(buf == stream.sep_id)
        spans.append((name, str(path), pos, int(seps[-1]) + 1 if len(seps) else len(buf)))
        pos += spans[-1][3]
    if workers > 1:
        with get_context("spawn").Pool(workers) as pool:
            parts = list(pool.imap(_label_span, spans))
    else:
        parts = [_label_span(span) for span in spans]
    return np.concatenate(parts)[:n_tokens].astype(np.int8)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the skill labels of a training stream")
    ap.add_argument("command", choices=["label"])
    ap.add_argument("--domain", required=True, choices=["chess", "othello", "smiles", "dyck"])
    ap.add_argument("--data", required=True, help="stream directory")
    ap.add_argument("--n-tokens", type=float, required=True, help="number of leading stream positions to label")
    ap.add_argument("--out", required=True, help="output .npy file")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    np.save(a.out, label_tokens(a.domain, a.data, int(a.n_tokens), workers=a.workers))
