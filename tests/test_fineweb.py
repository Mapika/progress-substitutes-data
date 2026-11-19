"""Web-text module: row packing, label alignment, token classes and the competence measure."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinning.data import TokenStream, read_rows, row_positions, write_stream
from thinning.masks import skill_positions
from thinning.webtext import fineweb as fw


def test_rows_start_with_separator_and_labels_follow_targets(tmp_path):
    rng = np.random.default_rng(0)
    lengths = [5, 0, 13, 1, 7, 22, 3, 9]
    docs = [(rng.integers(1, 50, n).astype(np.uint16), rng.integers(0, 4, n).astype(np.int8)) for n in lengths]
    tokens, classes = fw.pack_rows(docs, row=8, sep_id=0)
    assert len(tokens) % 8 == 0 and (tokens[::8] == 0).all() and (classes[tokens == 0] == 0).all()
    kept = tokens[tokens != 0]
    assert np.array_equal(kept, np.concatenate([d for d, _ in docs])[:len(kept)])
    assert np.array_equal(classes[tokens != 0], np.concatenate([c for _, c in docs])[:len(kept)])
    labels = fw.align_to_predictions(classes)
    assert labels.dtype == np.int8 and np.array_equal(labels[:-1], classes[1:]) and labels[-1] == 0
    write_stream(tmp_path, [tokens], vocab_size=50, sep_id=0)
    n_rows = len(tokens) // 8
    x, y = read_rows(TokenStream(tmp_path), 0, n_rows, 7)
    pos = row_positions(0, n_rows, 7)
    assert np.array_equal(tokens[pos + 1], y.numpy().reshape(-1)) and (y != -100).all()
    assert np.array_equal(labels[pos], classes[pos + 1])
    assert np.array_equal(skill_positions(labels, pos, fw.SKILLS["agreement"]), classes[pos + 1] > 0)


def test_token_classes_take_the_largest_class_of_their_characters():
    text = "They are here and it is."
    spans = [(5, 8, 2), (21, 23, 1)]
    starts, ends = np.array([0, 4, 8, 13, 4, 20, 23]), np.array([4, 8, 13, 13, 23, 23, 24])
    assert fw.token_classes(starts, ends, spans, len(text)).tolist() == [0, 2, 0, 0, 2, 1, 0]
    assert fw.token_classes(starts[:0], ends[:0], [], 0).tolist() == []


class Bigram(torch.nn.Module):
    """Next-token probabilities that depend on the input token only."""

    def __init__(self, probs):
        super().__init__()
        self.logits = torch.log(probs)

    def forward(self, x):
        return self.logits[x]


def test_measure_known_answer(tmp_path):
    tokens = np.array([0, 7, 1, 7, 2, 0, 3, 7, 7, 7])
    classes = np.array([0, 0, 1, 0, 2, 0, 1, 0, 0, 0], np.int8)
    write_stream(tmp_path, [tokens], vocab_size=8, sep_id=0, verb_form_ids=[1, 2, 3, 4, 5, 6])
    np.save(tmp_path / "labels.npy", fw.align_to_predictions(classes))
    probs = torch.full((8, 8), 0.05, dtype=torch.float64)  # every row sums to 1
    probs[:, 7] = 0.65
    probs[0, [3, 4, 7]] = torch.tensor([0.3, 0.1, 0.35], dtype=torch.float64)
    probs[7, [1, 2, 7]] = torch.tensor([0.4, 0.1, 0.25], dtype=torch.float64)
    evalset = fw.EvalSet(tmp_path, n_rows=2, seq_len=4)
    out = fw.measure(Bigram(probs), evalset, "agreement", "cpu", per_position=True)
    assert out["n_items"] == 2 and out["skill_mass"].dtype == np.float64
    assert np.allclose(out["skill_mass"], [0.4, 0.3]) and np.allclose(out["other_mass"], [0.1, 0.1])
    assert out["competence"] == pytest.approx((0.8 + 0.75) / 2, abs=1e-6)
    assert out["background_loss"] == pytest.approx(-np.log([0.35, 0.65, 0.65, 0.25, 0.25]).mean(), abs=1e-6)
    assert out["skill_loss"] == pytest.approx(-np.log([0.4, 0.1, 0.3]).mean(), abs=1e-6)
    assert out["loss"] == pytest.approx(-np.log([0.35, 0.65, 0.65, 0.25, 0.25, 0.4, 0.1, 0.3]).mean(), abs=1e-6)
    assert set(fw.measure(Bigram(probs), evalset, "agreement", "cpu")) == set(out) - {"skill_mass", "other_mass"}


def test_verb_classes_from_the_parser(monkeypatch):
    spacy = pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
    text = "The keys to the cabinet are on the table. She was late. There is a problem. They walked home."
    assert [(text[a:b], cls) for a, b, cls in fw.verb_spans(nlp(text))] == [("are", 1), ("was", 2), ("is", 3)]

    def encode(text, add_special_tokens=False):
        words = list(re.finditer(r" ?[^ ]+", text))
        return SimpleNamespace(ids=list(range(1, len(words) + 1)), offsets=[m.span() for m in words])

    monkeypatch.setattr(fw, "_NLP", nlp)
    monkeypatch.setattr(fw, "_BPE", SimpleNamespace(encode=encode))
    ids, classes = fw.label_document(text)
    assert ids.dtype == np.uint16 and classes.dtype == np.int8 and len(ids) == len(text.split())
    assert [(text.split()[i], classes[i]) for i in np.flatnonzero(classes)] == [("are", 1), ("was", 2), ("is", 3)]
    assert all(len(part) == 0 for part in fw.label_document(""))
