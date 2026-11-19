"""Othello, Dyck-4 and SMILES: rules, generators, labels, tokenizer and held-out measures on known cases."""

import math

import numpy as np
import pytest
import torch

from thinning.data import TokenStream, write_stream
from thinning.domains import dyck, othello, smiles
from thinning.model import Decoder, init_weights
from thinning.sizes import get_size


def make_stream(path, tokens, module):
    write_stream(path, [np.asarray(tokens)], module.VOCAB_SIZE, module.SEP_ID)
    return TokenStream(path)


def random_model(module):
    model = Decoder(get_size("2x64"), module.VOCAB_SIZE, module.SEP_ID)
    init_weights(model, seed=0)
    return model.eval()


def fixed_model(probs):
    """A model that predicts the same distribution at every position."""
    return lambda x: torch.log(torch.as_tensor(probs, dtype=torch.float32)).expand(*x.shape, len(probs))


def assert_finite(result):
    assert all(np.isfinite(v).all() for v in result.values())
    assert 0 <= result["competence"] <= 1 and result["background_loss"] > 0
    assert np.allclose(result["skill_mass"] + result["other_mass"], 1)


def square_token(name):
    return int(othello.SQ2TOK[(int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])])


def test_othello_first_moves():
    legal = int(othello.legal_moves(othello.INIT_BLACK, othello.INIT_WHITE))
    assert [sq for sq in range(64) if legal >> sq & 1] == [19, 26, 37, 44]  # d3, c4, f5, e6


def test_othello_forced_pass_label():
    game = [square_token(m) for m in ("c4", "c3", "c2", "b2", "f5", "c1", "a1", "a3")]
    assert othello.label_stream(np.array(game)).tolist() == [0, 0, 0, 0, 0, 0, 0, 2]
    assert othello.label_stream(np.array(game + [othello.PASS_ID, othello.SEP_ID]))[-1] == 0


def test_othello_games_are_deterministic_and_legal(tmp_path):
    buf, lengths = othello.generate_games(1, 0, np.arange(200))
    again, _ = othello.generate_games(1, 0, np.arange(200))
    one, _ = othello.generate_games(1, 0, np.array([5]))
    other_split, _ = othello.generate_games(1, 1, np.arange(200))
    assert (buf == again).all() and (buf[5] == one[0]).all() and not (buf == other_split).all()
    ev = othello.EvalSet(make_stream(tmp_path, buf[np.arange(buf.shape[1]) < lengths[:, None]], othello), 200)
    assert len(ev.docs) == 200 and len(ev.y) == lengths.sum()
    move = ev.y < 60
    assert ev.legal[move, ev.y[move]].all() and (ev.cls[move] <= 1).all()
    assert ((ev.cls == 2) == (ev.y == othello.PASS_ID)).all() and ((ev.cls == 3) == (ev.y == othello.SEP_ID)).all()
    assert ((ev.cls == 1) == (ev.legal.sum(1) == 1)).all()

    uniform = othello.measure(fixed_model(np.full(62, 1 / 62)), ev, "forced_pass", "cpu", per_position=True)
    assert uniform["competence"] == pytest.approx(1 / 62) and uniform["background_loss"] == pytest.approx(math.log(62))
    assert uniform["forced_move_legality"] == pytest.approx(1 / 60)
    assert uniform["legal_mass_two"] == pytest.approx(2 / 62) and np.allclose(uniform["skill_mass"], 1 / 62)
    assert len(uniform["skill_mass"]) == (ev.cls == 2).sum()
    assert_finite(othello.measure(random_model(othello), ev, "forced_move", "cpu", per_position=True))


def test_dyck_labels():
    labels = dyck.label_stream(np.array([0, 1, 5, 4, 8]))  # ( [ ] ) separator
    assert labels.tolist() == [0, 1, 2, 0, 0]
    nest = np.array([2] * 20 + [6] * 20 + [8])  # close j has match distance 1 + 2j and depth 20 - j before it
    assert dyck.label_stream(nest)[19:39].tolist() == [5, 6, 6, 6, 7] + [3] * 11 + [4] * 4
    assert dyck.label_stream(nest)[:19].tolist() == [0] * 19 and dyck.label_stream(nest)[39:].tolist() == [0, 0]


def test_dyck_corpus_is_balanced_and_split(tmp_path):
    dyck.main(["build", "--out", str(tmp_path), "--n-train", "3000", "--n-eval", "300", "--seed", "0"])
    train, held_out = TokenStream(tmp_path / "train"), TokenStream(tmp_path / "eval")
    docs = {}
    for name, stream in (("train", train), ("eval", held_out)):
        text = bytes(stream.window(0, stream.total_tokens)).split(bytes([dyck.SEP_ID]))
        assert text[-1] == b""
        docs[name] = text[:-1]
    assert len(docs["train"]) == 3000 and 0 < len(docs["eval"]) < 300
    assert not set(docs["eval"]) & set(docs["train"])
    for doc in docs["train"]:
        open_types, deepest = [], 0
        for t in doc:
            if t < dyck.K:
                open_types.append(t)
            else:
                assert open_types.pop() == t - dyck.K
            deepest = max(deepest, len(open_types))
        assert not open_types and 1 <= deepest <= 32 and 2 <= len(doc) <= 256 and len(doc) % 2 == 0
    N, H = dyck.path_counts()
    buf, lengths = dyck.generate_documents(0, 0, np.arange(3), N, H)
    assert [bytes(buf[i, :lengths[i] - 1]) for i in range(3)] == docs["train"][:3]

    ev = dyck.EvalSet(train, n_rows=8, seq_len=256)
    assert ev.x.shape == (8, 256) and len(ev.y) == len(ev.cls) == int(ev.scored.sum())
    uniform = dyck.measure(fixed_model(np.full(9, 1 / 9)), ev, "closing_2_7", "cpu", per_position=True)
    assert uniform["competence"] == pytest.approx(0.25) and uniform["background_loss"] == pytest.approx(math.log(9))
    assert np.allclose(uniform["skill_mass"], 0.25) and np.allclose(uniform["other_mass"], 0.75)
    assert_finite(dyck.measure(random_model(dyck), ev, "closing_deep", "cpu", per_position=True))


def test_smiles_tokenizer_round_trip():
    text = "C1CC1\nCC(C)C\n[13CH4]\nC/C=C\\C\n"
    ids = smiles.encode(text)
    assert smiles.decode(ids) == text and ids[:6].tolist() == [26, 12, 26, 26, 12, 1]
    assert int(ids.max()) < smiles.VOCAB_SIZE and (ids == smiles.SEP_ID).sum() == 4
    with pytest.raises(ValueError):
        smiles.encode("CXC")


def test_smiles_labels():
    assert smiles.label_stream(smiles.encode("C1CC1\n")).tolist() == [1, 0, 0, 2, 4, 0]
    assert smiles.label_stream(smiles.encode("CC(C)C\n")).tolist() == [0, 6, 0, 3, 0, 4, 0]
    assert smiles.label_stream(smiles.encode("[13CH4]\n")).tolist() == [0, 0, 0, 0, 0, 0, 4, 0]
    assert smiles.label_stream(smiles.encode("C1CC1\nC1CC1"))[5:].tolist() == [0, 1, 0, 0, 2, 0]
    assert smiles.label_stream(smiles.encode("C%12CC%12\n")).tolist() == [5, 5, 1, 0, 0, 5, 5, 2, 4, 0]


def test_smiles_build_from_text(tmp_path):
    (tmp_path / "a.smi").write_text("CCO\nC1CC1\nCXC\n\nCC(C)C\n")
    (tmp_path / "b.smi").write_text("CCO\nc1ccccc1\nCCN\n")
    args = ["--out", str(tmp_path / "out"), "--n-train", "2", "--n-eval", "3"]
    smiles.main(["build", "--smiles", str(tmp_path / "a.smi"), str(tmp_path / "b.smi")] + args)
    train, held_out = TokenStream(tmp_path / "out/train"), TokenStream(tmp_path / "out/eval")
    assert smiles.decode(train.window(0, train.total_tokens)) == "CCO\nC1CC1\n"
    assert smiles.decode(held_out.window(0, held_out.total_tokens)) == "CC(C)C\nc1ccccc1\n"
    assert train.sep_id == smiles.SEP_ID and train.vocab_size == smiles.VOCAB_SIZE


def test_smiles_measure_and_output_classes(tmp_path):
    ev = smiles.EvalSet(make_stream(tmp_path, smiles.encode("C1CC1\nC1CC2CC2C1\nCC(C)C\n"), smiles), 3)
    assert [smiles.decode(d) for d in ev.docs] == ["C1CC1", "C1CC2CC2C1", "CC(C)C"]
    closures = np.flatnonzero(ev.cls == 2)
    assert smiles.decode(ev.y[closures]) == "121"
    assert [np.flatnonzero(row).tolist() for row in ev.open_digits[closures]] == [[1], [1, 2], [1]]

    probs = np.zeros(smiles.VOCAB_SIZE)
    probs[[smiles.STOI[c] for c in "123C"] + [smiles.SEP_ID]] = 0.2
    classes = smiles.output_classes(fixed_model(probs), ev, "cpu")
    expected = {"true_digit": 1 / 5, "other_open_digit": 1 / 15, "digit_not_open": 1 / 3, "atom": 1 / 5,
                "branch_opening": 0, "other": 1 / 5}
    assert classes == pytest.approx(expected, abs=1e-6)
    result = smiles.measure(fixed_model(probs), ev, "ring_closure", "cpu", per_position=True)
    assert result["competence"] == pytest.approx(0.2) and result["ring_opening_choice"] == pytest.approx(1 / 3)
    assert np.allclose(result["skill_mass"], 0.2) and np.allclose(result["other_mass"], 0.8)
    assert_finite(smiles.measure(random_model(smiles), ev, "ring_closure", "cpu", per_position=True))
    assert_finite(smiles.measure(random_model(smiles), ev, "branch_closing", "cpu", per_position=True))
    assert sum(smiles.output_classes(random_model(smiles), ev, "cpu").values()) == pytest.approx(1, abs=1e-5)
