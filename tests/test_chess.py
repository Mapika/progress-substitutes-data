"""Chess: vocabulary, corpus builder, in-check labels, output classes and the held-out measure."""

import math

import chess
import numpy as np
import torch

from conftest import random_games
from thinning.data import TokenStream
from thinning.domains import chess as C
from thinning.domains import chess_data


def test_vocabulary():
    assert C.N_MOVES == 1968 and C.VOCAB_SIZE == 1970
    assert C.MOVES[C.move_id(chess.Move.from_uci("e1g1"))].uci() == "e1g1"
    assert C.MOVES[C.move_id(chess.Move.from_uci("a7b8n"))].uci() == "a7b8n"


def test_labels_mark_the_board_after_each_move():
    board, ids = chess.Board(), []
    for san in ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"):
        ids.append(C.move_id(board.push_san(san)))
    labels = C.label_stream(np.array(ids + [C.SEP] + ids[:3]))
    assert labels.tolist() == [0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1]


def test_builder_is_independent_of_file_order(chess_corpus, tmp_path):
    games = random_games(400)
    (tmp_path / "reversed.pgn").write_text("\n".join(reversed(games)))
    chess_data.build([str(tmp_path / "reversed.pgn")], tmp_path / "out", eval_frac=0.15, workers=1)
    for split in ("train", "eval"):
        a, b = TokenStream(chess_corpus / split), TokenStream(tmp_path / "out" / split)
        assert a.total_tokens == b.total_tokens > 0
        assert (a.window(0, a.total_tokens) == b.window(0, b.total_tokens)).all()
    train = TokenStream(chess_corpus / "train")
    n_games = int((train.window(0, train.total_tokens) == C.SEP).sum())
    assert 300 <= n_games <= 370


def test_header_filters():
    cfg = dict(min_elo=1500, terminations=["Normal"])
    ok = {"Event": "Rated Blitz game", "WhiteElo": "1600", "BlackElo": "1700", "Termination": "Normal"}
    assert chess_data.drop_reason(ok, cfg) is None
    assert chess_data.drop_reason(ok | {"BlackElo": "1400"}, cfg) == "elo"
    assert chess_data.drop_reason(ok | {"Event": "Casual game"}, cfg) == "unrated"
    assert chess_data.drop_reason(ok | {"Termination": "Time forfeit"}, cfg) == "termination"
    sans = chess_data.movetext_to_sans("1. e4 { [%eval 0.2] } e5?! 2. Nf3 (2. f4) Nc6 1-0")
    assert sans == ["e4", "e5", "Nf3", "Nc6"]


def test_output_classes_of_a_known_position():
    board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    cls = C.classify_check(board)
    assert cls[C.move_id(chess.Move.from_uci("e1e2"))] == 1
    assert {int(cls[C.move_id(chess.Move.from_uci(m))]) for m in ("e1d1", "e1f1", "e1d2", "e1f2")} == {6}
    assert cls[C.SEP] == 14 and cls[C.PAD] == 15


def test_legal_class_equals_the_legal_moves(chess_corpus):
    ev = C.EvalSet(TokenStream(chess_corpus / "eval"), n_games=30, offset=0)
    assert ev.in_check.sum() > 0 and ev.classes.shape == (ev.in_check.sum(), C.VOCAB_SIZE)
    k = 0
    for game in ev.games:
        board = chess.Board()
        for t in game:
            if board.is_check():
                legal = {C.move_id(m) for m in board.legal_moves}
                unsafe = {C.move_id(m) for m in board.pseudo_legal_moves if not board.is_legal(m)}
                assert set(np.flatnonzero(np.isin(ev.classes[k], C.LEGAL))) == legal
                assert set(np.flatnonzero(np.isin(ev.classes[k], C.KING_UNSAFE))) == unsafe
                k += 1
            board.push(C.MOVES[int(t)])
    assert k == ev.in_check.sum()


def test_measure_of_a_uniform_model(chess_corpus):
    ev = C.EvalSet(TokenStream(chess_corpus / "eval"), n_games=30, offset=0)
    uniform = lambda x: torch.zeros(*x.shape, C.VOCAB_SIZE)  # noqa: E731
    out = C.measure(uniform, ev, per_position=True)
    n_legal = np.isin(ev.classes, C.LEGAL).sum()
    n_unsafe = np.isin(ev.classes, C.KING_UNSAFE).sum()
    assert math.isclose(out["competence"], n_legal / (n_legal + n_unsafe), rel_tol=1e-5)
    assert math.isclose(out["background_loss"], math.log(C.VOCAB_SIZE), rel_tol=1e-5)
    assert out["skill_mass"].shape == (ev.in_check.sum(),)
    classes = C.output_classes(uniform, ev)
    assert math.isclose(sum(classes[name] for name in C.OUTPUT_CLASSES), 1.0, rel_tol=1e-4)
