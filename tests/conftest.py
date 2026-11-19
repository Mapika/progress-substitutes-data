"""A small chess corpus of random legal games, built once per test session with the release's own builder."""

import random

import chess
import numpy as np
import pytest

from thinning.domains import chess_data, label_tokens


def random_games(n_games: int, seed: int = 0) -> list[str]:
    """PGN records of random legal games with the headers of a rated Lichess game."""
    rng = random.Random(seed)
    records = []
    for g in range(n_games):
        board, sans = chess.Board(), []
        while not board.is_game_over() and len(sans) < rng.randint(30, 90):
            move = rng.choice(list(board.legal_moves))
            sans.append(board.san(move))
            board.push(move)
        moves = " ".join(f"{i // 2 + 1}. {san}" if i % 2 == 0 else san for i, san in enumerate(sans))
        records.append(f'[Event "Rated Blitz game"]\n[Site "https://lichess.org/g{g:07d}"]\n[WhiteElo "1700"]\n'
                       f'[BlackElo "1650"]\n[Termination "Normal"]\n\n{moves} 1-0\n')
    return records


@pytest.fixture(scope="session")
def chess_corpus(tmp_path_factory):
    """Directory with train/ and eval/ streams, the PGN file and labels.npy of the training stream."""
    root = tmp_path_factory.mktemp("chess")
    (root / "games.pgn").write_text("\n".join(random_games(400)))
    chess_data.build([str(root / "games.pgn")], root, eval_frac=0.15, workers=1)
    np.save(root / "labels.npy", label_tokens("chess", root / "train", 10 ** 9))
    return root
