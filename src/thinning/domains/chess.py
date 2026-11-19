"""Chess: move vocabulary, in-check labels, classes of output token at in-check positions, held-out evaluation.

A game is one token per move. The vocabulary holds every from-to square pair that a queen or a knight can travel
on an empty board (1,792 pairs) and the pawn promotions to queen, rook, bishop and knight (176), 1,968 moves in
all, followed by the separator (end of game) and a padding token. Castling is the king's two-square move.

The skill is king safety. Its positions are the boards with the side to move in check, and the label of a stream
position is 1 if the board after that move is in check, 0 if not, and -1 at separators and in an unfinished game.

At an in-check board every output token belongs to one fine class (``CLASSES``). The legal classes together are
the legal moves; the king-unsafe classes are the moves that obey the piece-movement rules and leave the king
attacked. Competence is legal / (legal + king-unsafe) probability mass pooled over in-check positions, and
background loss is the mean cross-entropy at the positions that are not in check.
"""

import chess
import numpy as np
import torch

from thinning.data import TokenStream

SKILLS = {"check": (1,)}
_PROMOTIONS = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def _move_list() -> list[chess.Move]:
    """Moves sorted by from-square, to-square and promotion piece (none, queen, rook, bishop, knight)."""
    moves = []
    for f in chess.SQUARES:
        for t in chess.SQUARES:
            df = abs(chess.square_file(f) - chess.square_file(t))
            dr = abs(chess.square_rank(f) - chess.square_rank(t))
            if f != t and (df == 0 or dr == 0 or df == dr or {df, dr} == {1, 2}):
                moves.append(chess.Move(f, t))
            rf, rt = chess.square_rank(f), chess.square_rank(t)
            if df <= 1 and ((rf == 6 and rt == 7) or (rf == 1 and rt == 0)):
                moves.extend(chess.Move(f, t, promotion=p) for p in _PROMOTIONS[1:])
    return moves


MOVES = _move_list()
N_MOVES = len(MOVES)
SEP, PAD, VOCAB_SIZE = N_MOVES, N_MOVES + 1, N_MOVES + 2
_MOVE_ID = {(m.from_square, m.to_square, m.promotion): i for i, m in enumerate(MOVES)}
FROM = np.array([m.from_square for m in MOVES], np.int64)

CLASSES = ["legal king move", "legal capture of the checking piece by the king",
           "legal capture of the checking piece by another piece", "legal interposition",
           "move of another piece that ignores the check", "castling that only the check forbids",
           "king move to a square attacked by the checking piece", "king move to a square attacked by another piece",
           "move that exposes the king to a further piece", "blocked move", "impossible shape for the piece",
           "empty from-square", "opponent piece on the from-square", "unavailable castling", "separator", "padding"]
LEGAL = (0, 1, 2, 3)
KING_UNSAFE = (4, 6, 7, 8)
OUTPUT_CLASSES = {"legal king move": (0,), "legal capture of the checking piece": (1, 2),
                  "legal interposition": (3,), "would be legal if the king were not in check": (4, 5),
                  "king move to an attacked square": (6, 7), "exposes the king to a further piece": (8,),
                  "not a move of the position": (9, 10, 11, 12, 13), "end of game or padding": (14, 15)}


def move_id(move: chess.Move) -> int:
    """Token id of a move; KeyError if the move is not in the vocabulary."""
    return _MOVE_ID[(move.from_square, move.to_square, move.promotion)]


def _shape_table() -> np.ndarray:
    """shape[token, colour, piece type]: the piece can make the move on an empty board (a promotion piece is
    present exactly when a pawn reaches the last rank)."""
    shape = np.zeros((N_MOVES, 2, 7), bool)
    for i, m in enumerate(MOVES):
        f, t = m.from_square, m.to_square
        df, dr = chess.square_file(t) - chess.square_file(f), chess.square_rank(t) - chess.square_rank(f)
        adf, adr = abs(df), abs(dr)
        for colour in (0, 1):
            fwd, start, last = (1, 1, 7) if colour else (-1, 6, 0)
            pawn = (dr == fwd or (dr == 2 * fwd and chess.square_rank(f) == start)) if df == 0 else \
                (adf == 1 and dr == fwd)
            shape[i, colour, chess.PAWN] = pawn and ((chess.square_rank(t) == last) == (m.promotion is not None))
            if m.promotion is None:
                shape[i, colour, chess.KNIGHT] = {adf, adr} == {1, 2}
                shape[i, colour, chess.KING] = max(adf, adr) == 1
                shape[i, colour, chess.ROOK] = df == 0 or dr == 0
                shape[i, colour, chess.BISHOP] = adf == adr
                shape[i, colour, chess.QUEEN] = df == 0 or dr == 0 or adf == adr
    return shape


_SHAPE = _shape_table()
_CASTLE_SHAPE = np.zeros((N_MOVES, 2), bool)
for _i, _m in enumerate(MOVES):
    _CASTLE_SHAPE[_i, 1] = (_m.from_square, _m.to_square) in ((chess.E1, chess.G1), (chess.E1, chess.C1))
    _CASTLE_SHAPE[_i, 0] = (_m.from_square, _m.to_square) in ((chess.E8, chess.G8), (chess.E8, chess.C8))


def _castling_but_for_check(board: chess.Board) -> list[int]:
    """Castling moves with rights, an empty path and unattacked transit squares."""
    colour = board.turn
    king_square = chess.E1 if colour else chess.E8
    if board.king(colour) != king_square:
        return []
    out, base = [], 0 if colour else 56
    for rook_file, path, transit, to_file in ((7, (5, 6), (5, 6), 6), (0, (1, 2, 3), (3, 2), 2)):
        if not board.castling_rights & chess.BB_SQUARES[base + rook_file]:
            continue
        if any(board.piece_at(base + f) is not None for f in path):
            continue
        if any(board.is_attacked_by(not colour, base + f) for f in transit):
            continue
        out.append(move_id(chess.Move(king_square, base + to_file)))
    return out


def classify_check(board: chess.Board) -> np.ndarray:
    """Fine class (index into ``CLASSES``) of every output token at a board with the side to move in check."""
    colour, king_square, checkers = board.turn, board.king(board.turn), board.checkers()
    piece_type, piece_colour = np.zeros(64, np.int64), np.full(64, -1, np.int64)
    for square, piece in board.piece_map().items():
        piece_type[square], piece_colour[square] = piece.piece_type, int(piece.color)
    from_type, from_colour = piece_type[FROM], piece_colour[FROM]
    castle = (from_type == chess.KING) & (from_colour == int(colour)) & _CASTLE_SHAPE[:, int(colour)]
    shape = _SHAPE[np.arange(N_MOVES), int(colour), from_type]
    cls = np.full(VOCAB_SIZE, 9, np.uint8)
    cls[:N_MOVES] = np.select([from_type == 0, from_colour != int(colour), castle, ~shape], [11, 12, 13, 10], default=9)
    cls[SEP], cls[PAD] = 14, 15
    for i in _castling_but_for_check(board):
        cls[i] = 5
    for move in list(board.generate_pseudo_legal_moves()):
        i, by_king = move_id(move), move.from_square == king_square
        if board.is_legal(move):
            captures = move.to_square in checkers
            if not captures and not by_king and board.is_en_passant(move):
                captures = (move.to_square + (-8 if colour else 8)) in checkers
            cls[i] = (1 if captures else 0) if by_king else (2 if captures else 3)
        else:
            board.push(move)
            new_attackers = int(board.attackers(not colour, board.king(colour))) & ~int(checkers)
            board.pop()
            cls[i] = (7 if new_attackers else 6) if by_king else (8 if new_attackers else 4)
    return cls


def label_stream(tokens: np.ndarray) -> np.ndarray:
    """1 where the board after the move at that position has the side to move in check, 0 where it has not,
    -1 at separators, in a trailing unfinished game and in games that do not replay."""
    out = np.full(len(tokens), -1, np.int8)
    cuts = np.flatnonzero(tokens == SEP)
    for a, b in zip(np.r_[0, cuts[:-1] + 1], cuts):
        board, labels = chess.Board(), []
        try:
            for t in tokens[a:b]:
                board.push(MOVES[int(t)])
                labels.append(board.is_check())
        except (IndexError, AssertionError, ValueError):
            continue
        out[a:b] = labels
    return out


class EvalSet:
    """Held-out games. Position k of a game is the board after k moves, with the k-th move as its target."""

    def __init__(self, stream: TokenStream, n_games: int = 1000, offset: int = 80_000_000,
                 min_len: int = 10, max_len: int = 200):
        self.games = stream.documents(offset, n_games, min_len, max_len)
        self.nper = np.array([len(g) for g in self.games], np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.nper)[:-1]])
        self.y = np.concatenate(self.games).astype(np.int64)
        self.game_of = np.repeat(np.arange(len(self.games)), self.nper)
        in_check, classes = [], []
        for game in self.games:
            board = chess.Board()
            for t in game:
                in_check.append(board.is_check())
                if in_check[-1]:
                    classes.append(classify_check(board))
                board.push(MOVES[int(t)])
        self.in_check = np.array(in_check, bool)
        self.classes = np.array(classes, np.uint8).reshape(-1, VOCAB_SIZE)

    def batches(self, batch_games: int = 32):
        """Inputs [separator, game, separators...] and, for every position of the batch, its index in the set
        and its (row, column) in the input."""
        for b0 in range(0, len(self.games), batch_games):
            sel = range(b0, min(b0 + batch_games, len(self.games)))
            x = torch.full((len(sel), int(self.nper[sel].max()) + 1), SEP, dtype=torch.long)
            pos, rows, cols = [], [], []
            for j, g in enumerate(sel):
                n = int(self.nper[g])
                x[j, 1:n + 1] = torch.as_tensor(self.games[g])
                pos.append(np.arange(self.starts[g], self.starts[g] + n))
                rows.append(np.full(n, j))
                cols.append(np.arange(n))
            yield x, np.concatenate(pos), np.concatenate(rows), np.concatenate(cols)


@torch.no_grad()
def _position_outputs(model, evalset: EvalSet, device):
    """Cross-entropy of the played move at every position, and the output probabilities at in-check positions."""
    nll = np.zeros(len(evalset.y))
    probs = np.zeros((int(evalset.in_check.sum()), VOCAB_SIZE), np.float32)
    check_index = np.cumsum(evalset.in_check) - 1
    for x, pos, rows, cols in evalset.batches():
        logp = torch.log_softmax(model(x.to(device)).float()[rows, cols], -1).cpu()
        nll[pos] = -logp.gather(1, torch.from_numpy(evalset.y[pos])[:, None])[:, 0].double().numpy()
        chk = evalset.in_check[pos]
        probs[check_index[pos[chk]]] = logp[torch.from_numpy(chk)].exp().numpy()
    return nll, probs


def measure(model, evalset: EvalSet, skill: str = "check", device: str = "cpu", per_position: bool = False,
            perm: np.ndarray | None = None) -> dict:
    """King-safety competence and background loss. ``perm`` (a permutation of the move ids) adds the same
    masses read on the permuted rule, whose targets are the images of the moves under ``perm``."""
    nll, probs = _position_outputs(model, evalset, device)
    chk = evalset.in_check
    is_legal, is_unsafe = np.isin(evalset.classes, LEGAL), np.isin(evalset.classes, KING_UNSAFE)
    legal = (probs * is_legal).sum(1).astype(np.float64)
    unsafe = (probs * is_unsafe).sum(1).astype(np.float64)
    out = dict(competence=legal.sum() / (legal.sum() + unsafe.sum()), background_loss=nll[~chk].mean(),
               legal_mass=legal.mean(), king_unsafe_mass=unsafe.mean(), loss=nll.mean(), skill_loss=nll[chk].mean())
    if perm is not None:
        permuted = probs[:, :N_MOVES][:, perm]
        legal_p = (permuted * is_legal[:, :N_MOVES]).sum(1).astype(np.float64)
        unsafe_p = (permuted * is_unsafe[:, :N_MOVES]).sum(1).astype(np.float64)
        out.update(competence_permuted=legal_p.sum() / (legal_p.sum() + unsafe_p.sum()),
                   legal_mass_permuted=legal_p.mean(), king_unsafe_mass_permuted=unsafe_p.mean())
    out = {k: float(v) for k, v in out.items()}
    if per_position:
        out.update(skill_mass=legal, other_mass=unsafe)
    return out


def output_classes(model, evalset: EvalSet, device: str = "cpu") -> dict:
    """Mean probability of each output class at in-check positions, the share of positions (with at least two
    legal replies) whose most probable legal reply is the move played, and that share for a uniform choice."""
    _, probs = _position_outputs(model, evalset, device)
    fine = np.stack([(probs * (evalset.classes == c)).sum(1) for c in range(len(CLASSES))], 1).astype(np.float64)
    out = {name: float(fine[:, list(group)].sum(1).mean()) for name, group in OUTPUT_CLASSES.items()}
    is_legal = np.isin(evalset.classes, LEGAL)
    many = is_legal.sum(1) >= 2
    top = np.where(is_legal, probs, -1.0).argmax(1)
    played = evalset.y[evalset.in_check]
    out["most_probable_legal_reply_is_played"] = float((top == played)[many].mean())
    out["uniform_legal_reply_is_played"] = float((1.0 / is_legal.sum(1)[many]).mean())
    out["n_positions"] = int(len(probs))
    return out
