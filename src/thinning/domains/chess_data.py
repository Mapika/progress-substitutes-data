"""Build the chess training and held-out token streams from Lichess monthly game files.

    python -m thinning.domains.chess_data 2024-01 2024-02 --out data/chess --workers 32

Inputs are ``YYYY-MM`` months of the Lichess standard rated database (streamed from database.lichess.org),
URLs, or local ``.pgn`` / ``.pgn.zst`` files. A game is kept if it is a rated standard game from the initial
position, ended normally, has at least ``--min-plies`` moves and both players are rated at least ``--min-elo``.

The result depends only on the set of games, not on the order of files or on the number of workers:
  split   a game is held out iff the first 8 bytes of blake2b(game id, key=salt) fall below eval_frac * 2^64;
  order   games are sorted into buckets by the last 8 bytes of that hash, and within a bucket by the hash and by
          a hash of the move sequence, which shuffles the months uniformly (the stream is read once, in order);
  held-out games whose exact move sequence occurs in the training split are removed.
"""

import argparse
import hashlib
import re
import shutil
import threading
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import zstandard

from thinning.data import write_stream
from thinning.domains.chess import SEP, VOCAB_SIZE, move_id

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
INDEX = np.dtype([("key", "<u8"), ("seq", "<u8"), ("len", "<u4")])
EVENT = b'\n[Event "'
_HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$', re.M)
_JUNK = re.compile(r"\{[^}]*\}|;[^\n]*|\$\d+|\d+\.(?:\.\.)?|1-0|0-1|1/2-1/2|\*")
_VARIATION = re.compile(r"\([^()]*\)")


def game_hash(game_id: str, salt: str) -> tuple[int, int]:
    """Two 64-bit values from a game id: the first decides the split, the second the order."""
    h = hashlib.blake2b(game_id.encode(), digest_size=16, key=salt.encode()).digest()
    return int.from_bytes(h[:8], "little"), int.from_bytes(h[8:], "little")


def sequence_hash(ids: np.ndarray) -> int:
    return int.from_bytes(hashlib.blake2b(np.asarray(ids, dtype="<u2").tobytes(), digest_size=8).digest(), "little")


def movetext_to_sans(movetext: str) -> list[str]:
    """Moves in standard algebraic notation, without comments, annotations, move numbers, results, variations."""
    s = movetext
    while "(" in s:
        stripped = _VARIATION.sub(" ", _JUNK.sub(" ", s))
        if stripped == s:
            break
        s = stripped
    return [t.rstrip("?!") for t in _JUNK.sub(" ", s).split()]


def sans_to_ids(sans) -> np.ndarray:
    board = chess.Board()
    ids = np.empty(len(sans), dtype=np.uint16)
    for k, san in enumerate(sans):
        move = board.parse_san(san)
        ids[k] = move_id(move)
        board.push(move)
    return ids


def drop_reason(headers: dict, cfg: dict) -> str | None:
    """Why a game is excluded by its headers, or None if it is kept."""
    if headers.get("Variant", "Standard") not in ("Standard", "standard"):
        return "variant"
    if "FEN" in headers or headers.get("SetUp", "0") == "1":
        return "setup"
    if not headers.get("Event", "").startswith("Rated"):
        return "unrated"
    if headers.get("Termination") not in cfg["terminations"]:
        return "termination"
    try:
        white, black = int(headers["WhiteElo"]), int(headers["BlackElo"])
    except (KeyError, ValueError):
        return "elo_missing"
    return "elo" if min(white, black) < cfg["min_elo"] else None


def process_chunk(args):
    """Parse a block of PGN text into {(split, bucket): (tokens, index records)} and counts."""
    chunk, cfg = args
    stats, groups = Counter(), defaultdict(lambda: ([], []))
    eval_threshold = int(cfg["eval_frac"] * 2**64)
    for game in re.split(r'\n(?=\[Event ")', chunk.decode("utf-8", "replace")):
        game = game.strip()
        if not game.startswith("[Event"):
            continue
        stats["scanned"] += 1
        head, _, movetext = game.partition("\n\n")
        headers = dict(_HEADER.findall(head))
        reason = drop_reason(headers, cfg)
        sans = movetext_to_sans(movetext) if reason is None else []
        if reason is None and len(sans) < cfg["min_plies"]:
            reason = "short"
        if reason is None:
            try:
                ids = sans_to_ids(sans)
            except (ValueError, KeyError):
                reason = "unreadable_moves"
        if reason is not None:
            stats["dropped_" + reason] += 1
            continue
        site = headers.get("Site") or ""
        game_id = site.rstrip("/").rsplit("/", 1)[-1] or \
            "nosite:" + hashlib.blake2b(game.encode(), digest_size=16).hexdigest()
        split_value, order_value = game_hash(game_id, cfg["salt"])
        split = "eval" if split_value < eval_threshold else "train"
        stats["kept_" + split] += 1
        tokens, index = groups[(split, order_value % cfg["n_buckets"])]
        tokens.append(np.append(ids, np.uint16(SEP)))
        index.append((order_value, sequence_hash(ids), len(ids) + 1))
    return stats, {k: (np.concatenate(t), np.array(i, dtype=INDEX)) for k, (t, i) in groups.items()}


def read_chunks(source: str, chunk_bytes: int):
    """Blocks of PGN text of about ``chunk_bytes`` bytes, cut at game boundaries."""
    if re.fullmatch(r"\d{4}-\d{2}", source):
        source = LICHESS_URL.format(source)
    raw = urllib.request.urlopen(source, timeout=120) if source.startswith(("http://", "https://")) \
        else open(source, "rb")
    stream = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20) if source.endswith(".zst") else raw
    buf = b""
    while True:
        block = stream.read(chunk_bytes)
        buf += block
        cut = len(buf) if not block else buf.rfind(EVENT) + 1
        if cut > 0:
            chunk, buf = buf[:cut], buf[cut:]
            if chunk.strip():
                yield chunk
        if not block:
            break
    stream.close()


def assemble_bucket(stem: str, keep) -> np.ndarray:
    """The games of one bucket in their final order, as one token array."""
    tokens = np.fromfile(f"{stem}.tok", dtype=np.uint16)
    index = np.fromfile(f"{stem}.idx", dtype=INDEX)
    lens_all = index["len"].astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(lens_all)[:-1]))
    order = np.lexsort((index["seq"], index["key"]))
    if keep is not None:
        order = order[keep[order]]
    lens, st = lens_all[order], starts[order]
    out_offset = np.concatenate(([0], np.cumsum(lens)[:-1]))
    return tokens[np.repeat(st - out_offset, lens) + np.arange(int(lens.sum()))]


def shards(parts, shard_tokens: int):
    acc, n = [], 0
    for tokens in parts:
        acc.append(tokens)
        n += tokens.size
        if n >= shard_tokens:
            yield np.concatenate(acc)
            acc, n = [], 0
    if n:
        yield np.concatenate(acc)


def build(sources, out, *, min_elo=1500, min_plies=20, terminations=("Normal",), eval_frac=0.005,
          salt="ie-chess-v1", workers=4, n_buckets=256, shard_tokens=1 << 28, chunk_bytes=4 << 20) -> dict:
    out = Path(out)
    tmp = out / "_tmp"
    for split in ("train", "eval"):
        (tmp / split).mkdir(parents=True, exist_ok=True)
    cfg = dict(min_elo=min_elo, min_plies=min_plies, terminations=list(terminations), eval_frac=eval_frac,
               salt=salt, n_buckets=n_buckets)
    in_flight = threading.Semaphore(4 * workers)

    def tasks():
        for source in sources:
            for chunk in read_chunks(source, chunk_bytes):
                in_flight.acquire()
                yield chunk, cfg

    stats = Counter()
    with get_context("spawn").Pool(workers) as pool:
        for chunk_stats, groups in pool.imap_unordered(process_chunk, tasks()):
            in_flight.release()
            stats.update(chunk_stats)
            for (split, bucket), (tokens, index) in groups.items():
                stem = tmp / split / f"b{bucket:05d}"
                with open(f"{stem}.tok", "ab") as f:
                    f.write(tokens.tobytes())
                with open(f"{stem}.idx", "ab") as f:
                    f.write(index.tobytes())
    stems = {s: sorted(p.with_suffix("") for p in (tmp / s).glob("*.idx")) for s in ("train", "eval")}
    train_sequences = np.unique(np.concatenate(
        [np.fromfile(f"{st}.idx", dtype=INDEX)["seq"] for st in stems["train"]] or [np.zeros(0, np.uint64)]))
    for split in ("train", "eval"):
        parts = []
        for stem in stems[split]:
            keep = None
            if split == "eval" and train_sequences.size:
                seq = np.fromfile(f"{stem}.idx", dtype=INDEX)["seq"]
                at = np.minimum(np.searchsorted(train_sequences, seq), train_sequences.size - 1)
                keep = train_sequences[at] != seq
                stats["eval_dropped_seen_in_train"] += int((~keep).sum())
            parts.append((str(stem), keep))
        write_stream(out / split, shards((assemble_bucket(*p) for p in parts), shard_tokens), VOCAB_SIZE, SEP,
                     domain="chess", split=split, sources=list(sources), filters=cfg, counts=dict(stats))
    shutil.rmtree(tmp)
    return dict(stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build chess token streams from Lichess game files")
    ap.add_argument("sources", nargs="+", help="YYYY-MM months, URLs or local .pgn / .pgn.zst files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-elo", type=int, default=1500, help="both players are rated at least this")
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.005)
    ap.add_argument("--salt", default="ie-chess-v1", help="key of the split and order hashes")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    print(build(a.sources, a.out, min_elo=a.min_elo, min_plies=a.min_plies, eval_frac=a.eval_frac, salt=a.salt,
                workers=a.workers))


if __name__ == "__main__":
    main()
