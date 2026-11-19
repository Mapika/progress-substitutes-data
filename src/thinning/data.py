"""Token streams: the on-disk format of every corpus, and the fixed-order reader used for training.

A stream directory holds

    meta.json         {"vocab_size", "dtype", "shard_token_counts", "sep_id", ...provenance}
    shard_00000.bin   raw token ids; documents are concatenated as ``document + [sep_id]``

A training corpus is read once, in order. Row k of a run with sequences of ``seq_len`` tokens occupies the tokens
[k * (seq_len + 1), (k + 1) * (seq_len + 1)) of the stream, so the batch schedule fixes the data order and a
position of the stream identifies one prediction in every run that reads it.
"""

import json
from pathlib import Path

import numpy as np
import torch

from thinning.model import segment_ids


def write_stream(dir, arrays, vocab_size: int, sep_id: int, **provenance) -> None:
    """Write token arrays as shard_00000.bin, shard_00001.bin, ... and meta.json."""
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if vocab_size <= 256 else np.uint16
    counts = []
    for i, arr in enumerate(arrays):
        a = np.ascontiguousarray(arr)
        if a.size and int(a.max()) >= vocab_size:
            raise ValueError(f"token id {int(a.max())} >= vocab_size {vocab_size}")
        a.astype(dtype).tofile(d / f"shard_{i:05d}.bin")
        counts.append(int(a.size))
    meta = {"vocab_size": int(vocab_size), "dtype": np.dtype(dtype).name,
            "shard_token_counts": counts, "sep_id": int(sep_id), **provenance}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


class TokenStream:
    """Read-only view of a stream directory as one array of tokens."""

    def __init__(self, dir):
        self.dir = Path(dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.vocab_size = int(self.meta["vocab_size"])
        self.sep_id = int(self.meta["sep_id"])
        self.dtype = np.dtype(self.meta["dtype"])
        self._counts = [int(c) for c in self.meta["shard_token_counts"]]
        self._starts = np.concatenate(([0], np.cumsum(self._counts)))
        self._mmaps = [None] * len(self._counts)

    @property
    def total_tokens(self) -> int:
        return int(self._starts[-1])

    def _shard(self, i):
        if self._mmaps[i] is None:
            self._mmaps[i] = np.memmap(self.dir / f"shard_{i:05d}.bin", dtype=self.dtype, mode="r")
        return self._mmaps[i]

    def window(self, start: int, length: int) -> np.ndarray:
        """Tokens [start, start + length) of the stream; the slice may span shards."""
        if start < 0 or length < 0 or start + length > self.total_tokens:
            raise IndexError(f"window [{start}, {start + length}) out of range "
                             f"for a stream of {self.total_tokens} tokens")
        out = np.empty(length, dtype=self.dtype)
        i = int(np.searchsorted(self._starts, start, side="right")) - 1
        pos, filled = start, 0
        while filled < length:
            off = pos - int(self._starts[i])
            n = min(length - filled, self._counts[i] - off)
            out[filled:filled + n] = self._shard(i)[off:off + n]
            filled += n
            pos += n
            i += 1
        return out

    def documents(self, offset: int, n_docs: int, min_len: int = 1, max_len: int = 1 << 30,
                  chunk: int = 4_000_000) -> list[np.ndarray]:
        """``n_docs`` whole documents (without separators) with a length in [min_len, max_len], starting at the
        first document boundary at or after token ``offset``."""
        docs, pos = [], offset
        while len(docs) < n_docs:
            buf = self.window(pos, min(chunk, self.total_tokens - pos)).astype(np.int64)
            seps = np.flatnonzero(buf == self.sep_id)
            if len(seps) < 2:
                raise ValueError("stream exhausted")
            for a, b in zip(seps[:-1], seps[1:]):
                if min_len <= b - a - 1 <= max_len:
                    docs.append(buf[a + 1:b])
                    if len(docs) == n_docs:
                        break
            pos += int(seps[-1])
        return docs


def read_rows(stream: TokenStream, first_row: int, n: int, seq_len: int):
    """Rows [first_row, first_row + n) as inputs x and next-token targets y, both (n, seq_len) int64.

    Rows cut documents. The leading fragment of a row is the end of a document that began in the previous row;
    with positions restarting at 0 it would be read as a whole document, so its targets are set to -100 (no
    loss) unless the row begins with a separator. The trailing fragment is a true prefix and is kept."""
    w = seq_len + 1
    rows = torch.from_numpy(stream.window(first_row * w, n * w).astype("int64")).view(n, w)
    x, y = rows[:, :-1], rows[:, 1:].clone()
    at_doc_start = x[:, 0] == stream.sep_id
    lead = (segment_ids(x, stream.sep_id) == 0) & ~at_doc_start[:, None]
    y[lead] = -100
    return x, y


def row_positions(first_row: int, n: int, seq_len: int) -> np.ndarray:
    """Stream position of every input token of rows [first_row, first_row + n), flattened. The target at
    position p is the token at p + 1, and label arrays are indexed by p."""
    return ((first_row + np.arange(n))[:, None] * (seq_len + 1) + np.arange(seq_len)[None, :]).reshape(-1)
