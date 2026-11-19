"""Train a model on a token stream with target thinning.

    python -m thinning.train --data data/chess/train --labels data/chess/labels.npy --skill-classes 1 \
        --keep 0.1 --size 9.7M --seed 0 --tokens-per-param 64 --snapshots 1,2,4,8,16,32,64 --out runs/chess-f0.1

The corpus is read once, in order, from row 0, so two runs with the same batch schedule see the same data; the
seed sets the weight initialization only. With ``--keep f`` the targets of the skill's positions (label classes
``--skill-classes``) stay in the loss iff hash(position) < f; every input token stays in the sequence. The loss of
an update is the summed cross-entropy of the retained targets divided by their number.

Recipe: AdamW with the CompleteP parameter groups, weight decay 0.1, gradient clipping at 1, a linear warm-up of
the learning rate and then a constant rate. The warm-up--stable recipe warms up over 2 tokens per parameter and
ramps the batch to the size's full batch over 16 tokens per parameter; the constant-rate chess recipe uses
``--warmup-tokens 15e6 --ramp-tokens-per-param 0 --batch-seqs 64``.

The run directory receives ``config.json``, ``metrics.jsonl`` (training loss every 20 steps) and one
``snap_<tokens>.pt`` per requested snapshot, holding the weights, the token count and the next unread row.
"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from thinning.data import TokenStream, read_rows, row_positions
from thinning.masks import apply_permuted_rule, dropped_targets, position_hash, skill_positions
from thinning.model import Decoder, build_optimizer, init_weights
from thinning.sizes import get_size


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def warmup_lr(tokens: float, warmup_tokens: float) -> float:
    """Learning-rate multiplier: linear from 0 to 1 over the warm-up, then 1."""
    return tokens / warmup_tokens if tokens < warmup_tokens else 1.0


def batch_ramp(tokens: float, ramp_tokens: float, full_batch_seqs: int) -> int:
    """Batch size in sequences: a quarter, a half, three quarters and then all of the full batch."""
    if tokens >= ramp_tokens:
        return full_batch_seqs
    return max(1, full_batch_seqs * (1 + int(4 * tokens / ramp_tokens)) // 4)


def set_lr(opt, mult: float) -> None:
    for group in opt.param_groups:
        group["lr"] = group["peak_lr"] * mult


class Session:
    """A model with its training stream and skill labels: reads thinned rows and takes optimizer steps."""

    def __init__(self, model: Decoder, stream: TokenStream, seq_len: int, labels=None, classes=(),
                 max_rows: int = 0, device: str = "cpu"):
        self.model, self.stream, self.seq_len = model, stream, seq_len
        self.labels, self.classes, self.max_rows, self.device = labels, tuple(classes), max_rows, device
        self.autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.startswith("cuda") \
            else nullcontext

    def rows(self, first: int, n: int, keep: float = 1.0, salt: int = 0, perm=None, drop_random: float = 0.0):
        """Inputs, targets and the skill's positions for rows [first, first + n). The skill's targets are
        thinned to the retained fraction ``keep``; with ``perm`` the retained ones follow the permuted rule;
        ``drop_random`` instead removes that fraction of all labelled targets, whatever their class."""
        x, y = read_rows(self.stream, first, n, self.seq_len)
        skill = torch.zeros(n, self.seq_len, dtype=torch.bool)
        pos = row_positions(first, n, self.seq_len)
        if drop_random > 0:
            drop = position_hash(pos, salt) < drop_random
            if self.labels is not None:
                drop &= np.asarray(self.labels[pos]) >= 0
            y = y.masked_fill(torch.from_numpy(drop).view(n, -1), -100)
        elif self.labels is not None:
            in_skill = skill_positions(self.labels, pos, self.classes)
            drop = dropped_targets(in_skill, pos, keep, salt)
            y = y.masked_fill(torch.from_numpy(drop).view(n, -1), -100)
            if perm is not None:
                y = apply_permuted_rule(y.reshape(-1), torch.from_numpy(in_skill & ~drop), perm).view(n, -1)
            skill = torch.from_numpy(in_skill).view(n, -1)
        return x.to(self.device), y.to(self.device), skill.to(self.device)

    def chunks(self, n: int):
        step = self.max_rows or n
        return [(r0, min(n, r0 + step)) for r0 in range(0, n, step)]

    def backward(self, x, y, weights=None, n_targets=None) -> float:
        """Accumulate the gradient of the summed cross-entropy of the targets in y (optionally weighted per
        target) divided by ``n_targets`` (default: the number of targets in y). Returns the loss."""
        valid = y != -100
        n_targets = valid.sum() if n_targets is None else n_targets
        total = 0.0
        for r0, r1 in self.chunks(x.shape[0]):
            with self.autocast():
                logits = self.model(x[r0:r1]).float()
            v = valid[r0:r1]
            nll = F.cross_entropy(logits[v], y[r0:r1][v], reduction="none")
            loss = (nll if weights is None else nll * weights[r0:r1][v]).sum() / n_targets
            loss.backward()
            total += float(loss.detach())
        return total

    def step(self, opt, x, y, lr_mult: float, **loss_args) -> tuple[float, float]:
        """One update at ``peak learning rate * lr_mult`` with gradient clipping at 1; returns loss and norm."""
        set_lr(opt, lr_mult)
        opt.zero_grad(set_to_none=True)
        loss = self.backward(x, y, **loss_args)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        return loss, float(grad_norm)


def save_snapshot(path, model, tokens: int, next_row: int, half: bool = False) -> None:
    state = {k: (v.to(torch.bfloat16) if half else v) for k, v in model.state_dict().items()}
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": state, "tokens": tokens, "next_row": next_row}, tmp)
    tmp.replace(path)


def load_snapshot(path, device: str = "cpu"):
    """(model, run config, training tokens, next unread row) of a snapshot written by this module."""
    path = Path(path)
    config = json.loads((path.parent / "config.json").read_text())
    ck = torch.load(path, map_location="cpu", weights_only=True)
    attn = "varlen" if device.startswith("cuda") else "sdpa"
    model = Decoder(get_size(config["size"]), config["vocab_size"], config["sep_id"], attn_impl=attn)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model.to(device), config, int(ck["tokens"]), int(ck["next_row"])


def open_labels(path, classes: str):
    """(memory-mapped label array or None, tuple of label classes)."""
    return (np.load(path, mmap_mode="r") if path else None), tuple(int(c) for c in str(classes).split(",") if c)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a model with target thinning")
    ap.add_argument("--data", required=True, help="training stream directory")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--size", required=True, help="model size, e.g. 9.7M, or <layers>x<width>")
    ap.add_argument("--seed", type=int, default=0, help="seed of the weight initialization")
    ap.add_argument("--tokens-per-param", type=float, required=True, help="length of the run")
    ap.add_argument("--snapshots", default="", help="comma-separated snapshot times in tokens per parameter")
    ap.add_argument("--labels", default=None, help="label array of the training stream (.npy)")
    ap.add_argument("--skill-classes", default="", help="comma-separated label classes of the skill")
    ap.add_argument("--keep", type=float, default=1.0, help="retained fraction f of the skill's targets")
    ap.add_argument("--salt", type=int, default=0, help="selects an independent position hash when > 0")
    ap.add_argument("--drop-random", type=float, default=0.0,
                    help="control: remove this fraction of all targets, whatever their class")
    ap.add_argument("--target-weight", type=float, default=0.0,
                    help="> 0: keep every target and multiply the loss of the skill's targets by this weight")
    ap.add_argument("--eligible-denominator", action="store_true",
                    help="divide the summed loss by the number of targets before thinning")
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-seqs", type=int, default=0, help="full batch in sequences (0: the size's batch)")
    ap.add_argument("--warmup-tokens-per-param", type=float, default=2.0)
    ap.add_argument("--warmup-tokens", type=float, default=None, help="warm-up length in tokens (overrides)")
    ap.add_argument("--ramp-tokens-per-param", type=float, default=16.0, help="batch ramp length; 0 for none")
    ap.add_argument("--max-rows", type=int, default=0, help="rows per forward pass (gradient accumulation)")
    ap.add_argument("--half-snapshots", action="store_true", help="store snapshot weights in bfloat16")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args(argv)

    size = get_size(a.size)
    n_params = size.n_params
    stream = TokenStream(a.data)
    model = Decoder(size, stream.vocab_size, stream.sep_id,
                    attn_impl="varlen" if a.device.startswith("cuda") else "sdpa")
    init_weights(model, seed=a.seed)
    model = model.to(a.device)
    opt = build_optimizer(model, a.lr, a.weight_decay)
    labels, classes = open_labels(a.labels, a.skill_classes)
    session = Session(model, stream, a.seq_len, labels, classes, a.max_rows, a.device)

    end = int(a.tokens_per_param * n_params)
    warmup = a.warmup_tokens if a.warmup_tokens is not None else a.warmup_tokens_per_param * n_params
    ramp = a.ramp_tokens_per_param * n_params
    full = a.batch_seqs or size.batch_seqs
    pending = sorted(int(round(float(s) * n_params)) for s in a.snapshots.split(",") if s)
    if end > stream.total_tokens:
        raise ValueError(f"the run needs {end} tokens; the stream has {stream.total_tokens}")

    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update(n_params=n_params, vocab_size=stream.vocab_size, sep_id=stream.sep_id, warmup_tokens=warmup,
                  ramp_tokens=ramp, full_batch_seqs=full)
    (a.out / "config.json").write_text(json.dumps(config, indent=1))
    metrics = a.out / "metrics.jsonl"
    metrics.unlink(missing_ok=True)

    tokens, step, row, recent = 0, 0, 0, []
    while tokens < end:
        n = batch_ramp(tokens, ramp, full)
        loss_args = {}
        if a.target_weight > 0:
            x, y, skill = session.rows(row, n)
            loss_args["weights"] = torch.where(skill, a.target_weight, 1.0)
        else:
            x, y, _ = session.rows(row, n, a.keep, a.salt, drop_random=a.drop_random)
            if a.eligible_denominator:
                loss_args["n_targets"] = (session.rows(row, n)[1] != -100).sum()
        loss, grad_norm = session.step(opt, x, y, warmup_lr(tokens, warmup), **loss_args)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {tokens} tokens")
        row += n
        tokens += n * a.seq_len
        step += 1
        recent.append(loss)
        if step % 20 == 0 or tokens >= end:
            with metrics.open("a") as f:
                f.write(json.dumps(dict(tokens=tokens, tokens_per_param=tokens / n_params, step=step,
                                        train_loss=float(np.mean(recent)), grad_norm=grad_norm,
                                        lr_mult=warmup_lr(tokens, warmup), batch_seqs=n)) + "\n")
            recent = []
        while pending and tokens >= pending[0]:
            save_snapshot(a.out / f"snap_{pending.pop(0):012d}.pt", model, tokens, row, a.half_snapshots)
    return config


if __name__ == "__main__":
    main()
