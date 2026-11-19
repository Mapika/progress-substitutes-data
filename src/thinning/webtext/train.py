"""Training of the web-text models of 20M to 7B parameters on the Dolma 1.7 pools, with target thinning.

    python -m thinning.webtext.train --size 20M --seed 0 --keep close=0.3,agr=0.1,copy=1,succ=0.3 \
        --cap-tokens 1e9 --data data/dolma --eval data/eval --tokenizer tokenizer.json --out runs/20M-s0
    torchrun --nproc_per_node 4 -m thinning.webtext.train --size 7B --lr 0.000265 --horizon-tokens 1.2e10 \
        --micro 8 --grad-ckpt --keep agr=0,copy=0 --fkeep cap=0 --cap-tokens 3359899648 --save-every-log2 2 ...

Model: the Llama architecture of the OLMo models of DataDecide (pre-norm RMSNorm, rotary positions, SwiGLU, no
biases, untied output head, vocabulary 50,304, no attention mask between the documents of a row), initialized
from a truncated normal with standard deviation 0.02. ``--size`` is 20M, 60M, 150M, 300M, 1B or 7B, or a custom
``<layers>x<width>x<heads>`` such as ``2x64x2`` for runs on a CPU.

Recipe: AdamW (betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.05 on every parameter), gradient clipping at 1,
a linear warm-up over 32 steps and then a constant learning rate (2.12e-3; ``--lr`` 1.06e-3 at 1B and 2.65e-4 at
7B), batches of 128 rows of 2,048 tokens, bf16 autocast on CUDA with fp32 weights. The loss of a step is the
summed cross-entropy over the retained targets of the batch divided by their number. Under torchrun every
process reads its share of each batch. The seed sets the initialization and the data order, which is drawn for
``--horizon-tokens`` (``thinning.webtext.dolma``). ``--keep`` (close, agr, succ, copy) and ``--fkeep`` (cap, ctry)
give the retained fraction f of each labelled class (``thinning.webtext.labels``); other classes keep every target.
``--filter`` removes documents of three domains from the stream (``thinning.webtext.docfilter``); train.jsonl and
eval.jsonl then have ``rows_read``, the rows of the data order consumed so far.

Outputs in ``--out``:
  order.npy    the data order
  train.jsonl  step, tokens, loss (mean over the retained targets of the batch), lr, grad_norm
  eval.jsonl   one row per evaluation: step, tokens, lr and the measures of ``thinning.webtext.evalsets`` (loss,
               background_loss, background_loss_structural and a competence_<skill> for agreement, repetition,
               counting, closing, closing_nested, capitals and city_countries). Evaluations fall at step 0, at
               steps log-spaced by 2^(1/8) from step 8, at every 1/50 of the horizon, in the windows of
               ``--dense-eval`` and at the last step.
  step<N>/     snapshots (bf16 safetensors, meta.json) at the steps 16 * 2^(i/k) of ``--save-every-log2 k`` and
               at the last step
  state.pt     weights, AdamW state and step, every ``--state-every`` steps and at the last step; a run started
               again with the same ``--out`` continues from it
"""

import argparse
import contextlib
import functools
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from thinning.webtext import labels as L
from thinning.webtext.dolma import Pool, instance_ok
from thinning.webtext.evalsets import Evaluator

SEQ, PAD = L.ROW, 1
LR = 0.0047 * (1_176_832_000 / 108e6) ** (-1 / 3)        # 2.12e-3
SIZES = {"20M": (192, 8, 16, 768), "60M": (384, 12, 16, 1536), "150M": (768, 12, 12, 3072),
         "300M": (1024, 16, 16, 4096), "1B": (2048, 16, 16, 8192), "7B": (4096, 32, 32, 11008)}


def get_shape(size: str):
    """(width, heads, layers, MLP width) of a named size or of a custom ``<layers>x<width>x<heads>``."""
    if size in SIZES:
        return SIZES[size]
    layers, width, heads = (int(v) for v in size.split("x"))
    return width, heads, layers, 4 * width


def build_model(size: str, seed: int, dev):
    """A freshly initialized model. Weights are drawn from one seeded generator in parameter order; models of
    more than 3B parameters are built and initialized on the training device."""
    from transformers import LlamaConfig, LlamaForCausalLM
    d, heads, layers, ff = get_shape(size)
    cfg = LlamaConfig(vocab_size=L.VOCAB, hidden_size=d, intermediate_size=ff, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=heads, hidden_act="silu", use_cache=False,
                      max_position_embeddings=SEQ, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
                      attention_bias=False, mlp_bias=False, bos_token_id=None, eos_token_id=L.EOS, pad_token_id=PAD)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    n_params = layers * (4 * d * d + 3 * d * ff + 2 * d) + d + L.VOCAB * d
    init_dev = dev if n_params > 3e9 else "cpu"
    with torch.device(init_dev):
        model = LlamaForCausalLM(cfg)
    g = torch.Generator(device=init_dev).manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.fill_(1.0)
            else:
                t = torch.empty(p.shape, device=p.device)
                torch.nn.init.trunc_normal_(t, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=g)
                p.copy_(t)
    return model.to(dev)


class LossSum(torch.nn.Module):
    """Summed cross-entropy over the targets that are not -100."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y):
        logits = self.m.lm_head(self.m.model(input_ids=x).last_hidden_state).float()
        return F.cross_entropy(logits.reshape(-1, L.VOCAB), y.reshape(-1), ignore_index=-100, reduction="sum")


def save_snapshot(model, path: Path, meta: dict):
    """Weights in bf16 safetensors with the model configuration, and meta.json."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
    (tmp / "meta.json").write_text(json.dumps(meta))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def warmup_then_constant(lr: float, warmup: int):
    """Learning rate of a step: linear over the first ``warmup`` steps, then constant."""
    return lambda step: lr * step / warmup if step < warmup else lr


def eval_steps(grid: int, end: int, dense: str = "") -> set:
    """Evaluation steps up to ``end``: log-spaced by 2^(1/8) from step 8, every 1/50 of ``grid`` steps, ``end``,
    and every k-th step from a to b for each window ``a:b:k`` of ``dense``."""
    s, x = {grid}, 8.0
    while x < grid:
        s.add(int(round(x)))
        x *= 2 ** 0.125
    s.update(int(round(grid * k / 50)) for k in range(1, 51))
    for w in filter(None, dense.split(",")):
        lo, hi, k = (int(float(v)) for v in w.split(":"))
        s.update(range(lo, min(hi, grid) + 1, k))
    return {v for v in s if 0 < v <= end} | {end}


def snapshot_steps(end: int, k: int) -> set:
    """Steps 16 * 2^(i/k) below ``end``, and ``end``; none for k = 0."""
    s, x = {end} if k else set(), 16.0
    while k and x < end:
        s.add(int(round(x)))
        x *= 2 ** (1 / k)
    return s


class Reader:
    """Batches of a data order for one process: inputs, and targets that are -100 where the loss is removed.
    ``rule``, if given, rewrites the targets (``thinning.webtext.returns``)."""

    def __init__(self, pool, order, batch, lo, hi, keep, fkeep, tables, rule=None):
        self.pool, self.order, self.batch, self.lo, self.hi = pool, order, batch, lo, hi
        self.keep, self.fkeep, self.tables, self.rule = keep, fkeep, tables, rule

    def rows(self, i):
        """Rows [lo, hi) of batch i, (n, 2048) int64, and the stream position of every token."""
        sel = self.order[i * self.batch + self.lo:i * self.batch + self.hi]
        x = np.stack([self.pool.row(s, r) for s, r in sel]).astype(np.int64)
        return x, L.stream_positions(sel[:, 0], sel[:, 1])

    def step(self, i):
        """Inputs (n, 2047), targets (n, 2047) and the number of targets in the loss."""
        x, pos = self.rows(i)
        ok = np.array([instance_ok(row) for row in x])
        x16 = x.astype(np.uint16)
        lab = L.label(x16, self.tables)
        fcls, fspan = L.label_facts(x16, self.tables)
        drop = L.thinned(lab, fcls, fspan, pos, self.keep, self.fkeep)
        drop[~ok] = True
        y = np.where(drop[:, 1:], -100, x[:, 1:])
        if self.rule is not None:
            y = self.rule(x, y, fcls, fspan, drop)
        return x[:, :-1], y, int((y != -100).sum())


class Trainer:
    """One process of a run: device, model, AdamW, evaluation and the training loop. Under torchrun there is one
    process per GPU, and rank 0 writes the outputs."""

    def __init__(self, a, evaluator=Evaluator):
        self.a, self.out = a, Path(a.out)
        self.world, self.rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
        self.ddp = self.world > 1
        if self.ddp:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
            self.dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        else:
            self.dev = torch.device(a.device)
        self.cuda = self.dev.type == "cuda"
        self.per = a.batch // self.world                 # rows of a batch read by this process
        self.micro = min(a.micro, self.per)
        if a.batch % self.world or self.per % self.micro:
            raise SystemExit("the batch must divide into processes and micro-batches")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        self.out.mkdir(parents=True, exist_ok=True)
        vocab = L.load_vocab(a.tokenizer)
        self.tables, self.pool = L.Tables(vocab), Pool(a.data)
        self.model = build_model(a.size, a.seed, self.dev)
        if a.grad_ckpt or a.ckpt_layers:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for i, layer in enumerate(self.model.model.layers):
                layer.gradient_checkpointing = a.grad_ckpt or i < a.ckpt_layers
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                                     weight_decay=0.05, fused=self.cuda)
        self.net = LossSum(self.model)
        if self.ddp:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[self.dev.index], gradient_as_bucket_view=True, bucket_cap_mb=100)
        self.fwd = torch.compile(self.net) if a.compile and self.cuda else self.net
        if a.terms:                                      # adds the term accuracy of the filtered domains
            from thinning.webtext.docfilter import TermEvaluator
            evaluator = functools.partial(TermEvaluator, a.terms)
        self.evaluator = evaluator(a.eval, vocab, self.tables, self.dev) if a.eval else None

    def reader(self, order, keep: str, fkeep: str, rule=None) -> Reader:
        args = (self.pool, order, self.a.batch, self.rank * self.per, (self.rank + 1) * self.per,
                L.parse_fractions(keep, L.STRUCTURAL), L.parse_fractions(fkeep, L.FACTUAL), self.tables, rule)
        if not self.a.filter:
            return Reader(*args)
        from thinning.webtext.docfilter import FilteredReader
        return FilteredReader(*args, filter_dir=self.a.filter_dir, shares=self.a.filter)

    def data_order(self, horizon: int) -> np.ndarray:
        """The order of the seed for ``horizon`` steps, stored as order.npy."""
        path = self.out / "order.npy"
        if path.exists():
            return np.load(path)
        order = self.pool.order(horizon * self.a.batch, 1_000_003 * (self.a.seed + 1))
        if self.rank == 0:
            np.save(path, order)
        return order

    def load_state(self, path) -> int:
        """Load weights and AdamW state; returns the step at which they were written."""
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        return int(ck["step"])

    def autocast(self):
        return torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.cuda)

    def train_step(self, x, y, n_valid, lr):
        """One optimizer step on the rows of this process; returns the summed loss, the gradient norm and the
        number of targets of the whole batch."""
        x, y = torch.from_numpy(x).to(self.dev), torch.from_numpy(y).to(self.dev)
        if self.ddp:
            total = torch.tensor([n_valid], device=self.dev, dtype=torch.float64)
            dist.all_reduce(total)
            n_valid = float(total.item())
        scale = self.world / max(float(n_valid), 1.0)    # DDP averages the gradients of the processes
        for g in self.opt.param_groups:
            g["lr"] = lr
        tot = torch.zeros((), device=self.dev)
        n_micro = self.per // self.micro
        for j in range(n_micro):
            sl = slice(j * self.micro, (j + 1) * self.micro)
            with self.net.no_sync() if self.ddp and j < n_micro - 1 else contextlib.nullcontext():
                with self.autocast():
                    loss = self.fwd(x[sl], y[sl])
                (loss * scale).backward()
            tot += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return tot, grad_norm, float(n_valid)

    def log(self, name, row):
        with open(self.out / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    def evaluate(self, **row):
        """Append one row to eval.jsonl: the given keys and the measures of the model."""
        if self.evaluator is not None:
            r = self.evaluator.measure(self.model, bs=8, rank=self.rank, world=self.world)
            self.model.train()
            if self.rank == 0:
                self.log("eval.jsonl", dict(row, **r))

    def fit(self, reader, offset, step0, end, lr_at, eval_at, save_at, extra=lambda step: {}):
        """Train steps step0 + 1 .. end; step k reads batch offset + k - 1 of the reader."""
        a, tokens = self.a, lambda step: step * self.a.batch * SEQ
        ahead = ThreadPoolExecutor(1)                    # the next batch is read while the current one trains
        batch = ahead.submit(reader.step, offset + step0)
        self.model.train()
        for step in range(step0 + 1, end + 1):
            lr, (x, y, n_valid) = lr_at(step), batch.result()
            if step < end:
                batch = ahead.submit(reader.step, offset + step)
            tot, grad_norm, n_valid = self.train_step(x, y, n_valid, lr)
            if step % a.log_every == 0 or step in (step0 + 1, end):
                if self.ddp:
                    dist.all_reduce(tot)
                loss = float(tot) / max(n_valid, 1)
                if not math.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                if self.rank == 0:
                    self.log("train.jsonl", dict(step=step, tokens=tokens(step), **extra(step), loss=loss, lr=lr,
                                                 grad_norm=float(grad_norm)))
            if step in eval_at:
                self.evaluate(step=step, tokens=tokens(step), **extra(step), lr=lr)
            if step in save_at and self.rank == 0:
                save_snapshot(self.model, self.out / f"step{step}", dict(step=step, tokens=tokens(step)))
            if a.state_every and (step % a.state_every == 0 or step == end):
                if self.rank == 0:
                    state = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), step=step)
                    torch.save(state, self.out / "state.tmp")
                    os.replace(self.out / "state.tmp", self.out / "state.pt")
                if self.ddp:
                    dist.barrier()
        if self.ddp:
            dist.barrier()
            dist.destroy_process_group()


def add_arguments(ap):
    """The options shared by training and return runs."""
    ap.add_argument("--size", required=True, help="20M, 60M, 150M, 300M, 1B, 7B or <layers>x<width>x<heads>")
    ap.add_argument("--seed", type=int, default=0, help="initialization and data order")
    ap.add_argument("--keep", default="", help="retained fractions, e.g. close=0.3,agr=0.05,succ=0.3,copy=1")
    ap.add_argument("--fkeep", default="", help="retained fractions of the factual classes, e.g. cap=0.1,ctry=1")
    ap.add_argument("--filter", default="", help="share f of each filtered domain, e.g. rust=0.03,astro=1,zh=0.3")
    ap.add_argument("--filter-dir", default=None, help="index, scores and thresholds (thinning.webtext.docfilter)")
    ap.add_argument("--terms", default=None, help="terms.npz: term accuracy of the three domains in eval.jsonl")
    ap.add_argument("--data", required=True, help="staged pools (thinning.webtext.dolma)")
    ap.add_argument("--eval", default=None, help="evaluation sets (thinning.webtext.evalsets); none: no evaluation")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-tokens", type=float, required=True, help="tokens of this run")
    ap.add_argument("--horizon-tokens", type=float, default=8e9, help="tokens the data order is drawn for")
    ap.add_argument("--batch", type=int, default=128, help="rows per step")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--warmup", type=int, default=32, help="steps of the linear warm-up")
    ap.add_argument("--micro", type=int, default=16, help="rows per forward pass of a process")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing in every layer")
    ap.add_argument("--ckpt-layers", type=int, default=0, help="activation checkpointing in the first K layers")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile on CUDA")
    ap.add_argument("--dense-eval", default="", help="extra evaluation windows a:b:k,...")
    ap.add_argument("--state-every", type=int, default=0, help="steps between writes of state.pt (0: never)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a web-text model with target thinning")
    add_arguments(ap)
    ap.add_argument("--save-every-log2", type=int, default=0, help="k: snapshots at steps 16 * 2^(i/k)")
    a = ap.parse_args(argv)
    tr = Trainer(a)
    horizon = int(round(a.horizon_tokens / (a.batch * SEQ)))
    end = min(horizon, int(math.ceil(a.cap_tokens / (a.batch * SEQ))))
    step0 = tr.load_state(tr.out / "state.pt") if (tr.out / "state.pt").exists() else 0
    lr_at = warmup_then_constant(a.lr, a.warmup)
    if step0 == 0:
        tr.evaluate(step=0, tokens=0, lr=lr_at(0))
    reader = tr.reader(tr.data_order(horizon), a.keep, a.fkeep)
    tr.fit(reader, 0, step0, end, lr_at, eval_steps(horizon, end, a.dense_eval), snapshot_steps(end, a.save_every_log2),
           lambda step: dict(rows_read=reader.rows_read) if a.filter else {})


if __name__ == "__main__":
    main()
