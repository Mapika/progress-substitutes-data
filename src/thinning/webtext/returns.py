"""Returned targets on web text: continue a source model with targets back in the loss.

Return from a snapshot (``--init-from RUN/step<S>``). The run continues the data order of the source run from
batch S, the first that the snapshot has not trained on, with the warm-up before the return:

  1. ``--mom-steps`` (20) steps at learning rate 0 on the batches S-20 .. S-1 under the source model's masks
     (``--src-keep``, ``--src-fkeep``): the AdamW moments are estimated and the weights do not change;
  2. ``--prewarm-steps`` (30) steps under the source model's masks while the learning rate rises linearly;
  3. from step 31 the masks of the return (``--keep``, ``--fkeep``, ``--fact-rule``) at the full learning rate.

    torchrun --nproc_per_node 4 -m thinning.webtext.returns --size 7B --lr 0.000265 --micro 8 \
        --init-from runs/source/step8192 --src-keep agr=0,copy=0 --src-fkeep cap=0 --keep agr=0,copy=0 \
        --fkeep cap=1 --cap-tokens 6e8 --dense-eval 0:400:10,400:4000:25 --data ... --eval ... --out ...

An agreement return keeps the capital targets removed and returns the auxiliaries: ``--keep agr=1,copy=0 --fkeep
cap=0``.

``--fact-rule cap=perm`` is the permuted assignment: of every retained capital mention only the first token is
supervised, and its target is replaced by the first token of the answer group that a fixed derangement of the 102
answer groups of the capital prompts assigns to the true group; mentions whose first token is in no group leave the
loss. ``--fact-rule cap=first`` is the matched true-rule control: the same positions with the true first token.

From initialization (no ``--init-from``): a model trained as ``thinning.webtext.train`` trains it, with the masks
and the rule of the run from the first step, as for the permuted assignment from initialization.

Continuation (``--resume-from RUN/state.pt --order RUN/order.npy --data-offset B``): the weights and the AdamW
state of a run are loaded and training goes on at the constant learning rate from batch B of the given data order,
on the unfiltered pools. It returns the unfiltered corpus to a model trained with document filtering and continues
the unfiltered model on the same batches.

eval.jsonl has the rows of a training run with three more keys: ``return_step`` and ``return_tokens`` count from
the return (0 during the warm-up before it; ``step`` and ``tokens`` count from the start of the run), and
``data_block`` is the batch of the data order that follows the step. With the permuted assignment the rows also have
``competence_capitals_permuted``, the mass on the assigned group. Evaluations fall at step 0 (after the moment
steps), at the end of the warm-up, on the log-spaced grid of the run's own length and in the ``--dense-eval``
windows. ``--save-steps`` writes snapshots at the given steps (0 is the loaded model) and at the last step.
"""

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
import torch

from thinning.webtext import labels as L
from thinning.webtext.entities import build_prompts
from thinning.webtext.evalsets import Evaluator
from thinning.webtext.train import SEQ, Trainer, add_arguments, eval_steps, save_snapshot, warmup_then_constant

PERM_SEED = 12345


def derangement(k: int, seed: int = PERM_SEED) -> np.ndarray:
    """The first seeded random permutation of 0..k-1 without a fixed point."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(k)
        if not np.any(p == np.arange(k)):
            return p


class AnswerRule:
    """Target rewriting for factual classes. ``rules`` maps a relation (``cap``, ``ctry``) to ``perm`` or ``first``;
    the answer groups are those of the evaluation prompts."""

    def __init__(self, vocab, rules: dict):
        prompts = build_prompts(vocab.encode)
        self.maps = {}
        for rel, rule in rules.items():
            if rule not in ("perm", "first"):
                raise SystemExit(f"--fact-rule: unknown rule {rule!r}")
            groups = prompts[rel]["groups"]
            group_of = np.full(L.VOCAB, -1, np.int64)
            for g, grp in enumerate(groups):
                group_of[np.asarray(grp, np.int64)] = g
            pi = derangement(len(groups)) if rule == "perm" else np.arange(len(groups))
            self.maps[L.FACTUAL[rel]] = dict(rule=rule, group_of=group_of, pi=pi,
                                             first_token=np.array([grp[0] for grp in groups], np.int64))

    def permutation(self, rel: str):
        """The derangement of the answer groups of a relation under the permuted assignment, else None."""
        m = self.maps.get(L.FACTUAL[rel])
        return m["pi"] if m and m["rule"] == "perm" else None

    def __call__(self, x, y, fcls, fspan, drop):
        """x (n, 2048) tokens; y (n, 2047) targets after thinning, y[:, c - 1] being the target x[:, c]; fcls,
        fspan, drop (n, 2048) labels and removed targets. Returns the rewritten targets."""
        y = y.copy()
        cols = np.arange(1, x.shape[1])[None, :]
        for code, m in self.maps.items():
            sel = (fcls[:, 1:] == code) & ~drop[:, 1:]
            if not sel.any():
                continue
            g = m["group_of"][x[:, 1:]]
            first = sel & (fspan[:, 1:] == cols) & (g >= 0)
            y[sel & ~first] = -100
            if m["rule"] == "perm":
                y[first] = m["first_token"][m["pi"][g[first]]]
        return y


def load_snapshot(model, path) -> dict:
    """Copy the weights of a snapshot into the fp32 parameters of the model; returns its meta.json."""
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        sd.update(load_file(str(f)))
    own = model.state_dict()
    if set(own) != set(sd):
        raise SystemExit(f"{path}: the parameters of the snapshot are not those of the model")
    with torch.no_grad():
        for k, v in own.items():
            v.copy_(sd[k].to(v.device))
    return json.loads((Path(path) / "meta.json").read_text())


class PermutedEvaluator(Evaluator):
    """The measures of ``Evaluator`` and ``competence_capitals_permuted``, the mass on the answer group that the
    permutation ``perm`` assigns to each capital prompt over the mass on all groups."""

    def __init__(self, perm, *args):
        super().__init__(*args)
        self.perm = torch.as_tensor(perm)

    def measure(self, model, bs=8, rank=0, world=1) -> dict:
        out = super().measure(model, bs, rank, world)
        was = model.training
        model.eval()
        out["competence_capitals_permuted"] = self.prompt_competence(model, "cap", 32 * bs, rank, world, self.perm)
        model.train(was)
        return out


class Phased:
    """The source reader for the batches before ``switch`` and the return reader from ``switch`` on."""

    def __init__(self, source, target, switch):
        self.source, self.target, self.switch = source, target, switch

    def step(self, i):
        return (self.source if i < self.switch else self.target).step(i)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Continue a web-text model with returned targets")
    add_arguments(ap)
    ap.add_argument("--init-from", default=None, help="snapshot of the source model, RUN/step<S>")
    ap.add_argument("--src-keep", default=None, help="masks of the source model (default: --keep)")
    ap.add_argument("--src-fkeep", default=None, help="(default: --fkeep)")
    ap.add_argument("--mom-steps", type=int, default=20, help="moment-estimation steps at learning rate 0")
    ap.add_argument("--prewarm-steps", type=int, default=30, help="warm-up steps before the targets return")
    ap.add_argument("--fact-rule", default="", help="cap=perm (permuted assignment) or cap=first (control)")
    ap.add_argument("--resume-from", default=None, help="state.pt of a run: continue with its AdamW state")
    ap.add_argument("--order", default=None, help="data order (default: order.npy beside the snapshot)")
    ap.add_argument("--data-offset", type=int, default=None, help="first batch (default: step of the snapshot)")
    ap.add_argument("--save-steps", default="", help="comma-separated steps for snapshots")
    a = ap.parse_args(argv)
    from_snapshot, continuation = a.init_from is not None, a.resume_from is not None
    if continuation and (from_snapshot or a.order is None or a.data_offset is None):
        raise SystemExit("--resume-from needs --order and --data-offset, and excludes --init-from")
    rules = dict(part.split("=") for part in filter(None, a.fact_rule.split(",")))
    rule = AnswerRule(L.load_vocab(a.tokenizer), rules) if rules else None
    perm = rule.permutation("cap") if rule else None
    tr = Trainer(a, Evaluator if perm is None else functools.partial(PermutedEvaluator, perm))
    if (tr.out / "eval.jsonl").exists():
        raise SystemExit(f"{tr.out}/eval.jsonl exists; use a new --out")
    end = int(math.ceil(a.cap_tokens / (a.batch * SEQ)))
    prewarm = a.prewarm_steps if from_snapshot else 0
    offset = 0
    if from_snapshot:
        order = np.load(a.order or Path(a.init_from).parent / "order.npy")
        meta = load_snapshot(tr.model, a.init_from)
        offset = int(meta["step"]) if a.data_offset is None else a.data_offset
        if offset < a.mom_steps:
            raise SystemExit("the moment steps need batches before the first batch of the run")
    elif continuation:
        order, offset = np.load(a.order), a.data_offset
        tr.load_state(a.resume_from)
    else:
        order = tr.data_order(int(round(a.horizon_tokens / (a.batch * SEQ))))
    if (offset + end) * a.batch > len(order):
        raise SystemExit(f"the data order has {len(order) // a.batch} batches; the run needs {offset + end}")
    reader = tr.reader(order, a.keep, a.fkeep, rule)
    tr.model.train()
    if from_snapshot:
        source = tr.reader(order, a.keep if a.src_keep is None else a.src_keep,
                           a.fkeep if a.src_fkeep is None else a.src_fkeep)
        for j in range(offset - a.mom_steps, offset):
            x, y, n_valid = source.step(j)
            tr.train_step(x, y, n_valid, 0.0)
        reader = Phased(source, reader, offset + prewarm)
    lr_at = warmup_then_constant(a.lr, prewarm if prewarm else 0 if continuation else a.warmup)

    def counted(step):
        since = max(0, step - prewarm)
        return dict(return_step=since, return_tokens=since * a.batch * SEQ, data_block=offset + step)

    saves = {int(v) for v in a.save_steps.split(",") if v.strip()}
    tr.evaluate(step=0, tokens=0, **counted(0), lr=lr_at(0))
    if 0 in saves and tr.rank == 0:
        save_snapshot(tr.model, tr.out / "step0", dict(step=0, tokens=0))
    tr.fit(reader, offset, 0, end, lr_at, eval_steps(end, end, a.dense_eval) | ({prewarm} if prewarm else set()),
           saves | {end} if saves else saves, counted)


if __name__ == "__main__":
    main()
