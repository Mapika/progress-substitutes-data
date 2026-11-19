"""Return a skill's withheld targets to a source model and record competence against the tokens after the return.

    python -m thinning.returns --preset chess --source runs/chess-f0/snap_000619073536.pt \
        --data data/chess/train --labels data/chess/labels.npy --eval-data data/chess/eval \
        --share 1 --max-tokens 40e6 --out returns/chess-9.7M-s1.json

A source model was trained with a retained fraction of the skill's targets (none for a never-supervised source). Stored snapshots hold
weights only, so the optimizer state is rebuilt first, and the steps that rebuild it contain none of the returned
targets and are not counted:

  moment estimation   ``--moment-steps`` steps at learning rate 0 on the source model's own targets: the AdamW
                      moments are estimated and no parameter changes;
  warm-up before      ``--warmup-before`` steps on the source model's own targets while the learning rate rises;
  the return          the targets return at share ``--share`` (retained fraction of the position hash), at full
                      learning rate or, with ``--warmup-at``, with a warm-up that starts at the return.

``--rule permuted`` returns the targets under the permuted rule (chess: each move target at an in-check position
is replaced by its image under a fixed derangement of the move tokens). Without ``--source``, ``--size`` and
``--seed`` train from initialization with the same loop, which gives the denominator of the acquisition cost.

Presets hold the settings of each domain (see ``PRESETS``); every value can be overridden on the command line.
The web-text preset differs from the others as follows: the moment-estimation rows and the returned rows are
fixed rows that no source run has read, the first moment is set to zero after the estimation, the steps before
the return run at the learning rate of the source schedule without a ramp, and evaluations are spaced by 15%.

The output file holds the configuration and one row per evaluation: step and tokens after the return, and the
scalars of the domain's ``measure`` (competence, background loss, ...). Evaluations fall at the return (step 0)
and then on the grid of ``eval_steps``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thinning import domains
from thinning.data import TokenStream
from thinning.masks import derangement
from thinning.model import Decoder, build_optimizer, init_weights
from thinning.sizes import get_size
from thinning.train import Session, default_device, load_snapshot, open_labels, set_lr, warmup_lr

PRESETS = {
    "chess": dict(domain="chess", lr=0.004, batch_seqs=64, warmup_before=30, warmup_at=0, n_eval=1000,
                  stop_level=0.80),
    "chess-warmup-at-return": dict(domain="chess", lr=0.004, batch_seqs=64, warmup_before=0, warmup_at=20,
                                   n_eval=1000, stop_level=0.80),
    "smiles": dict(domain="smiles", lr=0.004, batch_seqs=16, warmup_before=0, warmup_at=20, n_eval=8000,
                   stop_level=0.75),
    "othello": dict(domain="othello", lr=0.004, batch_seqs=64, warmup_before=0, warmup_at=20, n_eval=2000,
                    stop_level=0.6),
    "webtext": dict(domain="webtext", lr=0.008, batch_seqs=64, warmup_before=30, warmup_at=0, n_eval=384,
                    stop_level=0.90, stop_evals=1, stop_after_level=0.85, eval_grid="compound", eval_factor=1.15,
                    start_row=800_000,
                    moment_start_row=700_000, reset_first_moment=True, constant_warmup_before=True),
}


def eval_steps(grid: str, factor: float, dense_until: int, max_steps: int) -> set:
    """Evaluation steps. ``power``: step 0 and the rounded powers of ``factor``. ``compound``: steps 0-6, 8, 10,
    12, 14 and 16, then each step is the previous one times ``factor``, rounded. Every step up to
    ``dense_until`` is added."""
    if grid == "compound":
        steps = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
        while steps[-1] < max_steps:
            steps.append(int(round(steps[-1] * factor)))
    else:
        steps = [0] + [int(round(factor ** k)) for k in range(200) if factor ** k <= 2 * max_steps + 2]
    return set(steps) | set(range(dense_until + 1))


def evaluate(model, domain, evalset, skill, device, perm=None) -> dict:
    """Scalars of the domain's measure, computed in evaluation mode with the dense attention mask."""
    impl, model.attn_impl = model.attn_impl, "sdpa"
    model.eval()
    args = {} if perm is None else {"perm": perm}
    out = domain.measure(model, evalset, skill, device, **args)
    model.train()
    model.attn_impl = impl
    return {k: v for k, v in out.items() if np.isscalar(v)}


def source_lr(source_tokens: float, source_warmup: float, step_tokens: int) -> float:
    """Learning-rate multiplier of the source run's own schedule at ``source_tokens`` (at least one step's
    worth), so that a snapshot taken inside the warm-up continues it."""
    return max(warmup_lr(source_tokens, source_warmup), step_tokens / source_warmup)


def rebuild_optimizer(session, lr, weight_decay, first_row, n_steps, batch_seqs, keep, salt, reset_first_moment):
    """A fresh AdamW whose moments are estimated over ``n_steps`` steps at learning rate 0 on rows from
    ``first_row`` with the source model's retained fraction."""
    opt = build_optimizer(session.model, lr, weight_decay)
    for k in range(n_steps):
        x, y, _ = session.rows(first_row + k * batch_seqs, batch_seqs, keep, salt)
        session.step(opt, x, y, 0.0)
    if reset_first_moment:
        for state in opt.state.values():
            state["exp_avg"].zero_()
    return opt


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS), required=True)
    preset = PRESETS[pre.parse_known_args(argv)[0].preset]
    ap = argparse.ArgumentParser(description="Return withheld targets to a source model", parents=[pre])
    ap.add_argument("--source", type=Path, default=None, help="snapshot of the source model")
    ap.add_argument("--size", default=None, help="with --seed: train from initialization instead")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", required=True, help="training stream directory")
    ap.add_argument("--labels", required=True, help="label array of the training stream")
    ap.add_argument("--skill-classes", default=None, help="label classes of the skill (default: the source run's)")
    ap.add_argument("--eval-data", required=True, help="held-out stream directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domain", choices=list(domains.MODULES))
    ap.add_argument("--skill", default=None, help="skill name of the domain (default: its first skill)")
    ap.add_argument("--share", type=float, default=1.0, help="returned share s of the skill's targets")
    ap.add_argument("--rule", choices=["true", "permuted"], default="true")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch-seqs", type=int, help="sequences per step")
    ap.add_argument("--seq-len", type=int, default=None, help="default: the source run's sequence length")
    ap.add_argument("--max-rows", type=int, default=0, help="rows per forward pass (gradient accumulation)")
    ap.add_argument("--moment-steps", type=int, default=20)
    ap.add_argument("--reset-first-moment", action="store_true")
    ap.add_argument("--warmup-before", type=int, help="steps on the source model's targets before the return")
    ap.add_argument("--constant-warmup-before", action="store_true",
                    help="run those steps at the learning rate of the source schedule instead of a linear ramp")
    ap.add_argument("--warmup-at", type=int, help="warm-up steps that start when the targets return")
    ap.add_argument("--start-row", type=int, default=None, help="first returned row (default: the source's next)")
    ap.add_argument("--moment-start-row", type=int, default=None)
    ap.add_argument("--row-offset", type=int, default=0, help="skip this many rows before the returned rows")
    ap.add_argument("--max-tokens", type=float, required=True, help="token budget after the return")
    ap.add_argument("--eval-grid", choices=["power", "compound"], default="power")
    ap.add_argument("--eval-factor", type=float, default=1.25)
    ap.add_argument("--eval-every-step-until", type=int, default=0)
    ap.add_argument("--n-eval", type=int, help="held-out documents")
    ap.add_argument("--eval-offset", type=int, default=None, help="token offset of the held-out documents")
    ap.add_argument("--stop-level", type=float, help="stop after --stop-evals evaluations at or above it")
    ap.add_argument("--stop-evals", type=int, default=2)
    ap.add_argument("--stop-mass", type=float, default=0.0, help="chess: also require this legal-move mass")
    ap.add_argument("--stop-after-level", type=float, default=None,
                    help="also stop at twice the steps of the first evaluation at this level (at least 24 steps)")
    ap.add_argument("--device", default=default_device())
    ap.set_defaults(**preset)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    domain = domains.get(a.domain)
    skill = a.skill or next(iter(domain.SKILLS))
    stream = TokenStream(a.data)
    if a.source is not None:
        model, src, source_tokens, next_row = load_snapshot(a.source, a.device)
        source_keep, source_salt, source_warmup = src["keep"], src["salt"], src["warmup_tokens"]
        classes = a.skill_classes or src["skill_classes"]
        seq_len = a.seq_len or src["seq_len"]
    else:
        size = get_size(a.size)
        model = Decoder(size, stream.vocab_size, stream.sep_id,
                        attn_impl="varlen" if a.device.startswith("cuda") else "sdpa")
        init_weights(model, seed=a.seed)
        model = model.to(a.device)
        source_tokens, next_row, source_keep, source_salt, source_warmup = 0, 0, 1.0, 0, 2.0 * size.n_params
        classes, seq_len = a.skill_classes or ",".join(str(c) for c in domain.SKILLS[skill]), a.seq_len or 2048
    labels, classes = open_labels(a.labels, classes)
    session = Session(model, stream, seq_len, labels, classes, a.max_rows, a.device)
    perm = torch.from_numpy(derangement(domain.N_MOVES)) if a.rule == "permuted" else None
    step_tokens = a.batch_seqs * seq_len

    moment_row = a.moment_start_row
    if moment_row is None:
        moment_row = max(0, next_row - a.moment_steps * a.batch_seqs)
    before_row = next_row if a.moment_start_row is None else moment_row + a.moment_steps * a.batch_seqs
    n_moment = a.moment_steps if a.source is not None else 0
    opt = rebuild_optimizer(session, a.lr, a.weight_decay, moment_row, n_moment, a.batch_seqs, source_keep,
                            source_salt, a.reset_first_moment)
    before_lr = source_lr(source_tokens, source_warmup, step_tokens)
    for k in range(a.warmup_before):
        x, y, _ = session.rows(before_row + k * a.batch_seqs, a.batch_seqs, source_keep, source_salt)
        if a.constant_warmup_before:
            mult = warmup_lr(source_tokens + k * step_tokens, source_warmup) if source_tokens + k * step_tokens else 1.0
        else:
            mult = before_lr * (k + 1) / a.warmup_before
        session.step(opt, x, y, mult)
    if a.constant_warmup_before:
        source_tokens += a.warmup_before * step_tokens
    row = (a.start_row if a.start_row is not None else before_row + a.warmup_before * a.batch_seqs) + a.row_offset

    eval_args = {} if a.eval_offset is None else {"offset": a.eval_offset}
    evalset = domain.EvalSet(TokenStream(a.eval_data), a.n_eval, **eval_args)
    max_steps = int(np.ceil(a.max_tokens / step_tokens))
    schedule = eval_steps(a.eval_grid, a.eval_factor, a.eval_every_step_until, max_steps)
    key = "competence_permuted" if perm is not None else "competence"
    mass_key = "legal_mass_permuted" if perm is not None else "legal_mass"
    rows, hits, step, first_at_level, recent = [], 0, 0, None, []
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update(skill=skill, seq_len=seq_len, source_tokens=source_tokens, source_keep=source_keep,
                  first_returned_row=row, n_params=model.size.n_params)
    while True:
        if step in schedule or step >= max_steps:
            r = evaluate(model, domain, evalset, skill, a.device, None if perm is None else perm.numpy())
            r.update(step=step, tokens=step * step_tokens, train_loss=float(np.mean(recent)) if recent else None)
            rows.append(r)
            recent = []
            print(json.dumps(r), flush=True)
            reached = r[key] >= a.stop_level and (not a.stop_mass or r[mass_key] >= a.stop_mass)
            hits = hits + 1 if reached else 0
            if first_at_level is None and a.stop_after_level is not None and r[key] >= a.stop_after_level:
                first_at_level = step
            late = first_at_level is not None and step >= max(24, 2 * first_at_level)
            if (hits >= a.stop_evals and step >= (24 if a.stop_after_level is not None else 0)) or late \
                    or step >= max_steps:
                break
        x, y, _ = session.rows(row, a.batch_seqs, a.share, source_salt, perm)
        mult = min(1.0, (step + 1) / a.warmup_at) if a.warmup_at else 1.0
        if a.source is not None or not a.constant_warmup_before:
            mult *= source_lr(source_tokens + step * step_tokens, source_warmup, step_tokens)
        loss, _ = session.step(opt, x, y, mult)
        recent.append(loss)
        row += a.batch_seqs
        step += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(config=config, evaluations=rows), indent=1))
    return rows


if __name__ == "__main__":
    main()
