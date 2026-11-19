"""Remove a learned skill from a fully supervised model by one of three routes, then return its targets.

    python -m thinning.removal --route gradient-ascent --source runs/chess-f1/snap_000154768384.pt \
        --data data/chess/train --labels data/chess/labels.npy --eval-data data/chess/eval \
        --competence-target 0.38 --shares 1,0.1,0 --budgets 20e6,30e6,8e6 --out removal/chess-ga.json

Routes
  none             no removal: the source model is a never-supervised model, and only the return is run.
  withdrawn        the source run continues without the skill's targets for ``--withdraw-tokens`` (fresh AdamW,
                   moments estimated at learning rate 0, 20-step ramp to the rate of the source schedule).
  gradient-ascent  descent on the mean cross-entropy of the other targets minus ``--removal-weight`` times the
                   mean cross-entropy of the skill's targets.
  misdirection     at the skill's positions the residual stream after one block is pushed towards a fixed random
                   direction of ``--direction-scale`` times its mean norm; at the other positions it is held to
                   its value in the model before removal (weight ``--removal-weight``). Only that block and the
                   two before it are trained.
The two unlearning routes evaluate every ``--removal-eval-every`` steps and keep the state of lowest competence
whose background loss is at most ``--max-loss-increase`` above the start. They stop when the background loss
passes that limit, when competence has had no new minimum for ``--plateau-steps`` steps, at
``--competence-target`` (the competence of the never-supervised model of the case) or at ``--removal-max-steps``.

The return starts from a fresh AdamW: moments are estimated over 20 steps at learning rate 0 and the learning
rate rises over ``--warmup-before`` steps, all without the skill's targets. Every returned share then starts from
a copy of the same weights and optimizer state and runs until competence is at ``--stop-level`` at two
consecutive evaluations or the budget is spent.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from thinning import domains
from thinning.data import TokenStream
from thinning.model import build_optimizer
from thinning.returns import eval_steps, evaluate, rebuild_optimizer, source_lr
from thinning.train import Session, default_device, load_snapshot, open_labels, set_lr

ROUTES = {"none": {}, "withdrawn": {},
          "gradient-ascent": dict(removal_lr=1e-4, removal_weight=0.3, removal_eval_every=2),
          "misdirection": dict(removal_lr=3e-4, removal_weight=100.0, removal_eval_every=4)}
MOMENT_STEPS, RAMP_STEPS, UNLEARNING_RAMP_STEPS = 20, 20, 5


def gradient_ascent_step(session, opt, x, y, skill, weight: float, lr_mult: float) -> None:
    valid = y != -100
    n_skill, n_rest = (valid & skill).sum().clamp(min=1), (valid & ~skill).sum().clamp(min=1)
    session.step(opt, x, y, lr_mult, weights=torch.where(skill, -weight / n_skill, 1.0 / n_rest), n_targets=1)


class Misdirection:
    """Representation unlearning at one block of the model; ``x, y, skill`` is the first batch, whose first
    forward chunk sets the mean norm of the residual stream at the skill's positions."""

    def __init__(self, session, x, y, skill, layer_fraction: float, scale: float, weight: float, lr: float,
                 seed: int):
        model = session.model
        n_layers = len(model.layers)
        self.layer = max(2, min(n_layers - 1, int(round(layer_fraction * n_layers)) - 1))
        blocks = range(self.layer - 2, self.layer + 1)
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("layers.") and int(name.split(".")[1]) in blocks)
        self.frozen = copy.deepcopy(model).eval().requires_grad_(False)
        self.seen, self.weight = {}, weight
        self.hooks = [m.layers[self.layer].register_forward_hook(lambda _m, _i, out, k=k: self.seen.update({k: out}))
                      for k, m in (("model", model), ("frozen", self.frozen))]
        n = session.max_rows or x.shape[0]
        with torch.no_grad(), session.autocast():
            self.frozen(x[:n])
        norm = float(self.seen["frozen"].float()[skill[:n] & (y[:n] != -100)].norm(dim=-1).mean())
        direction = torch.rand(model.size.d_model, generator=torch.Generator().manual_seed(1234 + seed))
        self.target = scale * norm * (direction / direction.norm()).to(x.device)
        self.norm2 = norm ** 2
        self.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95),
                                     weight_decay=0.0)
        for group in self.opt.param_groups:
            group["peak_lr"] = group["lr"]

    def step(self, session, x, y, skill, lr_mult: float) -> None:
        valid = y != -100
        at_skill, at_rest = valid & skill, valid & ~skill
        n_skill, n_rest = at_skill.sum().clamp(min=1), at_rest.sum().clamp(min=1)
        set_lr(self.opt, lr_mult)
        self.opt.zero_grad(set_to_none=True)
        for r0, r1 in session.chunks(x.shape[0]):
            with session.autocast():
                with torch.no_grad():
                    self.frozen(x[r0:r1])
                session.model(x[r0:r1])
                h, h_frozen = self.seen["model"].float(), self.seen["frozen"].float()
                push = ((h[at_skill[r0:r1]] - self.target) ** 2).sum() / n_skill / self.norm2
                hold = ((h[at_rest[r0:r1]] - h_frozen[at_rest[r0:r1]]) ** 2).sum() / n_rest / self.norm2
                loss = push + self.weight * hold
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in session.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()

    def close(self, model) -> None:
        for hook in self.hooks:
            hook.remove()
        model.requires_grad_(True)


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--route", choices=list(ROUTES), required=True)
    route = ROUTES[pre.parse_known_args(argv)[0].route]
    ap = argparse.ArgumentParser(description="Remove a skill and return its targets", parents=[pre])
    ap.add_argument("--source", type=Path, required=True, help="snapshot of the source model")
    ap.add_argument("--domain", default="chess", choices=["chess", "smiles", "othello", "dyck", "webtext"])
    ap.add_argument("--skill", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--skill-classes", default=None, help="label classes of the skill (default: the domain's)")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-seqs", type=int, default=64, help="sequences per step (SMILES: 16)")
    ap.add_argument("--max-rows", type=int, default=32, help="rows per forward pass")
    ap.add_argument("--lr", type=float, default=0.004, help="peak learning rate of the source recipe")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--withdraw-tokens", type=float, default=8e6)
    ap.add_argument("--removal-lr", type=float)
    ap.add_argument("--removal-weight", type=float)
    ap.add_argument("--removal-eval-every", type=int)
    ap.add_argument("--removal-max-steps", type=int, default=600)
    ap.add_argument("--competence-target", type=float, default=0.0)
    ap.add_argument("--max-loss-increase", type=float, default=0.05)
    ap.add_argument("--plateau-steps", type=int, default=60)
    ap.add_argument("--misdirection-layer", type=float, default=0.7, help="block, as a fraction of the depth")
    ap.add_argument("--direction-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random direction")
    ap.add_argument("--warmup-before", type=int, default=30)
    ap.add_argument("--shares", default="1,0.1,0", help="returned shares")
    ap.add_argument("--budgets", default="20e6,30e6,8e6", help="token budget of each share")
    ap.add_argument("--stop-level", type=float, default=0.80)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--eval-offset", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.set_defaults(**route)
    a = ap.parse_args(argv)

    domain = domains.get(a.domain)
    skill = a.skill or next(iter(domain.SKILLS))
    model, src, source_tokens, row = load_snapshot(a.source, a.device)
    seq_len, salt, n = src["seq_len"], src["salt"], a.batch_seqs
    classes = a.skill_classes or ",".join(str(c) for c in domain.SKILLS[skill])
    session = Session(model, TokenStream(a.data), seq_len, *open_labels(a.labels, classes), a.max_rows, a.device)
    evalset = domain.EvalSet(TokenStream(a.eval_data), a.n_eval,
                             **({} if a.eval_offset is None else {"offset": a.eval_offset}))
    step_tokens = n * seq_len
    schedule = eval_steps("power", 1.25, 0, 10 ** 7)

    def measure(**at):
        r = evaluate(model, domain, evalset, skill, a.device) | at
        print(json.dumps(r), flush=True)
        return r

    def source_rate(tokens):
        return source_lr(tokens, src["warmup_tokens"], step_tokens)

    def weights_copy():
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    removal, used, stop, kept = [], 0, None, None
    if a.route == "withdrawn":
        opt = rebuild_optimizer(session, a.lr, a.weight_decay, max(0, row - MOMENT_STEPS * n), MOMENT_STEPS, n,
                                src["keep"], salt, False)
        step = 0
        while True:
            if step in schedule or used >= a.withdraw_tokens:
                removal.append(measure(step=step, tokens=used))
                if used >= a.withdraw_tokens:
                    stop = "tokens"
                    break
            x, y, _ = session.rows(row, n, 0.0, salt)
            session.step(opt, x, y, min(1.0, (step + 1) / RAMP_STEPS) * source_rate(source_tokens + used))
            row, step, used = row + n, step + 1, used + step_tokens
    elif a.route != "none":
        mis = None
        if a.route == "misdirection":
            mis = Misdirection(session, *session.rows(row, n), a.misdirection_layer, a.direction_scale,
                               a.removal_weight, a.removal_lr, a.seed)
        opt = mis.opt if mis else build_optimizer(model, a.removal_lr, a.weight_decay)
        step, best = 0, None
        while True:
            if step % a.removal_eval_every == 0 or step >= a.removal_max_steps:
                r = measure(step=step, tokens=used)
                removal.append(r)
                within = r["background_loss"] <= removal[0]["background_loss"] + a.max_loss_increase
                if within and (best is None or r["competence"] < best[1]["competence"]):
                    best = (weights_copy(), r, row, used)
                if best is not None and step - best[1]["step"] >= a.plateau_steps:
                    stop = "plateau"
                elif not within:
                    stop = "background loss"
                elif r["competence"] <= a.competence_target and step > 0:
                    stop = "target"
                elif step >= a.removal_max_steps:
                    stop = "steps"
                if stop:
                    break
            x, y, at_skill = session.rows(row, n)
            ramp = min(1.0, (step + 1) / UNLEARNING_RAMP_STEPS)
            if mis:
                mis.step(session, x, y, at_skill, ramp)
            else:
                gradient_ascent_step(session, opt, x, y, at_skill, a.removal_weight, ramp)
            row, step, used = row + n, step + 1, used + step_tokens
        state, kept, row, used = best
        model.load_state_dict(state)
        if mis:
            mis.close(model)

    opt = rebuild_optimizer(session, a.lr, a.weight_decay, max(0, row - MOMENT_STEPS * n), MOMENT_STEPS, n, 0.0,
                            salt, False)
    at_return = source_tokens + used
    before = [measure(step=0)]
    for k in range(a.warmup_before):
        x, y, _ = session.rows(row + k * n, n, 0.0, salt)
        session.step(opt, x, y, source_rate(at_return) * (k + 1) / a.warmup_before)
    before.append(measure(step=a.warmup_before))
    row, at_return = row + a.warmup_before * n, at_return + a.warmup_before * step_tokens
    weights, moments = weights_copy(), copy.deepcopy(opt.state_dict())
    returns = {}
    for share, budget in zip(a.shares.split(","), a.budgets.split(",")):
        model.load_state_dict(weights)
        opt.load_state_dict(copy.deepcopy(moments))
        rows, hits, step = [], 0, 0
        while True:
            tokens = step * step_tokens
            if step in schedule or tokens >= float(budget):
                rows.append(measure(share=float(share), step=step, tokens=tokens))
                hits = hits + 1 if rows[-1]["competence"] >= a.stop_level else 0
                if hits >= 2 or tokens >= float(budget):
                    break
            x, y, _ = session.rows(row + step * n, n, float(share), salt)
            session.step(opt, x, y, source_rate(at_return + tokens))
            step += 1
        returns[share] = rows
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update(skill=skill, source_tokens=source_tokens, removal_tokens=used, n_params=model.size.n_params)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(config=config, removal=removal, removal_stop=stop, kept=kept,
                                     before_return=before, returns=returns), indent=1))
    return returns


if __name__ == "__main__":
    main()
