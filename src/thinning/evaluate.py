"""Evaluate the snapshots of a run on held-out documents: competence and background loss against training tokens.

    python -m thinning.evaluate --domain chess --run runs/chess-f0.1 --eval-data data/chess/eval \
        --out eval/chess-f0.1.csv

writes one row per snapshot. ``--per-position DIR`` also writes, for each snapshot, the skill mass and the
competing mass at every test position of the skill (the input of ``thinning.analysis.thinning_relation``), and
``--output-classes`` adds the mean probability of each class of output token at the skill's positions (chess and
SMILES).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from thinning import domains
from thinning.data import TokenStream
from thinning.train import default_device, load_snapshot

N_EVAL = {"chess": 1000, "othello": 8000, "smiles": 8000, "dyck": 2000, "webtext": 384}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the snapshots of a run")
    ap.add_argument("--domain", required=True, choices=list(domains.MODULES))
    ap.add_argument("--run", type=Path, required=True, help="run directory written by thinning.train")
    ap.add_argument("--eval-data", required=True, help="held-out stream directory")
    ap.add_argument("--out", type=Path, required=True, help="output CSV file")
    ap.add_argument("--skill", default=None, help="skill name of the domain (default: its first skill)")
    ap.add_argument("--n-eval", type=int, default=None, help="held-out documents or rows (default: N_EVAL)")
    ap.add_argument("--eval-offset", type=int, default=None, help="token offset of the held-out documents")
    ap.add_argument("--per-position", type=Path, default=None, help="directory for per-position masses")
    ap.add_argument("--output-classes", action="store_true")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args(argv)

    domain = domains.get(a.domain)
    skill = a.skill or next(iter(domain.SKILLS))
    evalset = domain.EvalSet(TokenStream(a.eval_data), a.n_eval or N_EVAL[a.domain],
                             **({} if a.eval_offset is None else {"offset": a.eval_offset}))
    rows = []
    for path in sorted(a.run.glob("snap_*.pt")):
        model, config, tokens, _ = load_snapshot(path, a.device)
        model.attn_impl = "sdpa"
        model.eval()
        out = domain.measure(model, evalset, skill, a.device, per_position=a.per_position is not None)
        row = dict(tokens=tokens, tokens_per_param=tokens / config["n_params"])
        row.update({k: v for k, v in out.items() if np.isscalar(v)})
        if a.output_classes:
            row.update(domain.output_classes(model, evalset, a.device))
        if a.per_position is not None:
            a.per_position.mkdir(parents=True, exist_ok=True)
            np.savez(a.per_position / f"{path.stem}.npz", tokens=tokens, background_loss=out["background_loss"],
                     skill_mass=out["skill_mass"], other_mass=out["other_mass"])
        rows.append(row)
        print(row, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
