"""The thinning relation: offset(f) = kappa * ln[(f + tau) / (1 + tau)], with corpus support tau.

Derived form. At each test position the fully supervised model puts mass s_x on the skill's answer and o_x on the
competing answer (chess: legal and king-unsafe replies; other domains: s_x = q_x, o_x = 1 - q_x). Thinning to a
retained fraction f shifts the log odds of every position by ln[(f + tau) / (1 + tau)]. tau is read from the
never-supervised model: it is the value at which the shift for f = 0, ln[tau / (1 + tau)], brings the pooled log
odds of the fully supervised model to those of the never-supervised model at the same background loss. No thinned
model enters. Pooling over positions scales the response by a sensitivity kappa < 1.

    python -m thinning.analysis.thinning_relation --full eval/full --never eval/never \
        --thinned 0.1=eval/f0.1 0.3=eval/f0.3 --grid 1.85:2.80:0.05

reads the per-position files written by ``python -m thinning.evaluate --per-position`` and prints the derived and
the measured offsets.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from thinning.analysis.trajectories import on_grid

MIN_SHIFT = -40.0


def relation(f, kappa, tau):
    return kappa * np.log((np.asarray(f, dtype=float) + tau) / (1 + tau))


def fit_relation(f, offset, kappa=None, log_tau=(-14.0, 6.0)) -> tuple[float, float]:
    """Least-squares (kappa, tau) of the relation for offsets measured at retained fractions f. kappa is linear
    and solved in closed form (limited to [0.01, 5]) for each tau; ln tau is searched on a grid and refined.
    With ``kappa`` given, only tau is fitted."""
    f, d = np.asarray(f, dtype=float), np.asarray(offset, dtype=float)
    grid = np.linspace(*log_tau, int(round((log_tau[1] - log_tau[0]) * 8)) + 1)

    def solve(lt):
        z = np.log((f + np.exp(lt)) / (1 + np.exp(lt)))
        k = float(np.clip(d @ z / max(z @ z, 1e-300), 0.01, 5)) if kappa is None else kappa
        return float(np.sum((k * z - d) ** 2)), k

    loss = np.array([solve(lt)[0] for lt in grid])
    candidates = [(loss[0], grid[0]), (loss[-1], grid[-1])]
    for i in np.argsort(loss)[:3]:
        bounds = (grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)])
        best = minimize_scalar(lambda lt: solve(lt)[0], bounds=bounds, method="bounded", options=dict(xatol=1e-8))
        candidates.append((best.fun, best.x))
    lt = min(candidates)[1]
    return solve(lt)[1], float(np.exp(lt))


def pooled_log_odds(skill_mass, other_mass, shift: float = 0.0) -> float:
    """Log odds of the pooled masses after shifting the log odds of every position by ``shift`` at fixed
    total mass per position."""
    total = skill_mass + other_mass
    odds = skill_mass / other_mass * np.exp(shift)
    return float(np.log((total * odds / (1 + odds)).sum() / (total / (1 + odds)).sum()))


def corpus_support(skill_full, other_full, never_log_odds: float) -> float:
    """tau from the fully supervised model's per-position masses and the never-supervised model's pooled log
    odds at the same background loss. Infinite when the never-supervised model is not below the fully supervised
    one; the smallest representable value when the shift would have to pass ``MIN_SHIFT``."""
    if never_log_odds >= pooled_log_odds(skill_full, other_full):
        return float("inf")
    if pooled_log_odds(skill_full, other_full, MIN_SHIFT) > never_log_odds:
        shift = MIN_SHIFT
    else:
        shift = brentq(lambda d: pooled_log_odds(skill_full, other_full, d) - never_log_odds, MIN_SHIFT, 0.0)
    return float(1 / np.expm1(-shift))


def derived_offset(f: float, skill_full, other_full, never_log_odds: float) -> tuple[float, float]:
    """(offset of the pooled log odds at retained fraction f, tau) at one background loss."""
    tau = corpus_support(skill_full, other_full, never_log_odds)
    if np.isinf(tau):
        return (0.0 if f > 0 else float("nan")), tau
    shift = float(np.log((f + tau) / (1 + tau)))
    return pooled_log_odds(skill_full, other_full, shift) - pooled_log_odds(skill_full, other_full), tau


def pooling_sensitivity(skill_mass, other_mass) -> float:
    """kappa at f = 1 from the spread of per-position competence q_x in the fully supervised model:
    1 - Var(q_x) / [q (1 - q)], with positions weighted by their total mass."""
    total = skill_mass + other_mass
    q_x = skill_mass / total
    q = (total * q_x).sum() / total.sum()
    return float(1 - (total * (q_x - q) ** 2).sum() / total.sum() / (q * (1 - q)))


def masses_on_grid(run_dir, grid, interpolate: str = "log"):
    """Per-position (skill mass, other mass) of a run on a grid of background loss, from its per-position files.
    ``log`` interpolates each mass in log (chess); ``logit`` interpolates the log odds and gives every position
    total mass 1 (the other domains)."""
    files = sorted(Path(run_dir).glob("*.npz"))
    snaps = [np.load(p) for p in files]
    loss = [float(z["background_loss"]) for z in snaps]
    skill = np.array([z["skill_mass"] for z in snaps], dtype=np.float64)
    other = np.array([z["other_mass"] for z in snaps], dtype=np.float64)
    if interpolate == "log":
        to_log = lambda y: np.log(np.clip(y, 1e-9, None))  # noqa: E731
        return np.exp(on_grid(loss, to_log(skill), grid)), np.exp(on_grid(loss, to_log(other), grid))
    z = on_grid(loss, np.log(skill / other), grid)
    return 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(z))


def compare(full_dir, never_dir, thinned: dict, grid, interpolate: str = "log") -> list[dict]:
    """Derived and measured offsets of each thinned run at each grid value covered by all three runs."""
    s1, o1 = masses_on_grid(full_dir, grid, interpolate)
    s0, o0 = masses_on_grid(never_dir, grid, interpolate)
    rows = []
    for f, run_dir in thinned.items():
        sf, of = masses_on_grid(run_dir, grid, interpolate)
        for j, loss in enumerate(grid):
            if not (np.isfinite(s1[j]).all() and np.isfinite(s0[j]).all() and np.isfinite(sf[j]).all()):
                continue
            full = pooled_log_odds(s1[j], o1[j])
            derived, tau = derived_offset(f, s1[j], o1[j], float(np.log(s0[j].sum() / o0[j].sum())))
            rows.append(dict(f=f, background_loss=float(loss), tau=tau, derived=derived,
                             measured=float(np.log(sf[j].sum() / of[j].sum())) - full,
                             sensitivity=pooling_sensitivity(s1[j], o1[j])))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derived and measured offsets of thinned models")
    ap.add_argument("--full", required=True, help="per-position files of the fully supervised run")
    ap.add_argument("--never", required=True, help="per-position files of the never-supervised run")
    ap.add_argument("--thinned", nargs="+", required=True, help="f=DIR for each thinned run")
    ap.add_argument("--grid", default="1.85:2.80:0.05", help="lo:hi:step of background loss")
    ap.add_argument("--interpolate", choices=("log", "logit"), default="log")
    a = ap.parse_args(argv)
    lo, hi, step = (float(v) for v in a.grid.split(":"))
    grid = np.round(np.arange(lo, hi + step / 2, step), 4)
    thinned = {float(t.split("=", 1)[0]): t.split("=", 1)[1] for t in a.thinned}
    rows = compare(a.full, a.never, thinned, grid, a.interpolate)
    print("f background_loss tau derived measured")
    for r in rows:
        print(f"{r['f']:g} {r['background_loss']:.3f} {r['tau']:.4g} {r['derived']:+.4f} {r['measured']:+.4f}")
    error = np.array([r["derived"] - r["measured"] for r in rows])
    print(f"r.m.s. error {np.sqrt(np.mean(error ** 2)):.4f} logits over {len(rows)} cells")
    return rows


if __name__ == "__main__":
    main()
