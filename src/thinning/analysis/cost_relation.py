"""The cost relation: acquisition cost = P + 10^(a + b * min(x, 0)).

x is the logit distance at the return, P the progress term (``thinning.analysis.cost``). A row is a dictionary
with ``domain``, ``source`` (the source run), ``x``, ``P`` and ``y`` = log10 cost; the fits use the rows that
reached the acquisition level.

    python -m thinning.analysis.cost_relation table.csv

fits the single line, the per-domain lines with a random intercept for each source run, and the trend of the
residuals with model size (column ``n_params``) for a table with these columns and ``censored``.
"""

import csv
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares, minimize


def two_part(a, b, x, P):
    """log10 of the cost predicted for logit distance x and progress term P."""
    return np.log10(P + 10.0 ** (a + b * np.minimum(x, 0.0)))


def _arrays(rows):
    return (np.array([r["x"] for r in rows], dtype=float), np.array([r["P"] or 0.0 for r in rows], dtype=float),
            np.array([r["y"] for r in rows], dtype=float))


def _design(rows, per_domain: bool):
    domains = sorted({r["domain"] for r in rows})
    if not per_domain:
        return ["a", "b"], np.zeros(len(rows), int), np.ones(len(rows), int), domains
    names = [f"a@{d}" for d in domains] + [f"b@{d}" for d in domains]
    ia = np.array([domains.index(r["domain"]) for r in rows])
    return names, ia, ia + len(domains), domains


def _clusters(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["domain"], r["source"])].append(i)
    return [np.array(v) for v in groups.values()]


def fit_line(rows, per_domain: bool = False) -> dict:
    """Least squares of log10 cost on the two-part form, with one (a, b) or one per domain. Standard errors are
    from the sandwich estimator with source runs as clusters."""
    names, ia, ib, _ = _design(rows, per_domain)
    x, P, y = _arrays(rows)
    start = np.array([-1.0 if n.startswith("a") else 1.0 for n in names])
    fit = least_squares(lambda v: two_part(v[ia], v[ib], x, P) - y, start)
    J, e = fit.jac, fit.fun
    bread = np.linalg.pinv(J.T @ J)
    blocks = _clusters(rows)
    meat = sum(np.outer(J[i].T @ e[i], J[i].T @ e[i]) for i in blocks)
    cov = bread @ meat @ bread * (len(blocks) / max(len(blocks) - 1, 1))
    dof = max(len(rows) - len(names), 1)
    return dict(params=dict(zip(names, map(float, fit.x))),
                se={n: float(math.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)},
                residual_sd=float(math.sqrt(float(e @ e) / dof)), residuals=e, n=len(rows), clusters=len(blocks))


def fit_mixed(rows) -> dict:
    """Maximum-likelihood fit with an intercept a, a rate b and a residual s.d. for each domain and a normal
    random intercept for each source run. Standard errors are from the numerical Hessian."""
    names, ia, ib, domains = _design(rows, True)
    x, P, y = _arrays(rows)
    blocks = _clusters(rows)
    k, n_dom = len(names), len(domains)
    dom = np.array([domains.index(r["domain"]) for r in rows])

    def nll(v):
        e = y - two_part(v[ia], v[ib], x, P)
        var, between = np.exp(2 * v[k:k + n_dom])[dom], math.exp(2 * v[k + n_dom])
        total = 0.0
        for i in blocks:
            V = np.diag(var[i]) + between
            total += 0.5 * (np.linalg.slogdet(V)[1] + e[i] @ np.linalg.solve(V, e[i]) + len(i) * math.log(2 * math.pi))
        return total

    ls = fit_line(rows, per_domain=True)
    v0 = np.array([ls["params"][n] for n in names] + [math.log(ls["residual_sd"] * 0.8)] * n_dom
                  + [math.log(ls["residual_sd"] * 0.5)])
    best = minimize(nll, v0, method="L-BFGS-B")
    best = minimize(nll, best.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-9))
    v, h = best.x, 1e-4
    H = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(i, len(v)):
            ei, ej = np.eye(len(v))[i] * h, np.eye(len(v))[j] * h
            second = nll(v + ei + ej) - nll(v + ei - ej) - nll(v - ei + ej) + nll(v - ei - ej)
            H[i, j] = H[j, i] = second / (4 * h * h)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(H)), 0.0))
    return dict(params=dict(zip(names, map(float, v[:k]))), se=dict(zip(names, map(float, se[:k]))),
                residual_sd={d: float(math.exp(v[k + i])) for i, d in enumerate(domains)},
                source_sd=float(math.exp(v[k + n_dom])), neg_log_likelihood=float(best.fun), n=len(rows),
                clusters=len(blocks))


def ordering(rows, key: str, sign: int = 1) -> tuple[float | None, int]:
    """Share of pairs of runs within a domain whose order of cost agrees with the order of ``rows[key]``
    (sign +1: a larger value predicts a larger cost; ties count one half), and the number of pairs. A pair's
    order of cost is identified when its cheaper run reached the acquisition level."""
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    pairs = agree = 0.0
    for group in by_domain.values():
        for a in group:
            if a["censored"]:
                continue
            for b in group:
                if a["y"] < b["y"]:
                    pairs += 1
                    ka, kb = sign * a[key], sign * b[key]
                    agree += 1.0 if ka < kb else 0.5 if ka == kb else 0.0
    return (agree / pairs if pairs else None), int(pairs)


def clustered_line(x, y, groups) -> dict:
    """Ordinary least-squares line of y on x; the standard error of the slope is clustered by ``groups``."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    index = defaultdict(list)
    for i, g in enumerate(groups):
        index[g].append(i)
    meat = sum(np.outer(X[i].T @ e[i], X[i].T @ e[i]) for i in index.values())
    n, G = len(x), len(index)
    cov = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - 2)
    return dict(intercept=float(beta[0]), slope=float(beta[1]), se=float(math.sqrt(cov[1, 1])), n=n, clusters=G)


def size_trend(rows, params: dict) -> dict:
    """Line of the residuals from the single line (a, b) against log10 of the number of parameters."""
    x, P, y = _arrays(rows)
    residual = y - two_part(params["a"], params["b"], x, P)
    return clustered_line([math.log10(r["n_params"]) for r in rows], residual,
                          [(r["domain"], r["source"]) for r in rows])


def read_table(path) -> list[dict]:
    """Rows of a cost table (CSV) with a measured x: columns domain, source, x, P, cost, censored, n_params."""
    rows = []
    for r in csv.DictReader(open(path)):
        if r["x"] in ("", "None") or r["cost"] in ("", "None"):
            continue
        rows.append(dict(r, x=float(r["x"]), P=float(r["P"] or 0), y=math.log10(float(r["cost"])),
                         censored=r["censored"] == "True", n_params=float(r["n_params"])))
    return rows


def main(argv=None):
    rows = [r for r in read_table((argv or sys.argv[1:])[0]) if not r["censored"]]
    line, mixed = fit_line(rows), fit_mixed(rows)
    print(f"{line['n']} runs, {line['clusters']} source runs")
    print("single line: a = {a:.3f}, b = {b:.3f}".format(**line["params"]),
          f"(s.e. {line['se']['b']:.3f}), residual s.d. {line['residual_sd']:.3f} dex")
    for d in mixed["residual_sd"]:
        print(f"{d}: a = {mixed['params']['a@' + d]:.3f}, b = {mixed['params']['b@' + d]:.3f} "
              f"(s.e. {mixed['se']['b@' + d]:.3f})")
    trend = size_trend(rows, line["params"])
    print(f"size trend of the residuals: {trend['slope']:+.3f} dex per decade of parameters (s.e. {trend['se']:.3f})")
    return line, mixed, trend


if __name__ == "__main__":
    main()
