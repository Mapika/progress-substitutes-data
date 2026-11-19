"""Analysis functions on synthetic numbers with known answers."""

import math

import numpy as np

from thinning.analysis import cost, cost_relation, thinning_relation, trajectories


def test_onset_and_offsets_on_linear_trajectories():
    tokens, loss = np.arange(1, 11) * 100.0, np.linspace(3.0, 2.1, 10)
    q = np.linspace(0.05, 0.95, 10)
    at_loss, at_tokens = trajectories.onset(tokens, loss, q, 0.5)
    assert math.isclose(at_loss, 2.55) and math.isclose(at_tokens, 550.0)
    assert trajectories.onset(tokens, loss, q, 0.99) == (None, None)
    grid = np.array([2.0, 2.4, 2.8, 3.2])
    full = (loss, 1 / (1 + np.exp(-(4 - 2 * loss))))
    thinned = (loss, 1 / (1 + np.exp(-(4 - 2 * loss - 0.7))))
    d = trajectories.offsets(thinned, full, grid)
    assert np.isnan(d[0]) and np.isnan(d[3]) and np.allclose(d[1:3], -0.7)


def test_fit_relation_recovers_kappa_and_tau():
    f = np.array([0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0])
    kappa, tau = thinning_relation.fit_relation(f, thinning_relation.relation(f, 0.6, 0.02))
    assert math.isclose(kappa, 0.6, rel_tol=1e-4) and math.isclose(tau, 0.02, rel_tol=1e-3)
    assert math.isclose(thinning_relation.fit_relation(f, thinning_relation.relation(f, 1.0, 0.3), kappa=1.0)[1],
                        0.3, rel_tol=1e-3)


def test_derived_offset_for_identical_positions():
    skill, other = np.full(50, 0.6), np.full(50, 0.2)
    tau = 0.05
    never = math.log(3.0) + math.log(tau / (1 + tau))
    assert math.isclose(thinning_relation.corpus_support(skill, other, never), tau, rel_tol=1e-6)
    offset, _ = thinning_relation.derived_offset(0.1, skill, other, never)
    assert math.isclose(offset, math.log((0.1 + tau) / (1 + tau)), rel_tol=1e-6)
    assert math.isclose(thinning_relation.pooling_sensitivity(skill, other), 1.0)
    assert thinning_relation.derived_offset(0.1, skill, other, math.log(3.0) + 0.1) == (0.0, float("inf"))


def test_pooling_lowers_the_response_for_unequal_positions():
    rng = np.random.default_rng(0)
    z = rng.normal(1.0, 2.0, 2000)
    skill, other = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(z))
    kappa = thinning_relation.pooling_sensitivity(skill, other)
    assert 0.2 < kappa < 0.9
    small = thinning_relation.pooled_log_odds(skill, other, -0.01) - thinning_relation.pooled_log_odds(skill, other)
    assert math.isclose(small / -0.01, kappa, rel_tol=0.02)


def test_logit_distance_and_acquisition_loss_on_a_linear_reference():
    loss = np.linspace(3.0, 1.5, 31)
    score = 1 / (1 + np.exp(-(6.0 - 3.0 * loss)))
    level = 0.6
    x = cost.logit_distance(level, loss, score, 2.2, window=1)
    assert math.isclose(x, math.log(1.5) - (6.0 - 3.0 * 2.2), rel_tol=1e-6)
    assert math.isclose(cost.acquisition_loss(loss, score, level, window=1), (6.0 - math.log(1.5)) / 3.0, rel_tol=1e-4)
    assert cost.logit_distance(level, loss, score, 3.5) is None
    assert math.isclose(cost.smoothed_crossing(loss, score, level, window=1), (6.0 - math.log(1.5)) / 3.0, rel_tol=1e-3)


def test_acquisition_tokens_and_censoring():
    tokens = [0, 100, 200, 300]
    assert cost.acquisition_tokens(tokens, [0.1, 0.3, 0.7, 0.9], 0.5) == (150.0, False)
    assert cost.acquisition_tokens(tokens, [0.6, 0.3, 0.7, 0.9], 0.5) == (0.0, False)
    assert cost.acquisition_tokens(tokens, [0.1, 0.2, 0.3, 0.4], 0.5) == (300.0, True)


def test_loss_curve_and_progress_term():
    tokens = np.logspace(6, 9, 40)
    curve = cost.LossCurve(tokens, 1.5 + 40.0 * tokens ** -0.3)
    inside, beyond = 1.5 + 40.0 * 3e7 ** -0.3, 1.5 + 40.0 * 4e9 ** -0.3
    assert math.isclose(curve.tokens_at(inside), 3e7, rel_tol=0.01)
    assert math.isclose(curve.tokens_at(beyond), 4e9, rel_tol=0.05)
    assert cost.progress_term(curve, inside, inside + 0.1, 1e8) == 0.0
    start = 1.5 + 40.0 * 1e7 ** -0.3
    assert math.isclose(cost.progress_term(curve, start, inside, 1e8), 0.2, rel_tol=0.02)


def test_acquisition_loss_curves():
    s = np.array([1, 0.3, 0.1, 0.03, 1, 0.3, 0.1])
    fit = cost.fit_share_curve(s, 1.4 + 0.8 * s ** 0.3)
    assert math.isclose(fit["a"], 1.4, abs_tol=1e-6) and math.isclose(fit["c"], 0.3, abs_tol=1e-9) and fit["rms"] < 1e-9
    n = np.array([1e7, 2.4e7, 6.4e7, 1.6e8])
    fit = cost.fit_size_curve(n, 5.0 - 0.14 * np.log10(n))
    assert math.isclose(fit["b"], -0.14, rel_tol=1e-6) and fit["rms"] < 1e-9


def synthetic_rows(noise: float = 0.0):
    rng = np.random.default_rng(1)
    params = {"chess": (-0.5, 1.1), "smiles": (-0.1, 0.7)}
    rows = []
    for domain, (a, b) in params.items():
        for source in range(6):
            shift = rng.normal(0, noise)
            for x in np.linspace(-2.5, 0.8, 9):
                P = 0.3 * max(x, 0.0)
                y = float(cost_relation.two_part(a, b, x, P)) + shift + rng.normal(0, noise)
                rows.append(dict(domain=domain, source=f"run{source}", x=float(x), P=P, y=y, censored=False,
                                 n_params=10.0 ** (7 + source % 3)))
    return rows, params


def test_cost_relation_fits_recover_the_parameters():
    rows, params = synthetic_rows()
    chess_rows = [r for r in rows if r["domain"] == "chess"]
    line = cost_relation.fit_line(chess_rows)
    assert math.isclose(line["params"]["a"], -0.5, abs_tol=1e-5)
    assert math.isclose(line["params"]["b"], 1.1, abs_tol=1e-5)
    assert line["clusters"] == 6 and line["residual_sd"] < 1e-6
    assert abs(cost_relation.size_trend(chess_rows, line["params"])["slope"]) < 1e-6
    noisy, _ = synthetic_rows(noise=0.05)
    mixed = cost_relation.fit_mixed(noisy)
    for domain, (a, b) in params.items():
        assert abs(mixed["params"][f"a@{domain}"] - a) < 0.1 and abs(mixed["params"][f"b@{domain}"] - b) < 0.1
    assert 0.02 < mixed["source_sd"] < 0.12


def test_ordering_counts_concordant_pairs():
    rows = [dict(domain="chess", x=x, y=y, censored=c)
            for x, y, c in ((-2.0, -2.0, False), (-1.0, -1.0, False), (0.0, -0.5, True))]
    assert cost_relation.ordering(rows, "x") == (1.0, 3)
    assert cost_relation.ordering(rows, "x", sign=-1) == (0.0, 3)
