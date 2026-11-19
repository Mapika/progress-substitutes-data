"""Trajectories of competence against background loss: smoothing, crossings and offsets at matched background loss.

A trajectory is the series of evaluations of one run (training tokens, background loss, competence), as written by
``thinning.evaluate``.
"""

import numpy as np


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def running_mean(y, window: int) -> np.ndarray:
    """Centred running mean over ``window`` points; the window shrinks at the two ends."""
    y = np.asarray(y, dtype=float)
    half = window // 2
    return np.array([y[max(0, i - half):i + half + 1].mean() for i in range(len(y))])


def first_passage(x, y, x0):
    """y at the first place where the series x passes x0 (linear interpolation); None if x never reaches x0."""
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if min(a, b) <= x0 <= max(a, b) and a != b:
            return float(y[i] + (x0 - a) / (b - a) * (y[i + 1] - y[i]))
    return None


def onset(tokens, background_loss, competence, level: float):
    """(background loss, tokens) at which competence first rises through ``level``, interpolated linearly in
    competence between the two neighbouring evaluations; (None, None) if it never does."""
    q = np.asarray(competence, dtype=float)
    for i in range(1, len(q)):
        if q[i] >= level > q[i - 1]:
            w = (level - q[i - 1]) / (q[i] - q[i - 1])
            return (float(background_loss[i - 1] + w * (background_loss[i] - background_loss[i - 1])),
                    float(tokens[i - 1] + w * (tokens[i] - tokens[i - 1])))
    return None, None


def on_grid(background_loss, values, grid) -> np.ndarray:
    """``values`` (one row per evaluation; any trailing shape) interpolated linearly in background loss to
    ``grid``. Rows of the result outside the run's range of background loss are NaN."""
    loss = np.asarray(background_loss, dtype=float)
    grid = np.asarray(grid, dtype=float)
    order = np.argsort(loss)
    loss, values = loss[order], np.asarray(values, dtype=float)[order]
    out = np.full((len(grid),) + values.shape[1:], np.nan)
    inside = (grid >= loss[0]) & (grid <= loss[-1])
    i = np.clip(np.searchsorted(loss, grid[inside]) - 1, 0, len(loss) - 2)
    w = ((grid[inside] - loss[i]) / (loss[i + 1] - loss[i])).reshape(-1, *([1] * (values.ndim - 1)))
    out[inside] = (1 - w) * values[i] + w * values[i + 1]
    return out


def offsets(thinned, full, grid, eps: float = 1e-4) -> np.ndarray:
    """Offset of a thinned run from the fully supervised run on a grid of background loss:
    logit q_f(L) - logit q_1(L). ``thinned`` and ``full`` are (background loss, competence) pairs of arrays."""
    return on_grid(thinned[0], logit(thinned[1], eps), grid) - on_grid(full[0], logit(full[1], eps), grid)
