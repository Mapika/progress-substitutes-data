"""Acquisition cost of a return run and its two predictors, the logit distance and the progress term.

  cost       tokens from the return to the acquisition level, divided by the tokens a same-size model needs
             from initialization.
  reference  the same-size model trained from initialization with the returned share s of the skill's targets,
             as a trajectory of (background loss, score) in training order.
  x          logit distance: log odds of the acquisition level minus the reference's log odds at the background
             loss L0 at which the targets return. Negative x is a surplus.
  L*(s)      acquisition loss: the background loss at which the reference reaches the acquisition level. It is
             read either on one curve per domain fitted across reference runs (``fit_share_curve`` over shares
             and sizes, ``fit_size_curve`` over sizes; the paper's cost relation uses these curves) or on the
             same-size reference itself (``acquisition_loss``).
  P          progress term: tokens the source run needs to bring its own background loss from L0 to L*(s),
             divided by the tokens from initialization; zero when L0 <= L*(s).
"""

import math

import numpy as np
from scipy.optimize import curve_fit

from thinning.analysis.trajectories import first_passage, logit, running_mean


def acquisition_tokens(tokens, score, level: float) -> tuple[float, bool]:
    """(tokens after the return at which the score first reaches ``level``, censored). Linear interpolation
    between evaluations; 0 if the first evaluation is at the level; a run that never reaches it is censored
    at its last evaluation."""
    if score[0] >= level:
        return 0.0, False
    for i in range(1, len(score)):
        if score[i] >= level:
            a, b = score[i - 1], score[i]
            w = (level - a) / (b - a) if b != a else 1.0
            return float(tokens[i - 1] + w * (tokens[i] - tokens[i - 1])), False
    return float(tokens[-1]), True


def smoothed_crossing(x, score, level: float, window: int = 5, eps: float = 1e-4):
    """x at the first upward crossing of ``level`` by the score smoothed in log odds over ``window`` evaluations
    (the rule for the 7B runs, whose evaluations are dense and noisy); None if there is none. With x the
    background loss of a run from initialization this gives its acquisition loss."""
    smooth = 1 / (1 + np.exp(-running_mean(logit(score, eps), window)))
    for i in range(1, len(smooth)):
        if smooth[i - 1] < level <= smooth[i]:
            w = (level - smooth[i - 1]) / (smooth[i] - smooth[i - 1])
            return float(x[i - 1] + w * (x[i] - x[i - 1]))
    return None


def reference_log_odds(ref_loss, ref_score, loss: float, window: int = 3):
    """Log odds of the reference at a background loss: the log odds are smoothed by a centred running mean over
    ``window`` checkpoints in training order (the losses are not smoothed) and read at the first passage of the
    loss. None outside the reference's range of background loss."""
    if not min(ref_loss) <= loss <= max(ref_loss):
        return None
    return first_passage(ref_loss, running_mean(logit(ref_score), window), loss)


def logit_distance(level: float, ref_loss, ref_score, loss: float, window: int = 3):
    at = reference_log_odds(ref_loss, ref_score, loss, window)
    return None if at is None else float(logit(level)) - at


def acquisition_loss(ref_loss, ref_score, level: float, window: int = 3):
    """Background loss at which the reference's smoothed log odds first reach those of the level."""
    y, target = running_mean(logit(ref_score), window), float(logit(level))
    for i in range(1, len(y)):
        if y[i - 1] < target <= y[i]:
            w = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(ref_loss[i - 1] + w * (ref_loss[i] - ref_loss[i - 1]))
    return None


def fit_share_curve(shares, losses) -> dict:
    """Acquisition loss against returned share, L*(s) = a + b * s^c, by least squares over reference runs of
    several shares and sizes: c on a grid of step 0.002, a and b linear. Returns a, b, c and the r.m.s. residual."""
    s, y = np.asarray(shares, dtype=float), np.asarray(losses, dtype=float)
    best = None
    for c in np.arange(0.02, 1.6, 0.002):
        X = np.column_stack([np.ones_like(s), s ** c])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((y - X @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(beta[0]), float(beta[1]), float(c))
    return dict(a=best[1], b=best[2], c=best[3], rms=math.sqrt(best[0] / len(y)))


def fit_size_curve(n_params, losses) -> dict:
    """Acquisition loss against model size, L*(N) = a + b * log10(N), by least squares over reference runs of
    several sizes. Returns a, b and the r.m.s. residual."""
    x, y = np.log10(np.asarray(n_params, dtype=float)), np.asarray(losses, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return dict(a=float(a), b=float(b), rms=float(np.sqrt(np.mean((a + b * x - y) ** 2))))


class LossCurve:
    """Background loss of a run against training tokens, inverted to tokens(loss). Within the run: interpolation
    of log tokens in the running minimum of the loss. Below its lowest loss: a power law
    loss = E + A * tokens^-a fitted to the last decade of tokens."""

    def __init__(self, tokens, loss):
        order = np.argsort(tokens)
        self.tokens = np.asarray(tokens, dtype=float)[order]
        self.loss = np.minimum.accumulate(np.asarray(loss, dtype=float)[order])
        tail = self.tokens >= self.tokens[-1] / 10
        if tail.sum() < 4:
            tail = np.arange(len(self.tokens)) >= len(self.tokens) // 2
        try:
            self.power_law, _ = curve_fit(lambda t, e, a, k: e + a * np.power(t, -k), self.tokens[tail],
                                          self.loss[tail], p0=(self.loss[-1] * 0.8, 5.0, 0.3), maxfev=20000,
                                          bounds=([0, 0, 0.01], [self.loss[-1], 1e6, 3.0]))
        except (RuntimeError, ValueError):
            self.power_law = None

    def tokens_at(self, loss: float) -> float:
        """Tokens at which the run first reaches ``loss`` (infinite if the power law never does)."""
        if loss >= self.loss[0]:
            return float(self.tokens[0])
        if loss >= self.loss[-1]:
            i = int(np.argmax(self.loss <= loss))
            l1, l2, t1, t2 = self.loss[i - 1], self.loss[i], self.tokens[i - 1], self.tokens[i]
            w = (l1 - loss) / (l1 - l2) if l1 != l2 else 1.0
            return float(math.exp(math.log(t1) + w * (math.log(t2) - math.log(t1))))
        if self.power_law is None or loss <= self.power_law[0] + 1e-6:
            return float("inf")
        e, a, k = self.power_law
        return float(((loss - e) / a) ** (-1 / k))


def extend_with_training_loss(tokens, loss, train_tokens, train_loss):
    """Extend a run's evaluated background loss past its last snapshot with its training loss: the training
    loss (running mean over 5 records) is mapped linearly onto the evaluation scale at the snapshots. Needs at
    least 3 snapshots and more than 10 training records; otherwise the snapshots are returned unchanged."""
    tokens, loss = list(tokens), list(loss)
    train_tokens = np.asarray(train_tokens, dtype=float)
    if len(tokens) < 3 or len(train_tokens) <= 10:
        return tokens, loss
    smooth = np.convolve(train_loss, np.ones(5) / 5, mode="same")
    slope, intercept = np.polyfit(np.interp(tokens, train_tokens, smooth), loss, 1)
    later = [(t, intercept + slope * c) for t, c in zip(train_tokens[2:-2], smooth[2:-2]) if t > max(tokens)]
    return tokens + [t for t, _ in later], loss + [c for _, c in later]


def progress_term(source: LossCurve, loss_at_return: float, acquisition_loss_: float, tokens_from_init: float):
    if loss_at_return <= acquisition_loss_:
        return 0.0
    tokens = source.tokens_at(acquisition_loss_) - source.tokens_at(loss_at_return)
    return max(tokens, 0.0) / tokens_from_init if math.isfinite(tokens) else None


def cost_row(return_tokens, return_score, loss_at_return: float, level: float, reference, source: LossCurve,
             tokens_from_init: float, step_tokens: float, window: int = 3) -> dict:
    """One row of the cost table. ``reference`` is (background loss, score) in training order. A run already at
    the level at its first evaluation is charged one optimizer step."""
    tokens, censored = acquisition_tokens(return_tokens, return_score, level)
    star = acquisition_loss(*reference, level, window)
    return dict(x=logit_distance(level, *reference, loss_at_return, window), acquisition_loss=star,
                P=None if star is None else progress_term(source, loss_at_return, star, tokens_from_init),
                cost=(tokens or step_tokens) / tokens_from_init, censored=censored, tokens=tokens,
                background_loss=loss_at_return)
