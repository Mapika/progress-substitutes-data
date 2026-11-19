"""Target thinning: which of a skill's next-token targets stay in the loss.

A skill is a set of label classes. A label array gives, for every position p of a stream, the class of the
prediction made at p (the target is the token at p + 1). Under target thinning with retained fraction f, the
target of a skill position stays in the loss iff hash(p) < f. The hash depends on the stream position only, so
the retained set is the same in every run that reads the stream, and the set retained at f contains the set
retained at any smaller fraction. Inputs are never changed.
"""

import numpy as np
import torch


def position_hash(pos: np.ndarray, salt: int = 0) -> np.ndarray:
    """Uniform value in [0, 1) for each stream position. ``salt = 0`` is a multiplicative hash; ``salt > 0``
    gives an independent hash (splitmix64 of the position offset by the salt)."""
    if salt == 0:
        return ((pos.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)).astype(np.float64) / 2**32
    with np.errstate(over="ignore"):
        z = pos.astype(np.uint64) + np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / 2**53


def skill_positions(labels, pos: np.ndarray, classes) -> np.ndarray:
    """Boolean array: the prediction at each position belongs to the skill."""
    if int(pos.max()) >= len(labels):
        raise RuntimeError(f"label array has {len(labels)} entries; position {int(pos.max())} is needed")
    return np.isin(np.asarray(labels[pos]), list(classes))


def dropped_targets(skill: np.ndarray, pos: np.ndarray, keep: float, salt: int = 0) -> np.ndarray:
    """Boolean array: skill targets removed from the loss at retained fraction ``keep``."""
    if keep >= 1.0:
        return np.zeros(len(pos), bool)
    return skill & (position_hash(pos, salt) >= keep)


def derangement(n: int, seed: int = 12345) -> np.ndarray:
    """A fixed permutation of 0..n-1 with no fixed point: one cycle through a seeded random order."""
    order = np.random.default_rng(seed).permutation(n)
    pi = np.empty(n, np.int64)
    pi[order] = np.roll(order, -1)
    return pi


def apply_permuted_rule(y: torch.Tensor, retained_skill: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """The permuted rule: at the retained skill positions, a target among the first ``len(perm)`` token ids is
    replaced by its image under ``perm`` and any other target is removed from the loss. Targets elsewhere are
    unchanged. ``y`` and ``retained_skill`` are flat; a new tensor is returned."""
    y = y.clone()
    mapped = (y >= 0) & (y < len(perm))
    y[retained_skill & ~mapped] = -100
    sel = retained_skill & mapped
    y[sel] = perm[y[sel]]
    return y
