"""Model sizes: width and depth grow together; every size shares the head width and the MLP ratio.

A size is named by its number of non-embedding parameters. ``batch_seqs`` is the full batch, in sequences of
2,048 tokens, of the warm-up--stable recipe. Any other shape can be requested as ``"<layers>x<width>"``
(for example ``"2x64"``), which is how the small-scale examples and the tests build their models.
"""

from dataclasses import dataclass

HEAD_DIM = 64


@dataclass(frozen=True)
class Size:
    name: str
    n_layers: int
    d_model: int
    batch_seqs: int

    @property
    def n_heads(self) -> int:
        return self.d_model // HEAD_DIM

    @property
    def d_ff(self) -> int:
        """SwiGLU hidden width: 8/3 of the model width, rounded to a multiple of 64."""
        return max(64, round(self.d_model * 8 / 3 / 64) * 64)

    @property
    def n_params(self) -> int:
        """Non-embedding parameters: four attention matrices, three MLP matrices, two norm gains and the
        query/key norm gains per block, plus the final norm."""
        d = self.d_model
        per_block = 4 * d * d + 3 * d * self.d_ff + 2 * d + 2 * HEAD_DIM
        return self.n_layers * per_block + d


SIZES = {s.name: s for s in (
    Size("0.76M", 4, 128, 32),
    Size("2.2M", 5, 192, 32),
    Size("4.8M", 6, 256, 64),
    Size("9.7M", 8, 320, 96),
    Size("24M", 10, 448, 128),
    Size("64M", 13, 640, 160),
    Size("163M", 17, 896, 224),
    Size("431M", 22, 1280, 252),
    Size("1.28B", 29, 1920, 64),
    Size("3.0B", 38, 2560, 64),
)}
BASE = SIZES["2.2M"]  # anchor of the CompleteP parameterization


def get_size(name: str) -> Size:
    """A named size, or a custom shape written ``"<layers>x<width>"`` with a width that is a multiple of 64."""
    if name in SIZES:
        return SIZES[name]
    n_layers, d_model = (int(v) for v in name.split("x"))
    if d_model % HEAD_DIM:
        raise ValueError(f"width must be a multiple of {HEAD_DIM}, got {d_model}")
    return Size(name, n_layers, d_model, 16)
