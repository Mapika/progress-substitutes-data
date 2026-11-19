"""Model, rows, thinning masks and the training, evaluation, return and removal entry points on a tiny corpus."""

import csv
import json

import numpy as np
import torch

from thinning import evaluate, masks, removal, returns, train
from thinning.data import TokenStream, read_rows, row_positions, write_stream
from thinning.domains import chess as C
from thinning.model import Decoder, build_optimizer, init_weights
from thinning.sizes import SIZES, get_size

SIZE = "3x64"


def test_sizes():
    assert SIZES["9.7M"].n_params == 9673024 and SIZES["2.2M"].n_heads == 3
    assert get_size(SIZE).n_layers == 3 and get_size(SIZE).d_model == 64


def test_position_hash_is_uniform_and_nested():
    pos = np.arange(200_000)
    for salt in (0, 1):
        h = masks.position_hash(pos, salt)
        assert abs((h < 0.1).mean() - 0.1) < 0.005 and h.min() >= 0 and h.max() < 1
    skill = np.ones(len(pos), bool)
    kept_small = ~masks.dropped_targets(skill, pos, 0.1)
    kept_large = ~masks.dropped_targets(skill, pos, 0.3)
    assert (kept_large | ~kept_small).all() and not masks.dropped_targets(skill, pos, 1.0).any()
    assert masks.dropped_targets(skill, pos, 0.0).all()
    assert not (masks.position_hash(pos, 1) == masks.position_hash(pos, 2)).all()


def test_permuted_rule():
    perm = masks.derangement(1968)
    assert sorted(perm) == list(range(1968)) and (perm != np.arange(1968)).all()
    y = torch.tensor([5, 1968, 7, -100])
    out = masks.apply_permuted_rule(y, torch.tensor([True, True, False, True]), torch.from_numpy(perm))
    assert out.tolist() == [int(perm[5]), -100, 7, -100]


def test_rows_follow_the_stream(tmp_path):
    tokens = np.arange(1, 41) % 7 + 1
    tokens[[9, 25]] = 0
    write_stream(tmp_path, [tokens[:22], tokens[22:]], vocab_size=8, sep_id=0)
    stream = TokenStream(tmp_path)
    x, y = read_rows(stream, 1, 2, 9)
    assert x[0].tolist() == tokens[10:19].tolist() and x[1].tolist() == tokens[20:29].tolist()
    assert y[0].tolist() == [-100] * 9
    assert y[1].tolist() == [-100] * 5 + tokens[26:30].tolist()
    assert read_rows(stream, 0, 1, 9)[1][0].tolist() == [-100] * 9
    assert row_positions(1, 2, 9).tolist() == list(range(10, 19)) + list(range(20, 29))


def test_attention_stays_inside_a_document():
    model = Decoder(get_size(SIZE), 50, sep_id=0)
    init_weights(model, seed=0)
    x = torch.randint(1, 50, (1, 24))
    x[0, 0] = x[0, 10] = 0
    base = model(x)
    later = x.clone()
    later[0, 20] = (later[0, 20] % 48) + 1
    assert torch.allclose(model(later)[0, :20], base[0, :20], atol=1e-6)
    earlier = x.clone()
    earlier[0, 3] = (earlier[0, 3] % 48) + 1
    assert torch.allclose(model(earlier)[0, 10:], base[0, 10:], atol=1e-6)
    assert not torch.allclose(model(earlier)[0, 3:10], base[0, 3:10], atol=1e-6)
    groups = build_optimizer(model, 0.004, 0.1).param_groups
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_thinning_removes_only_skill_targets(chess_corpus):
    stream = TokenStream(chess_corpus / "train")
    labels = np.load(chess_corpus / "labels.npy")
    session = train.Session(None, stream, 64, labels, (1,))
    x, y, skill = session.rows(0, 16)
    x0, y0, _ = session.rows(0, 16, keep=0.0)
    xf, yf, _ = session.rows(0, 16, keep=0.5)
    assert skill.any() and (x == x0).all() and (x == xf).all()
    valid = y != -100
    assert (y0[valid & ~skill] == y[valid & ~skill]).all() and (y0[skill] == -100).all()
    assert 0 < (yf[skill & valid] != -100).float().mean() < 1
    board_in_check = labels[row_positions(0, 16, 64)].reshape(16, 64) == 1
    assert (board_in_check == skill.numpy()).all()


def test_train_evaluate_return_and_removal(chess_corpus, tmp_path):
    data, labels, held_out = str(chess_corpus / "train"), str(chess_corpus / "labels.npy"), str(chess_corpus / "eval")
    n_params = get_size(SIZE).n_params
    per_step = 4 * 64 / n_params

    def run(name, keep):
        train.main(["--data", data, "--labels", labels, "--skill-classes", "1", "--keep", str(keep), "--size", SIZE,
                    "--seed", "0", "--tokens-per-param", str(6 * per_step),
                    "--snapshots", f"{3 * per_step},{6 * per_step}",
                    "--seq-len", "64", "--batch-seqs", "4", "--warmup-tokens-per-param", str(2 * per_step),
                    "--ramp-tokens-per-param", "0", "--out", str(tmp_path / name), "--device", "cpu"])
        return sorted((tmp_path / name).glob("snap_*.pt"))

    never, full = run("never", 0.0), run("full", 1.0)
    assert len(never) == 2 and len(full) == 2
    model, config, tokens, next_row = train.load_snapshot(never[-1])
    assert tokens == 6 * 4 * 64 and next_row == 24 and config["keep"] == 0.0
    metrics = [json.loads(line) for line in (tmp_path / "never" / "metrics.jsonl").read_text().splitlines()]
    assert np.isfinite(metrics[-1]["train_loss"]) and metrics[-1]["step"] == 6

    rows = evaluate.main(["--domain", "chess", "--run", str(tmp_path / "never"), "--eval-data", held_out,
                          "--out", str(tmp_path / "never.csv"), "--n-eval", "20", "--eval-offset", "0",
                          "--per-position", str(tmp_path / "positions"), "--output-classes", "--device", "cpu"])
    assert len(rows) == 2 and 0 <= rows[0]["competence"] <= 1 and rows[0]["background_loss"] > 0
    assert len(list(csv.DictReader(open(tmp_path / "never.csv")))) == 2
    saved = np.load(tmp_path / "positions" / f"{never[-1].stem}.npz")
    pooled = saved["skill_mass"].sum() / (saved["skill_mass"].sum() + saved["other_mass"].sum())
    assert abs(pooled - rows[1]["competence"]) < 1e-6

    common = ["--data", data, "--labels", labels, "--eval-data", held_out, "--batch-seqs", "4", "--n-eval", "20",
              "--eval-offset", "0", "--device", "cpu"]
    short = ["--moment-steps", "2", "--max-tokens", "1024"]
    out = returns.main(["--preset", "chess", "--source", str(never[-1]), "--share", "1", "--warmup-before", "2",
                        "--out", str(tmp_path / "return.json")] + common + short)
    assert [r["step"] for r in out] == [0, 1, 2, 3, 4] and out[0]["train_loss"] is None
    saved = json.loads((tmp_path / "return.json").read_text())
    assert saved["config"]["first_returned_row"] == 24 + 2 * 4 and saved["config"]["source_keep"] == 0.0
    out = returns.main(["--preset", "chess-warmup-at-return", "--source", str(never[-1]), "--share", "0.3",
                        "--rule", "permuted", "--out", str(tmp_path / "permuted.json")] + common + short)
    assert "competence_permuted" in out[0]
    out = returns.main(["--preset", "chess-warmup-at-return", "--size", SIZE, "--seed", "0", "--seq-len", "64",
                        "--out", str(tmp_path / "init.json")] + common + short)
    assert out[-1]["step"] == 4 and out[-1]["train_loss"] < out[1]["train_loss"] + 1.0

    for route, extra in (("withdrawn", ["--withdraw-tokens", "512"]),
                         ("gradient-ascent", ["--removal-max-steps", "4"]),
                         ("misdirection", ["--removal-max-steps", "4"]), ("none", [])):
        source = never[-1] if route == "none" else full[-1]
        out = removal.main(["--route", route, "--source", str(source), "--max-rows", "2", "--warmup-before", "2",
                            "--shares", "1,0", "--budgets", "512,256", "--out", str(tmp_path / f"{route}.json")]
                           + common + extra)
        saved = json.loads((tmp_path / f"{route}.json").read_text())
        assert set(out) == {"1", "0"} and out["1"][-1]["tokens"] == 512 and out["0"][-1]["tokens"] == 256
        assert out["1"][0]["competence"] == out["0"][0]["competence"]
        assert len(saved["before_return"]) == 2
        if route in ("gradient-ascent", "misdirection"):
            assert saved["removal_stop"] in ("steps", "background loss", "plateau") and saved["kept"] is not None
            limit = saved["removal"][0]["background_loss"] + 0.05
            assert saved["kept"]["background_loss"] <= limit
