"""Web-text stack of the 20M-7B models: labels, thinning masks, data order, training and returned targets."""

import json
import os
import re

import numpy as np
import pytest

from thinning.webtext import dolma, entities, returns
from thinning.webtext import labels as L

EOS, FILL = L.EOS, 30000                          # ids from FILL on have no text and no label
WORDS = " a country whose capital apples pears"


@pytest.fixture(scope="module")
def vocab():
    """Word-level stand-in for the tokenizer: a new string takes the next free id."""
    strings, ids = [""] * L.VOCAB, dict(zip(L.AUX_WORDS + (".", ","), L.AUX_IDS + (L.DOT, L.COMMA)))

    def encode(text):
        pieces = re.findall(r" ?\w+|\S", text)
        for s in pieces:
            strings[ids.setdefault(s, 1000 + len(ids))] = s
        return [ids[s] for s in pieces]

    encode("".join(L.AUX_WORDS) + " . , ( ) [ ] { }" + WORDS + "".join(f" {v}" for v in range(50)))
    L.Tables(L.Vocab(strings, encode))            # gives an id to every piece of every name
    return L.Vocab(strings, encode)


@pytest.fixture(scope="module")
def tables(vocab):
    return L.Tables(vocab)


@pytest.fixture(scope="module")
def pool_dir(tmp_path_factory, vocab):
    """Three sources of 50 training and 40 further rows of sentences that name a country and its capital."""
    root, rng = tmp_path_factory.mktemp("pool"), np.random.default_rng(1)
    (root / "plan.json").write_text(json.dumps(dict(weights={"news": 0.5, "wiki": 0.3, "code": 0.2})))
    for part, n in (("train", 50), ("extra", 40)):
        (root / part).mkdir()
        for source in ("news", "wiki", "code"):
            ids, units = [], entities.CAPITALS
            while len(ids) < n * L.ROW:
                k, (countries, capitals) = int(rng.integers(40)), units[rng.integers(len(units))]
                ids += [EOS] + vocab.encode(f" {countries[0].removeprefix('the ')} is a country ( whose capital is"
                                            f" {capitals[0]} ) , {k} apples {k + 1} pears .")
            np.array(ids[:n * L.ROW], np.uint16).tofile(root / part / f"{source}.bin")
    return root


def row(ids):
    return np.array(ids + list(range(FILL, FILL + L.ROW - len(ids))), np.uint16)


def test_structural_classes_on_hand_built_rows(vocab, tables):
    enc = vocab.encode
    x = row([EOS] + enc(" a ( country ) ] 3 apples 4 pears 5 apples 7 is"))
    lab, kind, dist = L.label_detail(x, tables)
    assert [(vocab.strings[v], c) for v, c in zip(x, lab) if c] == [
        (")", L.CLOSE), (" 4", L.SUCC), (" 5", L.SUCC), (" is", L.AGR)]
    assert (kind[4], dist[4]) == (0, 2)           # a round bracket, two tokens after its opener
    gram, aux = enc(WORDS)[:4], enc(" a country whose is")
    x = row([EOS] + gram + aux + list(range(FILL, FILL + 70)) + gram + aux + gram + [EOS] + gram)
    lab = L.label(x[None], tables)[0]
    # the second copies of the 4-grams, 78 tokens after the first; the auxiliary keeps its class; a new document
    assert np.flatnonzero(lab == L.COPY).tolist() == [82, 83, 84, 85, 90] and lab[86] == L.AGR


def test_capital_mentions_are_labelled_within_the_window(vocab, tables):
    enc = vocab.encode
    near = enc(" Argentina is a country whose capital is Buenos Aires")
    x = row([EOS] + near + [EOS] + enc(" France") + list(range(FILL, FILL + 130)) + enc(" Paris") + [EOS]
            + enc(" Paris is a capital . Vienna , Austria"))[None]
    cls, span = L.label_facts(x, tables)
    first = len(near) - 1                         # column of " Buenos"
    assert np.flatnonzero(cls[0]).tolist() == [first, first + 1] and (cls[0, first:first + 2] == L.CAP).all()
    assert span[0, first:first + 2].tolist() == [first, first] and (span[0, :first] == -1).all()
    lab, kept = np.zeros_like(cls), []
    for r in range(40):                           # a mention is kept or removed as a whole
        drop = L.thinned(lab, cls, span, L.stream_positions(np.array([2]), np.array([r])), {}, {L.CAP: 0.5})
        assert drop[0, first] == drop[0, first + 1] and drop.sum() == 2 * drop[0, first]
        kept.append(not drop[0, first])
    assert 0 < sum(kept) < 40
    x, rule = x.astype(np.int64), returns.AnswerRule(vocab, {"cap": "perm"})
    y = rule(x, x[:, 1:], cls, span, np.zeros(cls.shape, bool))
    m = rule.maps[L.CAP]
    g = m["group_of"][x[0, first]]                # the answer group of " Buenos"
    assert g >= 0 and y[0, first - 1] == m["first_token"][m["pi"][g]] != x[0, first] and y[0, first] == -100
    same = returns.AnswerRule(vocab, {"cap": "first"})(x, x[:, 1:], cls, span, np.zeros(cls.shape, bool))
    assert same[0, first - 1] == x[0, first] and same[0, first] == -100 and (y != x[0, 1:]).sum() == 2


def test_thinning_keeps_the_fraction_and_nests():
    lab = np.random.default_rng(0).integers(0, 5, (64, L.ROW)).astype(np.uint8)
    none, pos, agr = np.zeros(lab.shape, np.int32), L.stream_positions(np.full(64, 3), np.arange(64)), lab == L.AGR
    drop = {f: L.thinned(lab, none, none, pos, {L.AGR: f, L.COPY: 1.0}, {}) for f in (0.0, 0.1, 0.3, 1.0)}
    assert drop[0.0][agr].all() and not drop[1.0].any() and not drop[0.3][~agr].any()
    assert abs((~drop[0.1][agr]).mean() - 0.1) < 0.02 and abs((~drop[0.3][agr]).mean() - 0.3) < 0.02
    assert not (drop[0.3] & ~drop[0.1]).any()     # the targets retained at 0.1 are retained at 0.3


def test_derangement_has_no_fixed_point():
    p, again = returns.derangement(102), returns.derangement(102)
    assert sorted(p) == list(range(102)) and not (p == np.arange(102)).any() and np.array_equal(p, again)


def test_data_order_is_a_seeded_permutation_with_planned_counts(pool_dir):
    pool = dolma.Pool(pool_dir)
    order, counts = pool.order(160, 7), pool.counts(160)
    assert counts == {"code": 32, "news": 80, "wiki": 48} and sum(pool.counts(7).values()) == 7
    assert len(np.unique(order, axis=0)) == 160
    for i, k in enumerate(pool.keys):
        rows = order[order[:, 0] == i, 1]
        assert len(rows) == counts[k] and rows.max() < pool.n_rows[k] == 90
    assert np.array_equal(order, pool.order(160, 7)) and not np.array_equal(order, pool.order(160, 8))
    assert np.array_equal(pool.row(0, 50), np.fromfile(pool_dir / "extra" / "code.bin", np.uint16)[:L.ROW])
    with pytest.raises(ValueError):
        pool.counts(400)
    periodic = np.array([5, 6, 7] * 40 + list(range(100)), np.int64)
    assert dolma.instance_ok(np.asarray(pool.row(0, 0)).astype(np.int64)) and not dolma.instance_ok(periodic)


def write_eval_sets(path, vocab, tables, rows):
    """Small held-out and closing sets in the layout of ``evalsets build``."""
    enc, cls = vocab.encode, L.label(rows, tables)
    r, c = np.nonzero(cls[:, 1:] == L.AGR)
    seq, nums = enc(WORDS), [enc(f" {v}")[0] for v in range(8)]
    np.savez(path / "heldout.npz", rows=rows, cls=cls,
             agr=np.stack([r, c + 1, rows[r, c + 1], tables.partner[rows[r, c + 1]]], 1),
             copy_rows=np.array([[EOS] + seq + seq], np.uint16),
             copy_pos=np.array([(0, 1 + len(seq) + i) for i in range(3, len(seq))]),
             succ_rows=np.array([[EOS] + nums[1:6]], np.uint16),
             succ_pos=np.array([(0, k, *nums[k - 1:k + 3]) for k in (3, 4, 5)]))
    seqs = [enc(f" a {o} country whose capital {c}") for o, c in zip(L.OPENERS, L.CLOSERS)]
    np.savez(path / "closing.npz", seqs=np.array(seqs, np.uint16), kind=np.arange(3), nested=np.array([0, 1, 0], bool))


def test_train_then_return_writes_evaluation_rows(tmp_path, pool_dir, vocab, tables, monkeypatch):
    pytest.importorskip("transformers")
    from thinning.webtext import evalsets, train
    monkeypatch.setattr(L, "load_vocab", lambda path: vocab)
    write_eval_sets(tmp_path, vocab, tables, np.array(dolma.Pool(pool_dir).parts["wiki"][1][:2, :96]))
    common = ["--size", "2x32x2", "--batch", "2", "--micro", "1", "--data", str(pool_dir), "--eval", str(tmp_path),
              "--tokenizer", "unused", "--horizon-tokens", str(80 * 2 * L.ROW), "--cap-tokens", str(2 * 2 * L.ROW),
              "--device", "cpu", "--log-every", "1", "--seed", "1"]
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines()]  # noqa: E731
    train.main(common + ["--out", str(tmp_path / "source"), "--keep", "agr=0", "--fkeep", "cap=0",
                         "--save-every-log2", "1"])
    source = read(tmp_path / "source" / "eval.jsonl")
    measures = set(evalsets.KEYS) | {"competence_capitals", "competence_city_countries"}
    assert [r["step"] for r in source] == [0, 2] and set(source[0]) == measures | {"step", "tokens", "lr"}
    assert all(np.isfinite(r[k]) for r in source for k in measures)
    returns.main(common + ["--out", str(tmp_path / "return"), "--init-from", str(tmp_path / "source" / "step2"),
                           "--src-keep", "agr=0", "--src-fkeep", "cap=0", "--keep", "agr=0", "--fkeep", "cap=1",
                           "--fact-rule", "cap=perm", "--mom-steps", "2", "--prewarm-steps", "1"])
    rows = read(tmp_path / "return" / "eval.jsonl")
    assert [(r["step"], r["return_step"], r["data_block"]) for r in rows] == [(0, 0, 2), (1, 0, 3), (2, 1, 4)]
    assert rows[2]["return_tokens"] == 2 * L.ROW and 0 <= rows[2]["competence_capitals_permuted"] <= 1
    assert abs(rows[0]["loss"] - source[1]["loss"]) < 0.01      # the moment steps leave the weights unchanged
    assert all(np.isfinite(r["loss"]) for r in read(tmp_path / "return" / "train.jsonl"))


@pytest.mark.skipif("WEBTEXT_TOKENIZER" not in os.environ, reason="needs tokenizer.json of EleutherAI/pythia-14m")
def test_prompts_of_the_tokenizer():
    prompts = entities.build_prompts(L.load_vocab(os.environ["WEBTEXT_TOKENIZER"]).encode)
    assert [(len(prompts[r]["ids"]), len(prompts[r]["groups"])) for r in ("cap", "ctry")] == [(816, 102), (864, 54)]
