"""Document filtering of web text: reference labels, classifier and thresholds, the filtered reader, term accuracy."""

import json

import numpy as np
import pytest
import torch

DF = pytest.importorskip("thinning.webtext.docfilter")
ROW, EOS, VOCAB, BIGRAMS = DF.ROW, DF.L.EOS, DF.L.VOCAB, DF.HASH_SIZE - DF.L.VOCAB


@pytest.fixture(scope="module")
def filtered(tmp_path_factory):
    """A pool of 300-token documents, the labelled ones with marker tokens, its index, classifiers and thresholds."""
    root, rng = tmp_path_factory.mktemp("docfilter"), np.random.default_rng(0)
    (root / "plan.json").write_text(json.dumps(dict(weights={"a": 0.6, "b": 0.4})))
    for j, part, source, n in ((0, "train", "a", 40), (1, "extra", "a", 20), (0, "train", "b", 30)):
        x, doc = rng.integers(1000, 3000, n * ROW).astype(np.uint16), np.arange(n * ROW) // 301
        lab, start = rng.choice([0, 0, 0, 1, 2, 4], doc[-1] + 1).astype(np.uint8), 301 * np.arange(doc[-1] + 1)
        marked = (lab[doc] > 0) & (rng.random(len(x)) < 0.3)
        x[marked], x[300::301] = np.uint16(5000) + lab[doc][marked], EOS
        (root / part).mkdir(exist_ok=True)
        x.tofile(root / part / f"{source}.bin")
        np.savez(root / f"{source}.{j}.npz", start=start, len=np.minimum(300, len(x) - start), labels=lab)
    with pytest.MonkeyPatch.context() as patch:   # the runs read the 16 rows of the data orders of seeds 0 and 1
        patch.setattr(DF, "HORIZON", 16)
        DF.score_pool(DF.Pool(root), root, DF.fit_classifiers(DF.Pool(root), root))
    return DF.Pool(root), root, json.loads((root / "thresholds.json").read_text())


def test_reference_labels_of_four_texts():
    texts = ("telescope galaxy supernova nebula quasar " * 12, "the cat was on the mat for that day " * 12,
             "use std::fmt;\nfn main() { let mut v = vec![1]; v.pop().unwrap(); }", "天文学是研究宇宙的科学。" * 20)
    bits = DF.reference_labels([DF.text_features(t) for t in texts])
    assert (bits & 7).tolist() == [1, 0, 2, 4] and bits[1] == 8


def test_classifier_thresholds_and_filtered_reader(filtered):
    pool, root, report = filtered
    files = [(si, tokens, np.load(root / f"{k}.{j}.npz"), np.load(root / f"{k}.{j}.scores.npy"))
             for si, k, j, tokens, _ in DF.pool_files(pool)]
    (_, tokens, z, scores), clf = files[0], np.load(root / "classifier.npz")
    x = tokens[z["start"][3]:][:z["len"][3]].tolist()
    ids = x + [VOCAB + ((a * 1000003 + b) * 2654435761 & 0x7FFFFFFFFFFF) % BIGRAMS for a, b in zip(x, x[1:])]
    assert scores[3, 1] == pytest.approx(float(clf["b"][1]) + float(clf["w"][1][ids].mean()), abs=1e-5)
    for r in report.values():                     # the markers separate the labelled documents; at most f remains
        assert r["average_precision"] > 0.9 and all(v["share"] <= float(f) for f, v in r["levels"].items())
    order, kept = pool.order(48, 5), {}
    reader = lambda f: DF.FilteredReader(pool, order, 4, 1, 3, {}, {}, None, filter_dir=root, shares=f)  # noqa: E731
    tau = [report[d]["levels"][f]["threshold"] for d, f in (("astro", "0.3"), ("rust", "0.03"))]
    for si, tokens, z, s in files:                # astronomy above its threshold and every Rust document are removed
        gone, keep = (s[:, 0] >= tau[0]) | (s[:, 1] >= tau[1]) | (z["labels"] & 2 > 0), np.ones(len(tokens), bool)
        for a, n in zip(z["start"][gone], z["len"][gone]):
            keep[a:a + n + 1] = False
        kept[si] = np.concatenate([kept.get(si, keep[:0]), keep])
    flat = lambda f: np.concatenate([f(s, r)[kept[s][r * ROW:(r + 1) * ROW]] for s, r in order])  # noqa: E731
    want, rd = (flat(pool.row), flat(lambda s, r: (s << 40) + r * ROW + np.arange(ROW))), reader("astro=0.3,rust=0")
    for i in (0, 1, 2, 4, 5):                     # rows 1-2 of each batch of 4; batch 3 is read on the way to batch 4
        assert all(np.array_equal(g, v[i * 4 * ROW:][:4 * ROW].reshape(4, ROW)[1:3]) for g, v in zip(rd.rows(i), want))
    n = [sum(map(len, reader(f"rust={f}").docs.values())) for f in ("1", "0.03", "0.01", "0")]
    assert 24 < rd.rows_read < 48 and n[0] == 0 < n[1] <= n[3] and 0.9 * n[3] <= n[2] <= n[3]


def test_term_accuracy_is_top1_accuracy_at_the_marked_targets():
    rng, model = np.random.default_rng(1), torch.nn.Module()
    rows, term = torch.from_numpy(rng.integers(0, 20, (10, 33))), torch.from_numpy(rng.random((10, 32)) < 0.3)
    model.forward = lambda input_ids: type("Out", (), dict(logits=torch.eye(20)[(input_ids + 1) % 20]))
    want = float(((rows[:, :-1] + 1) % 20 == rows[:, 1:])[term].float().mean())
    assert 0 < want < 1 and DF.term_accuracy(model, rows, term, "cpu", bs=4) == pytest.approx(want)
