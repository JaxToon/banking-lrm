"""SASRec Full: architecture, masking, loss, training, determinism, checkpoints."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from bri.seq import runtime as R  # noqa: E402
from bri.seq.data import SeqCases, build_seq_cases  # noqa: E402
from bri.seq.sasrec import SASRec, SASRecConfig, count_parameters, multi_positive_loss  # noqa: E402
from bri.seq.train import (TrainConfig, fit, load_checkpoint, per_case, save_checkpoint, score,  # noqa: E402
                           checkpoint_metadata, state_dict_sha256)

HAS_CUDA = R.cuda_usable()[0]
VOCAB = ["SVC_A", "SVC_B", "SVC_C", "SVC_D"]


def model(max_len=6, **kw):
    torch.manual_seed(0)
    return SASRec(SASRecConfig(n_items=len(VOCAB), max_len=max_len, dim=16, heads=2, blocks=2, dropout=0.0, **kw)).eval()


def batch(rows, max_len=6, max_tie=1):
    """rows: list of histories; each history is a list of steps; a step is an int or tuple of ints (1-based)."""
    out = torch.zeros(len(rows), max_len, max_tie, dtype=torch.long)
    for i, hist in enumerate(rows):
        off = max_len - len(hist)
        for p, step in enumerate(hist):
            step = step if isinstance(step, tuple) else (step,)
            out[i, off + p, :len(step)] = torch.tensor(step)
    return out


def synthetic_cases(n, split, seed, max_len=6, noise=0.0):
    """Histories of random items; the target is the item after the LAST step (cyclic)."""
    rng = np.random.default_rng(seed)
    items = np.zeros((n, max_len, 1), dtype=np.int64)
    targets = np.zeros((n, len(VOCAB)), dtype=np.float32)
    lengths = rng.integers(1, max_len + 1, size=n)
    for i, L in enumerate(lengths):
        hist = rng.integers(1, len(VOCAB) + 1, size=L)
        items[i, max_len - L:, 0] = hist
        nxt = hist[-1] % len(VOCAB) if rng.random() >= noise else rng.integers(0, len(VOCAB))
        targets[i, nxt] = 1.0
    cases = pd.DataFrame({"case_id": [f"SYN_NE_{split}_nbs_C_{i:05d}" for i in range(n)], "task": "nbs",
                          "split": split, "customer_token": [f"C{i % 50}" for i in range(n)]})
    z = np.zeros((n, max_len), dtype=np.int8)
    return SeqCases("nbs", split, VOCAB, max_len, cases, items, z, lengths, lengths, targets)


TIEBREAK = np.array([4.0, 3.0, 2.0, 1.0])


# --- architecture --------------------------------------------------------------------

def test_forward_shape_and_finite_scores():
    m = model()
    s = m(batch([[1, 2], [3], [4, 1, 2, 3, 1, 2]]))
    assert s.shape == (3, len(VOCAB)) and torch.isfinite(s).all()
    assert count_parameters(m) > 0


def test_invalid_architectures_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        SASRecConfig(n_items=4, max_len=10, dim=30, heads=4)
    with pytest.raises(ValueError):
        SASRecConfig(n_items=4, max_len=10, variant="nope")
    with pytest.raises(ValueError):
        model()(torch.zeros(1, 5, 1, dtype=torch.long))  # wrong sequence length


def test_attention_is_causal():
    """Hidden state at position i never depends on positions after i."""
    m = model()
    a, valid = m.encode(batch([[1, 2, 3, 4, 1, 2]]))
    b, _ = m.encode(batch([[1, 2, 3, 2, 4, 4]]))  # changes positions 3..5 only
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-6)
    assert not torch.allclose(a[0, 3:], b[0, 3:])


def test_padding_is_masked():
    """Left padding never influences the representation, whatever the padded positions' embeddings are."""
    m = model()
    x = batch([[2, 3]])
    before = m(x).clone()
    with torch.no_grad():
        m.pos_emb.weight[:4] = torch.randn(4, 16) * 100  # positions 0..3 are padding here
    assert torch.allclose(before, m(x), atol=1e-5)


def test_scores_do_not_depend_on_batch_composition_or_tie_slot_padding():
    m = model()
    alone = m(batch([[1, 3]]))
    together = m(batch([[1, 3], [2, 2, 4, 1, 3, 2]], max_tie=3))
    assert torch.allclose(alone[0], together[0], atol=1e-6)


def test_tied_items_are_an_unordered_mean():
    m = model()
    assert torch.allclose(m(batch([[(1, 3), 2]], max_tie=2)), m(batch([[(3, 1), 2]], max_tie=2)), atol=1e-6)
    emb, valid = m.step_embeddings(batch([[(1, 3)]], max_tie=2))
    expect = (m.item_emb.weight[1] + m.item_emb.weight[3]) / 2
    assert torch.allclose(emb[0, -1], expect) and valid[0].sum() == 1


def test_empty_history_gets_zero_scores_so_the_shared_tiebreak_decides():
    s = model()(batch([[]]))
    assert torch.equal(s, torch.zeros_like(s))


def test_scoring_reads_only_inputs_never_targets():
    m = model()
    cases = synthetic_cases(20, "val", 1)
    a = score(m, cases, "cpu")
    cases.targets[:] = 0
    cases.targets[:, 0] = 1
    assert np.array_equal(a, score(m, cases, "cpu"))


# --- loss ----------------------------------------------------------------------------

def test_multi_positive_loss_matches_the_formula():
    s = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    p = torch.softmax(s, dim=1)[0]
    two = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert math.isclose(float(multi_positive_loss(s, two)), -math.log(float(p[1] + p[2])), rel_tol=1e-6)
    one = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ce = torch.nn.functional.cross_entropy(s, torch.tensor([0]))
    assert math.isclose(float(multi_positive_loss(s, one)), float(ce), rel_tol=1e-6)
    assert math.isclose(float(multi_positive_loss(torch.zeros(1, 4), two)), math.log(2), rel_tol=1e-6)
    assert float(multi_positive_loss(s, torch.ones(1, 4))) == pytest.approx(0.0, abs=1e-6)


def test_loss_is_finite_with_extreme_scores_and_rejects_cases_without_positives():
    s = torch.tensor([[1e4, -1e4, 0.0, 0.0]], requires_grad=True)
    loss = multi_positive_loss(s, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    with pytest.raises(ValueError):
        multi_positive_loss(s, torch.zeros(1, 4))


# --- training --------------------------------------------------------------------------

CFG = SASRecConfig(n_items=4, max_len=6, dim=16, heads=2, blocks=1, dropout=0.1)
TCFG = TrainConfig(lr=5e-3, batch_size=32, max_epochs=15, patience=5, seed=3)


@pytest.fixture(scope="module")
def trained():
    train, val = synthetic_cases(600, "train", 1), synthetic_cases(200, "val", 2)
    return fit(CFG, TCFG, train, val, TIEBREAK, "cpu"), train, val


def test_cpu_training_learns_the_transition_and_selects_on_validation(trained):
    res, train, val = trained
    assert res.history[0]["train_loss"] > res.history[-1]["train_loss"]
    assert res.best_val > 0.9  # target = function of the last step: learnable
    assert res.best_val == max(h["val_ndcg@3"] for h in res.history)
    assert res.history[res.best_epoch - 1]["val_ndcg@3"] == res.best_val
    pc = per_case(val, score(res.model, val, "cpu"), TIEBREAK, 3)
    assert round(float(pc["ndcg"].mean()), 6) == res.best_val  # restored best weights
    rt = res.runtime
    assert rt["parameters"] == count_parameters(res.model) and rt["train_seconds"] > 0
    assert rt["peak_gpu_memory_mib"] is None


def test_training_is_deterministic_on_cpu(trained):
    res, train, val = trained
    again = fit(CFG, TCFG, train, val, TIEBREAK, "cpu")
    assert state_dict_sha256(again.model) == state_dict_sha256(res.model)
    assert again.history == res.history


def test_early_stopping_stops_after_patience():
    train, val = synthetic_cases(200, "train", 1, noise=1.0), synthetic_cases(100, "val", 2, noise=1.0)
    res = fit(CFG, TrainConfig(lr=5e-3, max_epochs=40, patience=2, seed=0), train, val, TIEBREAK, "cpu")
    assert len(res.history) < 40 and len(res.history) - res.best_epoch == 2


def test_fit_refuses_anything_but_train_then_val():
    a, b = synthetic_cases(10, "train", 1), synthetic_cases(10, "val", 2)
    test = synthetic_cases(10, "test", 3)
    with pytest.raises(ValueError):
        fit(CFG, TCFG, a, test, TIEBREAK)
    with pytest.raises(ValueError):
        fit(CFG, TCFG, b, b, TIEBREAK)


def test_checkpoint_round_trip_and_metadata(trained, tmp_path):
    res, train, val = trained
    meta = checkpoint_metadata(res, TCFG, train, val, "cpu")
    size = save_checkpoint(tmp_path / "m.pt", res.model, meta)
    loaded, m2 = load_checkpoint(tmp_path / "m.pt")
    assert size > 0 and m2 == json.loads(json.dumps(meta, sort_keys=True))
    assert np.array_equal(score(loaded, val, "cpu"), score(res.model, val, "cpu"))
    for key in ("variant", "architecture", "hyperparameters", "item_mapping_sha256", "training_seed",
                "dataset_fingerprints", "torch", "device"):
        assert key in m2


def test_tampered_checkpoint_is_rejected(trained, tmp_path):
    res, train, val = trained
    meta = checkpoint_metadata(res, TCFG, train, val, "cpu")
    meta["state_dict_sha256"] = "0" * 64
    save_checkpoint(tmp_path / "bad.pt", res.model, meta)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(tmp_path / "bad.pt")


def test_training_on_real_sequence_cases(processed_dir):
    """End-to-end on the session dataset: next-event cases -> SASRec -> shared evaluator."""
    from bri.experiments.baselines import load_processed
    from bri.prep.split import compute_cutoffs
    from bri.seq.train import train_popularity
    data = load_processed(processed_dir)
    c, p = data["meta"]["cutoffs"], data["meta"]["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    seqs = build_seq_cases(data["interactions"], cut, data["candidates"], ("train", "val"), 10)
    tr, va = seqs[("nbs", "train")], seqs[("nbs", "val")]
    tb = train_popularity(data["interactions"], "nbs", data["candidates"]["nbs"])
    res = fit(SASRecConfig(n_items=len(tr.items_vocab), max_len=10, dim=16, heads=2, blocks=1),
              TrainConfig(max_epochs=3, seed=1), tr, va, tb, "cpu")
    assert 0.0 <= res.best_val <= 1.0 and len(res.history) == 3


# --- CUDA (conditional) --------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.skipif(not HAS_CUDA, reason="no usable CUDA GPU")
def test_cuda_training_is_deterministic_and_runs_in_deterministic_mode():
    code = (
        "import json, sys; sys.path.insert(0, 'tests')\n"
        "from bri.seq import runtime as R; R.configure_determinism()\n"
        "from test_sasrec import synthetic_cases, CFG, TCFG, TIEBREAK\n"
        "from bri.seq.train import fit, state_dict_sha256\n"
        "tr, va = synthetic_cases(600, 'train', 1), synthetic_cases(200, 'val', 2)\n"
        "r = [fit(CFG, TCFG, tr, va, TIEBREAK, 'cuda:0') for _ in range(2)]\n"
        "print(json.dumps({'sha': [state_dict_sha256(x.model) for x in r], 'best': [x.best_val for x in r],\n"
        "  'det': R.determinism_settings()['use_deterministic_algorithms'], 'peak': r[0].runtime['peak_gpu_memory_mib']}))\n")
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["det"] is True and res["sha"][0] == res["sha"][1] and res["best"][0] == res["best"][1]
    assert res["best"][0] > 0.9 and res["peak"] > 0


def test_bench_refuses_missing_tuning_data(tmp_path):
    from bri.seq import bench
    with pytest.raises(FileNotFoundError, match="bri.seq.prepare"):
        bench.run(tmp_path, devices=("cpu",))


def test_determinism_settings_record_cpu_threads():
    s = R.determinism_settings()
    assert s["cpu_threads"] == torch.get_num_threads() and "thread count" in s["scope"]
