"""SASRec ablations: each variant changes exactly its intended component, nothing else."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bri.seq import runtime as R  # noqa: E402
from bri.seq.sasrec import VARIANTS, SASRec, SASRecConfig, count_parameters, shuffle_history  # noqa: E402
from bri.seq.train import TrainConfig, fit, load_checkpoint, save_checkpoint, score, checkpoint_metadata, \
    state_dict_sha256  # noqa: E402
from test_sasrec import TIEBREAK, VOCAB, batch, synthetic_cases  # noqa: E402

HAS_CUDA = R.cuda_usable()[0]
L, D = 6, 16


def make(variant, seed=0):
    torch.manual_seed(seed)
    return SASRec(SASRecConfig(n_items=len(VOCAB), max_len=L, dim=D, heads=2, blocks=2, dropout=0.0,
                               variant=variant)).eval()


def flags(rows):
    out = torch.zeros(len(rows), L, dtype=torch.long)
    for i, f in enumerate(rows):
        out[i, L - len(f):] = torch.tensor(f)
    return out


X = batch([[1, 2, 3], [4, 1], [2, 2, 3, 4, 1, 3]])
F = flags([[0, 0, 1], [0, 1], [0, 0, 1, 0, 0, 1]])


def test_every_variant_shares_initial_parameters_with_full():
    """Same seed -> every parameter a variant shares with Full starts bitwise identical."""
    full = make("full").state_dict()
    for v in VARIANTS:
        sd = make(v).state_dict()
        for name, t in sd.items():
            if name in full:
                assert torch.equal(t, full[name]), (v, name)


def test_parameter_inventory_differs_only_by_the_ablated_component():
    full = make("full")
    keys = set(full.state_dict())
    n = count_parameters(full)
    assert set(make("shuffled").state_dict()) == keys and count_parameters(make("shuffled")) == n
    nopos = make("no_position")
    assert keys - set(nopos.state_dict()) == {"pos_emb.weight"}
    assert count_parameters(nopos) == n - L * D
    sb = make("session_boundary")
    assert set(sb.state_dict()) - keys == {"boundary_emb.weight"} and count_parameters(sb) == n + 2 * D
    mp = make("meanpool")
    assert not any(k.startswith("blocks.") or k.startswith("pos_emb") for k in mp.state_dict())
    assert not any(isinstance(m, torch.nn.Module) and type(m).__name__ == "CausalSelfAttention" for m in mp.modules())


def test_no_position_equals_full_with_zeroed_positional_embeddings():
    full, nopos = make("full"), make("no_position")
    with torch.no_grad():
        full.pos_emb.weight.zero_()
    nopos.load_state_dict({k: v for k, v in full.state_dict().items() if k != "pos_emb.weight"})
    assert torch.allclose(full(X), nopos(X), atol=1e-6)


def test_no_position_keeps_causal_masking_and_is_not_fully_order_invariant():
    m = make("no_position")
    a, _ = m.encode(batch([[1, 2, 3, 4, 1, 2]]))
    b, _ = m.encode(batch([[1, 2, 3, 2, 4, 4]]))
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-6)  # causal
    s1, s2 = m(batch([[1, 2, 3, 4]])), m(batch([[3, 1, 4, 2]]))
    assert not torch.allclose(s1, s2, atol=1e-6)  # stacked causal layers still see order


def test_session_boundary_equals_full_when_boundary_embeddings_are_zero():
    full, sb = make("full"), make("session_boundary")
    sb.load_state_dict({**full.state_dict(), "boundary_emb.weight": torch.zeros(2, D)})
    assert torch.allclose(full(X), sb(X, F), atol=1e-6)


def test_only_session_boundary_reads_the_flags():
    other = flags([[1, 1, 0], [1, 0], [1, 1, 0, 1, 1, 0]])
    for v in VARIANTS:
        m = make(v)
        if v == "session_boundary":
            assert not torch.allclose(m(X, F), m(X, other))
            with pytest.raises(ValueError, match="new_session"):
                m(X)
        else:
            assert torch.equal(m(X, F), m(X, other))


def test_meanpool_is_a_padding_free_order_free_masked_mean():
    m = make("meanpool")
    assert torch.allclose(m(batch([[1, 2, 3]])), m(batch([[3, 1, 2]])), atol=1e-6)  # order-free
    alone = m(batch([[1, 3]]))
    mixed = m(batch([[1, 3], [2, 2, 4, 1, 3, 2]], max_tie=2))
    assert torch.allclose(alone[0], mixed[0], atol=1e-6)  # padding-free
    emb, valid = m.step_embeddings(batch([[1, 3]]))
    manual = m.ln_final((emb * D ** 0.5)[0, valid[0]].mean(0))
    assert torch.allclose(m.representation(batch([[1, 3]]))[0][0], manual, atol=1e-6)
    assert torch.equal(m(batch([[]])), torch.zeros(1, len(VOCAB)))


def test_shuffle_history_permutes_valid_steps_only_and_is_seeded():
    x = batch([[1, (2, 3), 4], [2], [1, 2, 3, 4, 1, 2]], max_tie=2)
    a = shuffle_history(x, torch.Generator().manual_seed(5))
    b = shuffle_history(x, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert not torch.equal(a, shuffle_history(x, torch.Generator().manual_seed(6)))
    for i in range(len(x)):
        valid = (x[i] > 0).any(-1)
        n = int(valid.sum())
        assert not (a[i, :L - n] > 0).any()  # padding stays at the front
        steps = lambda t: sorted(tuple(r.tolist()) for r in t[L - n:])  # noqa: E731
        assert steps(a[i]) == steps(x[i])  # same steps, tied items kept together
    assert not torch.equal(a[2], x[2])


def test_shuffled_variant_shuffles_training_only_and_scores_chronologically():
    train, val = synthetic_cases(400, "train", 1), synthetic_cases(150, "val", 2)
    cfg = dict(n_items=len(VOCAB), max_len=L, dim=D, heads=2, blocks=1, dropout=0.1)
    tcfg = TrainConfig(lr=5e-3, batch_size=32, max_epochs=6, early_stopping=False, seed=3)
    full = fit(SASRecConfig(**cfg), tcfg, train, val, TIEBREAK)
    shuf = fit(SASRecConfig(**cfg, variant="shuffled"), tcfg, train, val, TIEBREAK)
    again = fit(SASRecConfig(**cfg, variant="shuffled"), tcfg, train, val, TIEBREAK)
    assert state_dict_sha256(shuf.model) == state_dict_sha256(again.model)  # deterministic
    assert state_dict_sha256(shuf.model) != state_dict_sha256(full.model)  # training differs
    # Scoring is chronological: identical weights in a Full model give identical scores.
    clone = SASRec(SASRecConfig(**cfg))
    clone.load_state_dict(shuf.model.state_dict())
    assert np.array_equal(score(shuf.model, val, "cpu"), score(clone, val, "cpu"))
    # The target depends on the LAST step, so destroying order during training must hurt.
    assert full.best_val > shuf.best_val


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_share_training_scoring_and_checkpoint_interfaces(variant, tmp_path):
    train, val = synthetic_cases(200, "train", 1), synthetic_cases(80, "val", 2)
    tcfg = TrainConfig(lr=5e-3, batch_size=32, max_epochs=3, seed=1)
    res = fit(SASRecConfig(n_items=len(VOCAB), max_len=L, dim=D, heads=2, blocks=1, variant=variant),
              tcfg, train, val, TIEBREAK)
    s = score(res.model, val, "cpu")
    assert s.shape == (len(val.cases), len(VOCAB)) and np.isfinite(s).all()
    meta = checkpoint_metadata(res, tcfg, train, val, "cpu")
    assert meta["variant"] == variant
    save_checkpoint(tmp_path / "m.pt", res.model, meta)
    loaded, m2 = load_checkpoint(tmp_path / "m.pt")
    assert np.array_equal(score(loaded, val, "cpu"), s) and m2["variant"] == variant
    assert m2["unused_hyperparameters"] == (["heads", "blocks"] if variant == "meanpool" else [])


def test_device_guard_refuses_a_cpu_fallback_for_a_cuda_experiment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    choice = R.select_device("cuda")
    with pytest.raises(R.DeviceMismatchError, match="never merged"):
        R.require_device(choice, "cuda:0")
    R.require_device(R.select_device("cpu"), "cpu")


def test_runtime_fingerprint_changes_with_the_device_configuration(monkeypatch):
    cpu = R.runtime_fingerprint("cpu")
    assert cpu["runtime"]["device"] == "cpu" and len(cpu["runtime_sha256"]) == 64
    assert R.runtime_fingerprint("cpu")["runtime_sha256"] == cpu["runtime_sha256"]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1 if old != 1 else 2)
        assert R.runtime_fingerprint("cpu")["runtime_sha256"] != cpu["runtime_sha256"]
    finally:
        torch.set_num_threads(old)


@pytest.mark.cuda
@pytest.mark.skipif(not HAS_CUDA, reason="no usable CUDA GPU")
def test_all_variants_train_deterministically_on_cuda():
    code = (
        "import json, sys; sys.path.insert(0, 'tests')\n"
        "from bri.seq import runtime as R; R.configure_determinism()\n"
        "from test_sasrec import synthetic_cases, TIEBREAK, VOCAB\n"
        "from bri.seq.sasrec import SASRecConfig, VARIANTS\n"
        "from bri.seq.train import fit, TrainConfig, state_dict_sha256\n"
        "tr, va = synthetic_cases(300, 'train', 1), synthetic_cases(100, 'val', 2)\n"
        "t = TrainConfig(lr=5e-3, batch_size=64, max_epochs=3, early_stopping=False, seed=4)\n"
        "out = {}\n"
        "for v in VARIANTS:\n"
        "    c = SASRecConfig(n_items=len(VOCAB), max_len=6, dim=16, heads=2, blocks=1, variant=v)\n"
        "    out[v] = [state_dict_sha256(fit(c, t, tr, va, TIEBREAK, 'cuda:0').model) for _ in range(2)]\n"
        "print(json.dumps({'sha': out, 'fp': R.runtime_fingerprint('cuda:0')['runtime']}))\n")
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    for v, (a, b) in res["sha"].items():
        assert a == b, v
    assert res["fp"]["device"] == "cuda:0" and res["fp"]["determinism"]["matmul_allow_tf32"] is False
