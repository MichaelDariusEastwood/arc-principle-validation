"""The checkable code domain: the pool freezes, the suite reads, the four balance objects count, and
the whole P5 pipeline runs end to end on real verification against a system with a known coupling.

The end-to-end case is the point of this file. Every other test in this suite scores a simulated
capability; this one writes Python, runs it against hidden checks, and reads the pass count, which is
the reading the registration actually commits to.
"""
import numpy as np
import pytest

from arc_runner import code_domain as CD, ladder as L, p5


def _lib(n_correct, n_total):
    lines = []
    for i in range(n_total):
        body = "return x + y" if i < n_correct else "return None"
        lines.append("def add_%d(x, y):\n    %s" % (i, body))
    return "\n".join(lines)


def test_a_pool_freezes_and_hashes_and_refuses_an_item_that_shows_every_check():
    p = CD.reference_pool(8)
    assert len(p) == 8 and len(p.sha256) == 64
    same = CD.reference_pool(8)
    assert same.sha256 == p.sha256, "the same pool must hash the same or the seal means nothing"
    t = CD.Task(id="x", statement="s", signature="sig", shown_examples=("assert True",), checks=("assert True",))
    with pytest.raises(ValueError):
        CD.TaskPool([t])
    with pytest.raises(ValueError):
        CD.TaskPool([CD.Task(id="y", statement="s", signature="sig", checks=())])


def test_the_pool_prompt_never_carries_the_checks():
    p = CD.reference_pool(4)
    prompt = p.prompt()
    for task in p.tasks:
        assert task.statement in prompt
        for c in task.checks:
            if c not in task.shown_examples:
                assert c not in prompt, "a hidden check reached the prompt; the loop would see the ladder"


def test_the_verifier_passes_correct_code_fails_wrong_code_and_survives_a_hang():
    v = CD.subprocess_verifier(timeout_s=3.0)
    t = CD.reference_pool(1).tasks[0]
    assert v(_lib(1, 1), t) is True
    assert v(_lib(0, 1), t) is False
    assert v("while True:\n    pass\n", t) is False, "a hanging artefact must fail rather than block the run"


def test_the_batch_runner_isolates_items_from_each_other_and_from_a_hang():
    pool = CD.reference_pool(4)
    run = CD.subprocess_batch_runner(timeout_s=8.0)
    assert run(_lib(4, 4), pool.tasks) == {t.id for t in pool.tasks}
    assert run(_lib(2, 4), pool.tasks) == {t.id for t in pool.tasks[:2]}
    assert run("while True:\n    pass\n", pool.tasks) == set()
    # an item whose check poisons its namespace must not carry into the next item
    poison = CD.Task(id="poison", statement="s", signature="sig",
                     checks=("globals()['add_0'] = None", "assert False"))
    got = run(_lib(4, 4), [poison] + list(pool.tasks))
    assert "poison" not in got and {t.id for t in pool.tasks} <= got


def test_the_default_ladder_reads_in_a_child_interpreter():
    lad = CD.SuiteLadder(CD.reference_pool(3))
    assert lad.batch_runner is not None and lad.verifier is None, \
        "the default must isolate the artefact from the process holding the seal"


def test_a_read_is_binomial_in_the_subset_size_and_the_whole_pool_read_names_the_items():
    p = CD.reference_pool(12)
    lad = CD.SuiteLadder(p, subset_size=5, batch_runner=CD.inprocess_batch_runner())
    rng = np.random.default_rng(3)
    r = lad.score(CD.new_artefact(_lib(12, 12)), rng)
    assert r.n_items == 5 and r.passes == 5
    full, passed = lad.score_all(CD.new_artefact(_lib(7, 12)))
    assert full.passes == 7 and len(passed) == 7 and full.n_items == 12


def test_headroom_fires_at_the_registered_fraction_and_not_only_at_a_full_pass():
    p = CD.reference_pool(10)
    lad = CD.SuiteLadder(p, batch_runner=CD.inprocess_batch_runner(), headroom_fraction=0.9)
    assert lad.score_all(CD.new_artefact(_lib(9, 10)))[0].at_ceiling is True
    assert lad.score_all(CD.new_artefact(_lib(8, 10)))[0].at_ceiling is False


def test_the_precision_curve_tightens_as_the_subset_grows():
    p = CD.reference_pool(40)
    lad = CD.SuiteLadder(p, batch_runner=CD.inprocess_batch_runner())
    art = CD.new_artefact(_lib(20, 40))
    rows = CD.precision_curve(lad, [art], [4, 40], np.random.default_rng(5), reads=12)
    small = [r for r in rows if r["subset_size"] == 4][0]
    whole = [r for r in rows if r["subset_size"] == 40][0]
    assert whole["relative_se"] < small["relative_se"]
    assert whole["sd"] == 0.0, "a whole-pool read of a deterministic suite has no sampling error"


def test_the_checkpoint_store_round_trips_and_a_missing_state_is_refused(tmp_path):
    store = CD.CheckpointStore(str(tmp_path))
    sha = store.save("seed", CD.new_artefact(_lib(2, 4)))
    assert store.load("seed")["text"] == _lib(2, 4)
    assert store.hashes()["seed"] == sha
    place = CD.place_at_state_factory(store)
    with pytest.raises(FileNotFoundError):
        place(100)
    store.save(CD.state_name(100), CD.new_artefact(_lib(3, 4)))
    assert place(100)["text"] == _lib(3, 4)


def test_start_for_falls_back_to_the_seed_and_carries_the_system_name(tmp_path):
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(1, 2)))
    start = CD.start_for_factory(store)
    a = start("S1")
    assert a["system"] == "S1" and a["text"] == _lib(1, 2)


def test_the_four_balance_objects_count_fixes_regressions_and_carried_faults():
    p = CD.reference_pool(6)
    ids = [t.id for t in p.tasks]
    tr = CD.BalanceTracker(p)
    a = tr.observe(set(ids[:3]), 0)
    assert (a.level, a.trend, a.backlog, a.event, a.fixes) == (3, 3, 0, 0, 0)
    b = tr.observe(set(ids[:2]) | {ids[3]}, 1)          # one regression, one fix
    assert (b.level, b.trend, b.event, b.fixes, b.backlog) == (3, 0, 1, 1, 3)
    assert b.margin == pytest.approx((1 - 1 - 3) / 3.0)
    c = tr.observe(set(ids[:2]) | {ids[3], ids[4]}, 2)  # one more fix, no regression, backlog falls
    assert (c.event, c.fixes, c.backlog) == (0, 1, 2)
    assert c.margin == pytest.approx(0.25), "a shrinking backlog is never charged as growth"


def test_the_dose_schedule_holds_the_baseline_before_the_switch_and_leaves_the_sham_alone():
    d = CD.DoseSchedule()
    assert d.dose_for("dose+0.6", 2.6, 3, 8) == (d.base_fraction, d.base_passes)
    assert d.dose_for("sham", 2.0, 40, 8) == (d.base_fraction, d.base_passes)
    assert d.dose_for("baseline", 2.0, 40, 8) == (d.base_fraction, d.base_passes)
    f_hi, p_hi = d.dose_for("dose+0.6", 2.6, 40, 8)
    f_lo, p_lo = d.dose_for("dose-0.6", 1.4, 40, 8)
    assert f_hi > d.base_fraction > f_lo and p_hi > p_lo


def test_locate_boundary_recovers_a_known_zero_and_refuses_a_rising_line():
    pts = [(a, -0.5 * (a - 2.0)) for a in (1.4, 1.7, 2.3, 2.6, 2.9)]
    got = CD.locate_boundary(pts)
    assert got["zero"] == pytest.approx(2.0, abs=1e-6) and got["falling"] is True
    assert CD.locate_boundary([(a, 0.5 * (a - 2.0)) for a in (1.4, 2.0, 2.6)])["falling"] is False
    assert np.isnan(CD.locate_boundary([(1.0, 1.0)])["zero"])


class ScriptedCodeSystem:
    """A system that writes code and whose coupling is known. One round implements a number of further
    functions that scales as (available capability) to the power beta, so the pass count on the hidden
    suite grows by the registered law and every reading is a real verification rather than a number."""
    name = "scripted-code"

    def __init__(self, n_total, beta=0.5, a=1.6, noise=0.0):
        self.n_total, self.beta, self.a, self.noise = n_total, beta, a, noise

    def revise(self, artefact, retained, task, rng):
        text = artefact.get("text", "")
        done = text.count("return x + y")
        available = max(float(retained.get("fraction", 1.0)) * max(done, 1), 1e-9)
        step = self.a * available ** self.beta
        if self.noise:
            step *= float(np.exp(rng.normal(0.0, self.noise)))
        new_done = min(self.n_total, done + max(1, int(round(step))))
        out = dict(artefact)
        out["text"] = _lib(new_done, self.n_total)
        out["rounds"] = int(artefact.get("rounds", 0)) + 1
        return out


def test_p5_runs_end_to_end_on_the_code_domain_with_real_verification(tmp_path):
    pool = CD.reference_pool(400)
    lad = CD.SuiteLadder(pool, subset_size=200, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(4, 400)))
    for target in (20, 40, 70):
        store.save(CD.state_name(target), CD.new_artefact(_lib(target, 400)))
    ad = ScriptedCodeSystem(400, beta=0.5, a=1.6, noise=0.02)
    cfg = p5.P5Config(states=(20, 40, 70), fractions=(0.4, 0.7, 1.0), reps=3, window_end=16,
                      checkpoints=(4, 8, 16), calibration_depths=(1, 2, 3, 4),
                      cal_reads=4, reads=2, heldout_reads=2, replicates=2, bootstrap=40)
    res = p5.run_p5(ad, lad, cfg, 11, CD.place_at_state_factory(store), CD.start_for_factory(store), ["S1"])
    r = res["routes"]
    assert np.isfinite(r["beta_pooled"]), r
    assert res["diagnostics"]["PREDICTION"] in ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT EVALUABLE")
    assert res["manifest"]["ladder_sha256"] == lad.sha256
    assert res["manifest"]["seal"], "the seal must exist before any verdict is read"
