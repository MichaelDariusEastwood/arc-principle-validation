"""The checkable code domain: the artefact, the frozen hidden suite, the checkpoint store, the four
balance objects, and the dose lever. This is what turns the runner from a thing that scores a
simulated system into a thing that can run a pilot against a real one.

WHAT THE REGISTRATIONS REQUIRE OF THIS FILE, AND WHERE EACH REQUIREMENT LANDS.

Unit 79 (P5) reads capability as a verified pass count on a ladder that is frozen, hidden from every
system under test, hashed into the seal, and carrying headroom to the final checkpoint, with the
ladder's own binomial precision entering the errors-in-variables step. `TaskPool` freezes and hashes;
`SuiteLadder` draws a subset and returns the pass count with the subset size as its denominator, and
the read's error is the finite-population sampling error of that draw against the pool rather than a
binomial in the subset alone, so a whole-pool read reports the zero error it has (finding A4, and see
arc_runner.sampling); `at_ceiling` fires at the registered headroom
fraction rather than at a full pass, because a system that has passed nine tenths of the pool can no
longer be resolved by it and the honest report is NOT EVALUABLE.

Unit 04 (P16) reads a correction margin from four objects the theory-level registration names, being
trend, level, backlog and event. `BalanceTracker` computes all four from the same suite results the
ladder produces, so the margin is measured rather than inferred and no rater is involved.

THE ONE THING THIS FILE CANNOT DO FOR YOU. Running code a model wrote is running code someone else
wrote. `subprocess_verifier` isolates each check in its own interpreter with a wall-clock timeout and,
on platforms that support it, a CPU and address-space limit, which is enough to stop a runaway loop
and an accidental fork bomb. It is NOT enough to run an untrusted model's output on a machine that
holds anything you care about: a registered run puts the verifier in a container with no network and
no writable mount outside its own scratch directory. That is an operational requirement of the
protocol and this module states it rather than implying it is handled.

WHAT DOES NOT EXIST HERE, said plainly. The task pool itself. The items, their statements, their
examples and their checks are written by a person for the study, are never drawn from a public
benchmark, and no pool has been written. `reference_pool` builds a tiny arithmetic pool so that the
real path has a test, and it is a smoke test and not an instrument.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import sampling as SAMPLING
from .custody import attest_exact_check
from .ladder import Ladder, LadderResult, outcome_digest


# --------------------------------------------------------------------------------------------------
# The artefact
# --------------------------------------------------------------------------------------------------

def new_artefact(text: str = "", system: str = "system") -> Dict[str, Any]:
    """A code artefact is one source file plus its bookkeeping. `retain` in trajectory.py subsamples
    its lines, which is the retention channel the bank titrates."""
    return {"kind": "code", "text": text, "rounds": 0, "system": system}


def artefact_sha256(artefact: Dict[str, Any]) -> str:
    return hashlib.sha256(artefact.get("text", "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------
# The task pool, frozen and hashed
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One item. `shown_examples` is what the system is given; `checks` is what decides the item and is
    never shown. The protocol requires the shown set to be a strict subset of the checks, so that
    passing what is shown is not passing the item, and `TaskPool` refuses a pool that breaks it."""
    id: str
    statement: str
    signature: str
    shown_examples: Tuple[str, ...] = ()
    checks: Tuple[str, ...] = ()
    difficulty: int = 1

    def prompt_block(self) -> str:
        ex = ("\nExamples:\n" + "\n".join(self.shown_examples)) if self.shown_examples else ""
        return "%s\n%s%s" % (self.statement, self.signature, ex)


class TaskPool:
    """A frozen, hashed set of items. Freezing happens at construction: the hash covers every item's
    statement, signature, shown examples and checks, so an edited pool is a different pool and the
    seal will say so."""

    def __init__(self, tasks: Sequence[Task], name: str = "pool", smoke_only: bool = False):
        if not tasks:
            raise ValueError("a pool with no items is not a ladder")
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item ids in the pool")
        for t in tasks:
            if not t.checks:
                raise ValueError("item %s has no checks; an item that cannot fail is not an item" % t.id)
            if not set(t.shown_examples) < set(t.checks):
                raise ValueError("item %s shows every check; passing what is shown would pass the item" % t.id)
        self.tasks = tuple(tasks)
        self.name = name
        # A pool that nobody wrote for the study carries that fact with it, so that the ladder built
        # on it carries it into the seal and a confirmatory run can refuse it before it spends.
        self.smoke_only = bool(smoke_only)
        self.spec = {"kind": "code-task-pool", "name": name,
                     "items": [{"id": t.id, "statement": t.statement, "signature": t.signature,
                                "shown": list(t.shown_examples), "checks": list(t.checks),
                                "difficulty": t.difficulty} for t in self.tasks]}
        self.sha256 = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.tasks)

    def prompt(self) -> str:
        """Every task statement, which is what a round is given. The checks never appear here."""
        return "\n\n".join(t.prompt_block() for t in self.tasks)

    def subset(self, rng: np.random.Generator, k: int) -> List[Task]:
        k = min(int(k), len(self.tasks))
        idx = rng.choice(len(self.tasks), size=k, replace=False)
        return [self.tasks[int(i)] for i in sorted(idx)]


# --------------------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------------------

def subprocess_verifier(timeout_s: float = 10.0, cpu_s: int = 10, mem_mb: int = 512) -> Callable[[str, Task], bool]:
    """Run one item's checks against the artefact in a fresh interpreter. Returns a callable so that a
    pool can carry its verifier by reference and the ladder never holds the limits itself.

    The artefact and the checks are concatenated into one file rather than imported, because an import
    would need the artefact on the path and that is one more thing to get wrong. Exit code zero and
    nothing else counts as a pass: an item that crashes, hangs or exits non-zero has failed."""

    def preexec():                                     # POSIX only; ignored where unavailable
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        except Exception:
            pass

    def verify(text: str, task: Task) -> bool:
        # Require completion of the checks, not merely a zero process exit.
        return task.id in subprocess_batch_runner(timeout_s,cpu_s,mem_mb)(text,[task])

    # It decides an item by running that item's hidden checks and reading the exit code, so it is a
    # measurement of whether the task was solved and attests it. See custody.attest_exact_check.
    return attest_exact_check(verify)


def inprocess_batch_runner() -> Callable[[str, Sequence[Task]], Set[str]]:
    """FOR TESTS ONLY, for the reason given under `inprocess_verifier`. Compiles the artefact once per
    read rather than once per item and runs each item's checks in its own copy of the resulting
    namespace, so one item cannot pass or fail because of another."""

    def run(text: str, items: Sequence[Task]) -> Set[str]:
        base: Dict[str, Any] = {}
        try:
            exec(compile(text, "<artefact>", "exec"), base)
        except Exception:
            return set()
        out = set()
        for it in items:
            ns = dict(base)
            try:
                for c in it.checks:
                    exec(compile(c, "<check>", "exec"), ns)
                out.add(it.id)
            except Exception:
                pass
        return out

    return attest_exact_check(run)


def subprocess_batch_runner(timeout_s: float = 60.0, cpu_s: int = 60, mem_mb: int = 1024) -> Callable[[str, Sequence[Task]], Set[str]]:
    """The default. One child interpreter per READ rather than per item: the artefact is compiled
    once, each item's checks run in a fresh namespace inside that child, and the passing ids come back
    on standard output. The property that matters is that nothing the artefact does happens in the
    process holding the seal, and that is preserved; per-item process isolation is traded for a cost
    that a real pool makes prohibitive, since a read of two hundred items would otherwise be two
    hundred interpreters. An item whose checks hang takes the whole read down, which is why the read
    carries its own timeout and returns nothing rather than a partial pass count.

    The container requirement in this module's header is not softened by any of this."""

    def preexec():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        except Exception:
            pass

    def run(text: str, items: Sequence[Task]) -> Set[str]:
        payload = [{"id": it.id, "checks": list(it.checks)} for it in items]
        driver = (
            "import json, sys\n"
            "_ITEMS = json.loads(sys.argv[1])\n"
            "_BASE = {}\n"
            "_SRC = open(sys.argv[2], encoding='utf-8').read()\n"
            "try:\n"
            "    exec(compile(_SRC, '<artefact>', 'exec'), _BASE)\n"
            "except BaseException:\n"
            "    print(''); raise SystemExit(0)\n"
            "_ok = []\n"
            "for _it in _ITEMS:\n"
            "    _ns = dict(_BASE)\n"
            "    try:\n"
            "        for _c in _it['checks']:\n"
            "            exec(compile(_c, '<check>', 'exec'), _ns)\n"
            "        _ok.append(_it['id'])\n"
            "    except BaseException:\n"
            "        pass\n"
            "print(json.dumps(_ok))\n"
        )
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "artefact.py")
            drv = os.path.join(d, "driver.py")
            with open(art, "w", encoding="utf-8") as f:
                f.write(text)
            with open(drv, "w", encoding="utf-8") as f:
                f.write(driver)
            try:
                r = subprocess.run([sys.executable, "-I", drv, json.dumps(payload), art], cwd=d,
                                   timeout=timeout_s, capture_output=True, text=True,
                                   preexec_fn=preexec if os.name == "posix" else None)
                if r.returncode != 0: return set()
                line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
                got = json.loads(line) if line.startswith("[") else []
                allowed = {it.id for it in items}
                if not isinstance(got,list) or any(not isinstance(v,str) for v in got) or not set(got)<=allowed:
                    return set()
                return set(got)
            except Exception:
                return set()

    return attest_exact_check(run)


def inprocess_verifier() -> Callable[[str, Task], bool]:
    """A verifier that executes in this interpreter. It is FOR TESTS ONLY, and the reason is not
    fastidiousness: a registered run executes code a model wrote, and code a model wrote does not run
    in the process holding the seal. It exists because the subprocess verifier costs one interpreter
    per item, which a test suite cannot afford, and because a test needs to prove that the bank, the
    ladder and the verdicts compose on real verification rather than on a simulated pass count.

    `SuiteLadder` will not take it unless the caller passes it explicitly, so a run that forgets gets
    the isolated one."""

    def verify(text: str, task: Task) -> bool:
        ns: Dict[str, Any] = {}
        try:
            exec(compile(text, "<artefact>", "exec"), ns)
            for c in task.checks:
                exec(compile(c, "<check>", "exec"), ns)
            return True
        except Exception:
            return False

    return attest_exact_check(verify)


# --------------------------------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------------------------------

class SuiteLadder(Ladder):
    """The frozen hidden suite. A read draws `subset_size` items uniformly at random and returns how
    many pass, so the read's error is binomial with the subset size as its denominator, which is what
    the errors-in-variables step consumes. A whole-pool read is what the negative-control cells use.

    THE HEADROOM RULE IS APPLIED HERE AND NOT LEFT TO THE CALLER. `at_ceiling` fires when the read's
    pass fraction reaches `headroom_fraction`, which the execution protocol sets at nine tenths. A
    system above that line cannot be resolved further by this pool, and the registration's answer to
    that is NOT EVALUABLE rather than a score at a ceiling.

    WHAT A REPEATED READ OF THIS LADDER RESAMPLES (finding A4). The item form, and nothing else. The
    verifiers are fixed, the pool is frozen and the artefact is the artefact: the only thing that
    differs between two reads is which items were drawn, so the sampling unit is the item form and
    the target is the artefact's pass fraction over the whole pool. Two consequences follow and the
    docstring above used to state the first of them wrongly. A read's error is NOT binomial in the
    subset size: the draw is without replacement, so the finite population correction applies. And a
    read of the WHOLE pool is the population rather than a sample of it, so it has no sampling error
    at all, and repeating it draws the same items again and buys nothing. See arc_runner.sampling."""

    assay_unit = SAMPLING.AssayUnit.ITEM_FORM
    assay_unit_declared = True

    def __init__(self, pool: TaskPool, subset_size: Optional[int] = None,
                 verifier: Optional[Callable[[str, Task], bool]] = None, headroom_fraction: float = 0.9,
                 batch_runner: Optional[Callable[[str, Sequence[Task]], Set[str]]] = None):
        super().__init__(subset_size or len(pool), {"kind": "code-suite", "pool_sha256": pool.sha256,
                                                    "pool_name": pool.name, "pool_items": len(pool),
                                                    "subset_size": int(subset_size or len(pool)),
                                                    "headroom_fraction": float(headroom_fraction),
                                                    "pool_smoke_only": bool(getattr(pool, "smoke_only", False))})
        # Both arguments are checked at construction, because a ladder built with either of them
        # wrong reads a different pool from the one its own hash says it read: a subset size outside
        # the pool silently becomes the whole pool, and a headroom fraction outside the unit interval
        # makes the ceiling rule either unreachable or always reached.
        if subset_size is not None and (isinstance(subset_size, bool) or int(subset_size) != subset_size
                                        or not 1 <= subset_size <= len(pool)):
            raise ValueError("subset size must be an integer within the pool")
        if not 0 < headroom_fraction <= 1:
            raise ValueError("invalid headroom fraction")
        self.pool = pool
        self.smoke_only = bool(getattr(pool, "smoke_only", False))
        self.subset_size = int(subset_size or len(pool))
        self.headroom_fraction = float(headroom_fraction)
        # a per-item verifier is honoured when one is given, because a domain may need it; otherwise
        # the batch runner reads the whole subset in one child interpreter
        self.verifier = verifier
        self.batch_runner = batch_runner or (None if verifier else subprocess_batch_runner())

    def population_size(self) -> Optional[int]:
        """The frozen pool, which is what a subset read is a sample OF and what a whole-pool read is."""
        return len(self.pool)

    def _run(self, text: str, items: Sequence[Task]) -> Set[str]:
        if self.batch_runner is not None:
            return self.batch_runner(text, items)
        return {t.id for t in items if self.verifier(text, t)}

    @staticmethod
    def form_digest(items: Sequence[Task]) -> str:
        """Which items this read drew, as one value. It travels on the result so that two readings can
        be compared on the form and not only on the count: two different forms of thirty items can
        both count eighteen passes and are still two independent draws, and a ladder that draws one
        form and reads it twice shows the same digest. Finding A4, and see arc_runner.sampling."""
        h = hashlib.sha256(b"arc-item-form/1")
        for t in items:
            b = str(t.id).encode("utf-8")
            h.update(str(len(b)).encode()); h.update(b":"); h.update(b)
        return h.hexdigest()

    def resampling_witness(self, rng: np.random.Generator) -> Optional[str]:
        """What a repeat of a read here draws afresh: the item form, and nothing else.

        It draws one and digests it, which reads no artefact and runs no check, so a deciding run can
        be shown before it spends that this ladder's repeats really are fresh draws. A whole-pool
        ladder answers the same digest every time, which is the truth about it: it takes no reduction
        from repeats and `arc_runner.sampling.claims_a_reduction` never asks it."""
        return self.form_digest(self.pool.subset(rng, self.subset_size))

    def score(self, artefact: Dict[str, Any], rng: np.random.Generator) -> LadderResult:
        items = self.pool.subset(rng, self.subset_size)
        passed = self._run(artefact.get("text", ""), items)
        n = len(items)
        return LadderResult(passes=len(passed), n_items=n,
                            at_ceiling=(len(passed) >= self.headroom_fraction * n),
                            population_size=len(self.pool), form_sha256=self.form_digest(items),
                            # WHAT EACH DRAWN ITEM DID (finding A8). The form digest says which items
                            # were asked; without this the saved count cannot be checked against any
                            # response at all, because two artefacts read on one form give the same
                            # form digest and different answers.
                            outcome_sha256=outcome_digest([t.id for t in items], passed))

    def score_all(self, artefact: Dict[str, Any]) -> Tuple[LadderResult, Set[str]]:
        """The whole pool, and which items passed. The control cells read this way, and the balance
        objects need the identities and not only the count."""
        passed = self._run(artefact.get("text", ""), self.pool.tasks)
        n = len(self.pool)
        result = LadderResult(passes=len(passed), n_items=n,
                              at_ceiling=(len(passed) >= self.headroom_fraction * n),
                              population_size=n,     # the read IS the pool, so it has no sampling error
                              form_sha256=self.form_digest(self.pool.tasks),
                              outcome_sha256=outcome_digest([t.id for t in self.pool.tasks], passed))
        # The artefact travels with the record so the count can be tied to what it counted
        self.record_read(result, "whole-pool", artefact)   # a whole-pool read is a read; the bundle counts it
        return (result, passed)


def precision_curve(ladder: SuiteLadder, artefacts: Sequence[Dict[str, Any]], subset_sizes: Sequence[int],
                    rng: np.random.Generator, reads: int = 8) -> List[Dict[str, Any]]:
    """What the pilot reports about the pool: relative standard error of a read against the reading,
    at each subset size, for each artefact offered. The registered configuration's lowest state must
    come back at or below two per cent, and the sizing rule enlarges the subset until it does.

    Each row carries the OBSERVED spread of the reads and, beside it, the error the run-time model
    will report for a read of that size. The two must agree, and a row where they do not is the
    sizing rule and the reading model disagreeing about the same pool, which is worth seeing before
    the bank is run rather than after. Finding A4."""
    out = []
    for art in artefacts:
        full, _ = ladder.score_all(art)
        for k in subset_sizes:
            sub = SuiteLadder(ladder.pool, k, ladder.verifier, ladder.headroom_fraction, ladder.batch_runner)
            results = [sub.score(art, rng) for _ in range(reads)]
            vals = [r.score for r in results]
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            modelled = SAMPLING.read_uncertainty(sub, results[:1]).sd     # the error of ONE read
            out.append({"artefact_sha256": artefact_sha256(art)[:16], "whole_pool_passes": full.passes,
                        "subset_size": int(k), "mean": mean, "sd": sd, "modelled_sd": float(modelled),
                        "relative_se": (sd / mean) if mean > 0 else float("inf"),
                        "modelled_relative_se": (modelled / mean) if mean > 0 else float("inf"),
                        "at_ceiling": bool(full.at_ceiling)})
    return out


# --------------------------------------------------------------------------------------------------
# The checkpoint store: what `place_at_state` and `start_for` read from
# --------------------------------------------------------------------------------------------------

class CheckpointStore:
    """Artefacts on disk, addressed by name, each hashed. The bank places every cell at a state from
    ONE checkpoint per state, and the held-out panel's sealed window is generated from ONE checkpoint
    per system: both are reads from here, so the thing that was sealed is the thing that was run."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.root, name + ".py")

    def save(self, name: str, artefact: Dict[str, Any]) -> str:
        with open(self.path(name), "w", encoding="utf-8") as f:
            f.write(artefact.get("text", ""))
        return artefact_sha256(artefact)

    def load(self, name: str) -> Dict[str, Any]:
        with open(self.path(name), encoding="utf-8") as f:
            return new_artefact(f.read(), system=name)

    def has(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def hashes(self) -> Dict[str, str]:
        out = {}
        for n in sorted(os.listdir(self.root)):
            if n.endswith(".py"):
                with open(os.path.join(self.root, n), encoding="utf-8") as f:
                    out[n[:-3]] = hashlib.sha256(f.read().encode()).hexdigest()
        return out


def state_name(target: int) -> str:
    if isinstance(target,bool) or int(target)!=target or target<0:
        raise ValueError("state target must be a nonnegative integer, never rounded")
    return "state-%d" % int(target)


def place_at_state_factory(store: CheckpointStore):
    """`place_at_state(target)` for the P5 bank: the checkpointed artefact whose reading sits at that
    target. Refuses a missing state rather than starting from an empty file, because a cell placed at
    a state that was never built is not a cell at that state."""
    def place(target):
        name = state_name(target)
        if not store.has(name):
            raise FileNotFoundError("no checkpoint for %s; build the state ladder before the bank runs" % name)
        return store.load(name)
    place.checkpoint_store = store      # the declaration a confirmatory run checks; see arc_runner.mode
    return place


def start_for_factory(store: CheckpointStore, seed_name: str = "seed"):
    """`start_for(system)` for the held-out panel: that system's own checkpoint where one exists, and
    the frozen seed artefact where the panel has not yet been started."""
    def start(system: str):
        art = store.load(system) if store.has(system) else store.load(seed_name)
        art["system"] = system
        return art
    start.checkpoint_store = store      # as above: a loader that reads no store is a placeholder
    return start


def build_state_ladder(adapter, ladder: SuiteLadder, store: CheckpointStore, targets: Sequence[int],
                       task: str, rng: np.random.Generator, max_rounds: int = 400,
                       reads: int = 8) -> List[Dict[str, Any]]:
    """Run the seed artefact forward, checkpointing when the reading crosses each target. This is how
    a state is realised: it is a real artefact at that capability and not a parameter."""
    from .trajectory import run_round
    from .ladder import read_mean
    art = store.load("seed")
    todo = sorted(int(t) for t in targets)
    made = []
    for r in range(1, max_rounds + 1):
        art = run_round(adapter, art, 1.0, task, rng)
        mean, sd, ceiling = read_mean(ladder, art, rng, reads)
        while todo and mean >= todo[0]:
            t = todo.pop(0)
            sha = store.save(state_name(t), art)
            made.append({"target": t, "round": r, "reading": mean, "read_sd": sd, "sha256": sha})
        if not todo:
            break
        if ceiling:
            raise RuntimeError("the pool reached its headroom limit at round %d before state %d was built" % (r, todo[0]))
    if todo:
        raise RuntimeError("states not reached within %d rounds: %s" % (max_rounds, todo))
    return made


# --------------------------------------------------------------------------------------------------
# The four balance objects, and the correction margin
# --------------------------------------------------------------------------------------------------

@dataclass
class BalanceReading:
    round_index: int
    level: int                 # items passing now
    trend: int                 # change in the pass count this round produced
    backlog: int               # items that failed at some earlier round and still fail
    event: int                 # items that passed last round and fail now
    fixes: int                 # backlog items cleared this round
    margin: float              # (fixes - events - backlog growth) / level
    # THE TWO THE REGISTERED BALANCE IS A RATIO OF (finding A1). The margin above is a per-round
    # surplus rate, which is a level object: the registered Delta is the logarithmic slope of the
    # ratio of correction service to offered burden, and a ratio cannot be recovered from a
    # difference that has been divided by a level. Both are kept, separately and unreduced, so that
    # the reading can be read either way and re-checked afterwards.
    service: int = 0           # Q: faults the correction service cleared this round
    burden: int = 0            # W: faults this round introduced, being the burden it offered

    def as_dict(self) -> Dict[str, Any]:
        return {"round": self.round_index, "level": self.level, "trend": self.trend,
                "backlog": self.backlog, "event": self.event, "fixes": self.fixes,
                "margin": self.margin, "service": self.service, "burden": self.burden}


class BalanceTracker:
    """The four objects, from the same whole-pool reads the ladder performs. Every term is a count of
    items on a frozen suite, which is why the P16 registration's margin is measured and not judged.

    `fixes` counts items that were in the backlog and now pass. `event` counts items that passed last
    round and fail now, being regressions this round introduced. Backlog growth is the change in the
    carried-fault set. The margin divides by the level so that the quantity is a rate, which is what
    the balance line is fitted to."""

    def __init__(self, pool: TaskPool):
        self.pool = pool
        self.ever_failed: Set[str] = set()
        self.prev_pass: Optional[Set[str]] = None
        self.prev_backlog = 0
        self.history: List[BalanceReading] = []

    def observe(self, passed: Set[str], round_index: int) -> BalanceReading:
        all_ids = {t.id for t in self.pool.tasks}
        failing = all_ids - passed
        if self.prev_pass is None:
            trend, event, fixes = len(passed), 0, 0
            backlog = 0                                   # nothing can be carried at the first reading
            self.ever_failed |= failing
            growth = 0
        else:
            trend = len(passed) - len(self.prev_pass)
            event = len(self.prev_pass - passed)
            # a fix is an item that has failed before, was not passing last round, and passes now
            fixes = len((self.ever_failed - self.prev_pass) & passed)
            self.ever_failed |= failing
            backlog = len(self.ever_failed & failing)     # carried faults: failed before and failing still
            growth = backlog - self.prev_backlog
        level = max(len(passed), 1)
        margin = (fixes - event - max(growth, 0)) / float(level)
        # Q is the service delivered this round and W is the burden this round offered. Carried
        # backlog growth is NOT added to W here: growth is already the difference between the faults
        # that arrived and the faults that were cleared, so adding it would count the same items on
        # both sides of the ratio. The margin keeps its own definition, unchanged.
        r = BalanceReading(round_index, len(passed), trend, backlog, event, fixes, margin,
                           service=fixes, burden=event)
        self.history.append(r)
        self.prev_pass, self.prev_backlog = set(passed), backlog
        return r

    def margins(self) -> List[float]:
        return [r.margin for r in self.history]


def suite_margin_source(adapter, ladder: SuiteLadder, store: CheckpointStore, task: str,
                        dose_for, switch_round: int, start_name: str = "seed"):
    """The P16 observation source for a real system, TYPED at the boundary (finding A1).

    One call per arm per round: the round is run at that arm's dose, the whole pool is read, the four
    balance objects are computed, and a reading is returned carrying Q and W SEPARATELY rather than a
    single number whose meaning nobody has settled. Q is the count of carried faults the round
    cleared. W is the count of regressions the round introduced, being the burden it offered. R is
    the artefact's accumulated revision depth, which is the recursive coordinate the registered
    balance is read against and is not the round number: a high-dose arm spends more revision passes
    per round and so travels further in R, and reading it on the round clock would misattribute that.

    The declared quantity is the service ratio Q/W, whose logarithm is fitted against log R, so the
    fitted slope is the balance elasticity itself. The four balance objects travel with each reading,
    so nothing that was measured before is lost.

    Arms are independent, so each keeps its own artefact and its own tracker.
    """
    from . import observation as OBS
    state: Dict[str, Dict[str, Any]] = {}

    def src(arm: str, alpha_arm: float, r: int, rng: np.random.Generator) -> Dict[str, Any]:
        from .trajectory import run_round
        key = "%s@%.4f" % (arm, alpha_arm)
        # r=0 starts a new assigned replicate, never a continuation of the previous one.
        if key not in state or r == 0:
            art = store.load(start_name)
            art["system"] = key
            state[key] = {"artefact": art, "tracker": BalanceTracker(ladder.pool)}
        s = state[key]
        expected = len(s["tracker"].history)
        if r != expected: raise ValueError("trajectory rounds must be consecutive")
        fraction, passes = dose_for(arm, alpha_arm, r, switch_round)
        art = s["artefact"]
        for _ in range(max(1, int(passes))):
            art = run_round(adapter, art, fraction, task, rng)
        s["artefact"] = art
        _, passed = ladder.score_all(art)
        reading = s["tracker"].observe(passed, r)
        # THE REVISION DEPTH, OR NOTHING. This read `float(art.get("rounds", 0) or 0) or float(r + 1)`,
        # so an artefact that does not track its revision depth silently became the round number,
        # which is the conflation the paragraph above says must not happen: a high-dose arm spends
        # more revision passes per round and travels further in R than the rounds it took. An
        # untracked artefact has no recursive coordinate, and a reading with no R is refused by
        # arc_runner.observation rather than given a substitute that means something else. Every
        # adapter in this runner increments `rounds` once per revision pass, so nothing that tracks
        # its own depth is affected.
        tracked = float(art.get("rounds", 0) or 0)
        depth = tracked if tracked > 0 else None
        dosed = bool(r >= switch_round and arm not in ("sham", "baseline"))
        extra = dict(reading.as_dict())
        # THE APPARATUS'S OWN RECORD THAT THE DOSE WAS ADMINISTERED, AND WHAT IT MOVED (finding A2).
        # An unattested arm cannot be told apart from an arm whose dose silently failed, and the
        # support branch treats an unattested arm as unsupplied evidence rather than as a yes.
        extra[OBS.DELIVERY_KEY] = {"applied": dosed,
                                   "lever": {"retention_fraction": float(fraction),
                                             "revision_passes": int(passes)} if dosed else None}
        # U IS THE CAPABILITY THE LADDER MEASURED, being the count of items passing now, and it is
        # carried so that the arm's REALISED growth exponent can be measured against the revision
        # depth it reached. Finding A2: the assigned dose is what the apparatus was asked to deliver
        # and is not a measurement of what the system was exposed to.
        return {"round": r, "Q": float(reading.service), "W": float(reading.burden), "R": depth,
                "level": float(reading.level), "U": float(reading.level), "extra": extra}

    OBS.declare(src, OBS.service_ratio_observation(
        supplies_q_and_w=True, source="code-domain-suite",
        note="Q is the count of carried faults cleared this round and W the count of regressions it "
             "introduced, both read from the same frozen hidden suite the ladder reads. R is the "
             "artefact's accumulated revision depth. A round that introduced no regression has W = 0, "
             "for which the ratio does not exist: it is excluded and counted, never treated as an "
             "unbounded service ratio."))
    # THE SOURCE CARRIES THE ADAPTER'S REACH, so the refusal can happen at the library boundary. A
    # margin source is a plain callable by the time it reaches the run, and without this flag the run
    # cannot tell a simulated margin from one bought from a remote endpoint.
    src.uses_remote_endpoint = getattr(adapter, "uses_remote_endpoint", False)
    src.state = state                                                    # the caller may inspect and hash it
    return src


# --------------------------------------------------------------------------------------------------
# The dose lever, and the locating stage
# --------------------------------------------------------------------------------------------------

@dataclass
class DoseSchedule:
    """The first lever: how much of its own prior output a round may reuse, and how many revision
    passes it may spend. Both rise with the dose. Before the switch every arm runs at the baseline;
    after it, each arm holds its own dose for the whole horizon. The sham moves the apparatus and not
    the dose, which is what makes it a coefficient-only sham."""
    base_fraction: float = 0.5
    base_passes: int = 1
    fraction_per_unit: float = 0.25
    passes_per_unit: float = 1.0
    max_fraction: float = 1.0

    def for_offset(self, offset: float) -> Tuple[float, int]:
        f = min(self.max_fraction, max(0.05, self.base_fraction + self.fraction_per_unit * offset))
        p = max(1, int(round(self.base_passes + self.passes_per_unit * max(offset, 0.0))))
        return f, p

    def dose_for(self, arm: str, alpha_arm: float, r: int, switch_round: int) -> Tuple[float, int]:
        if r < switch_round or arm in ("sham", "baseline"):
            return self.base_fraction, self.base_passes
        try:
            offset = float(arm.replace("dose", ""))
        except ValueError:
            return self.base_fraction, self.base_passes
        return self.for_offset(offset)


def locate_boundary(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """The locating stage: the balance line fitted across systems that never enter an arm. `points`
    are (realised exponent, post-settling margin slope). The zero is the located boundary and the
    slope is the measured correction elasticity. Refuses to locate from a line that is not falling,
    which is the same gate the verdict applies later and for the same reason."""
    if len(points) < 3:
        return {"zero": float("nan"), "slope": float("nan"), "n": len(points), "falling": False}
    x = np.asarray([p[0] for p in points], float)
    y = np.asarray([p[1] for p in points], float)
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    zero = -intercept / slope if slope != 0 else float("nan")
    return {"zero": zero, "slope": slope, "intercept": intercept, "n": len(points),
            "falling": bool(slope < 0)}


# --------------------------------------------------------------------------------------------------
# A reference pool, for tests of the real path only
# --------------------------------------------------------------------------------------------------

def reference_pool(n: int = 24, seed: int = 20260905) -> TaskPool:
    """A tiny arithmetic pool so that the real path has a test. A registered run replaces this with a
    pool a person wrote for the study. This is a smoke test and is not an instrument."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        a, b = int(rng.integers(2, 40)), int(rng.integers(2, 40))
        name = "add_%d" % i
        checks = ("assert %s(%d, %d) == %d" % (name, a, b, a + b),
                  "assert %s(0, 0) == 0" % name,
                  "assert %s(-1, 1) == 0" % name)
        tasks.append(Task(id=name, statement="Define %s(x, y) returning the sum of x and y." % name,
                          signature="def %s(x, y): ..." % name,
                          shown_examples=(checks[0],), checks=checks, difficulty=1 + i // 8))
    return TaskPool(tasks, name="reference-arithmetic", smoke_only=True)
