"""Custody: the verifiable commitment, the portable code identity, and the evidence bundle.

WHY THIS FILE EXISTS. Finding A8: the seal was made in memory, its local timestamp was treated as
evidence of order, and `require_scoreable` asked only whether a pilot flag was off and a seal object
existed. Nothing checked that the predictions being scored were the predictions that were sealed,
nothing checked that the configuration had not moved after the seal, and nothing required the one
thing that makes a commitment checkable by somebody who was not in the room: an external anchor on
the seal's own hash. Meanwhile `--out` wrote the manifest and threw away the bank, the reads, the
replicate series, every P16 margin series and the provider's own account of what it charged for.

THE SIX THINGS THIS FILE ADDS, AND THE DEFECT EACH PREVENTS.

1. A RECEIPT, AND WHAT IT ATTESTS. An anchor receipt is an identifier plus the sha256 it attests. The
   digest it attests is the seal record's own hash, so the receipt binds the predictions, the
   specification hash, the sealing time and the sealer in one value: altering any of them after
   anchoring makes the recomputed digest disagree with the receipt. This code NEVER performs an
   anchor. It takes a callable from the operator, hands it the digest and records what comes back,
   because an anchor performed by the runner is an assertion by the party being checked. The only
   anchor shipped here is `mock_anchor`, which is labelled a mock in its identifier, in its label and
   in a boolean field, and which `arc_runner.manifest.require_scoreable` refuses at proposition level.
   A mock receipt is honest in a demonstration or a pilot and is a forgery in a deciding run.

2. THE COMMITMENT COMMITS TO THE RUN'S OWN KIND. `spec_hash_of` once covered the ladder, the code, the
   configuration and the ladder identity, and nothing else, so `execution_mode`, `pilot`,
   `scoreable_at_proposition_level`, the adapter, the experiment and the whole `confirmatory_inputs`
   record sat OUTSIDE both the sealed specification hash and the anchored digest. The consequence was
   demonstrated rather than theorised: a demonstration run against the simulated system, honestly
   anchored by a real service, could have two fields rewritten in its own bundle and would then pass
   every custody check as a confirmatory run. That is the promotion ruling 28 forbids and the one
   A9's mode field exists to stop, and the commitment is the only place it can be stopped, because
   the mode is what decides whether the commitment is read at all. Every one of those fields is now
   inside the sealed specification hash and therefore inside the anchored digest, so a rehearsal that
   is later relabelled a deciding run refuses instead of scoring.

3. A PORTABLE IDENTITY, WITH THE ENVIRONMENT IT RAN IN. The old code hash fed absolute filesystem
   paths into the digest, so the same bytes checked out in another directory produced a different
   identity and an independent analyst could not reproduce it at all. `code_identity` hashes paths
   RELATIVE to the package root, with every entry length delimited: the length of the path, then the
   path, then the length of the bytes, then the bytes. The lengths are not decoration. Without them a
   file named `a` holding `bc` and a file named `ab` holding `c` hash identically, and a rename could
   be concealed by a compensating edit. The identity also binds the ladder spec, and for a ladder that
   decides items by running code it binds the source of the module each verifier comes from, because a
   pool hash says which items were asked and says nothing about what counted as passing them; and
   `ladder_differences` recomputes that binding against the LIVE ladder at scoring time, because an
   identity captured once when the manifest was made and never checked again is a documentary claim
   and not an enforced one.

   Identity is not only the package's own bytes. `environment_manifest` reads the imports out of the
   two packages' source with the parser rather than from a hand-written list that would drift, and
   records the version of every third-party distribution among them beside the interpreter and the
   platform. Without it the acceptance case "an independent analyst regenerates every result" cannot
   be met: the fits run through scipy's optimiser and numpy's generator, and a result reproduced under
   unstated versions of those has been reproduced under an unstated method.

4. AN EVIDENCE BUNDLE WITH A SCHEMA AND A BOUNDARY. `EvidenceBundle` writes the manifest and the
   sealed record with its receipt AT SEAL TIME, so a crash between the seal and the last continuation
   still leaves the commitment on disk where it can be checked; it appends a PROGRESS RECORD as the
   collection proceeds, so a crash midway through the continuation leaves the rows that were already
   collected rather than only the commitment; and it writes the complete run record at the end: the
   full bank rows, every ladder read with its subset size and pass count, every held-out replicate
   series, every P16 arm's margin series, the manifest, the seal, the receipt and the provider
   metadata. Every write passes through `redact`, which removes answer keys, hidden checks and
   anything shaped like a credential, because the bundle is the public artefact and the held-out
   material is not public. A bundle can be reloaded and its verdicts recomputed from it alone, which
   is the only test of "an independent analyst can regenerate the table" that does not rely on
   trusting a terminal summary.

5. A BOUNDARY THAT DOES NOT REST ONLY ON A NAME LIST. `redact` still removes by key name, because a
   boundary that depends on every caller remembering is not a boundary; but a held-out value under an
   unlisted key used to travel through untouched. Keys are now split into words as well as matched
   whole, so `hidden_answer` and `gold_solution` are removed although neither is in the list, and
   values are matched against the shapes credentials actually take, so a key smuggled through under
   an innocent name is removed by what it looks like. The word rule deliberately does not treat
   `check` as a word, because `checkpoints` is a registered configuration field and a boundary that
   eats the configuration destroys the evidence it was meant to protect.

6. A SENTENCE NO DIGEST CAN REPLACE. `require_scoreable` was given two of the three things the
   finding names, and the third was that the held-out material had not already been inspected when
   the predictions were chosen. No hash can answer it: a hash proves the material has not moved
   since, and says nothing about who had read it before. So the seal carries a named party's
   statement, with the digest of the material it speaks about, made before the seal and written on
   the seal BEFORE the anchor, which puts it inside the anchored digest: an attestation added,
   removed or edited afterwards makes the recomputed digest disagree with the receipt. As with the
   anchor, the only one shipped here is a placeholder that attests nothing, is labelled in three
   places, and is refused at proposition level. The runner composes no part of an attestation: not
   the digest, not the sentence and not the time it was made, because each of them would be the
   party being checked answering the question that was asked of it. A record arriving short of any
   of them is refused rather than completed, which is the repair a review of the first version
   asked for: the defaults it filled in were the two fields that carry the attester's own words.

WHAT THE JSON HERE IS NOT. The bundle uses Python's JSON extension for NaN and Infinity rather than
writing nulls. An unresolved standard error is not a missing standard error, and a bundle that turns
one into the other silently changes what a later reader can conclude.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import sys
import sysconfig
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from arc_instruments import sealing

BUNDLE_SCHEMA = "arc-evidence-bundle/1"
CODE_IDENTITY_SCHEMA = "arc-code-identity/1"
DEPENDENCY_SCHEMA = "arc-dependency-manifest/1"
# The specification hash's recipe is versioned because it changed: version 1 covered the ladder, the
# code, the configuration and the ladder identity, and left the run's own kind outside the
# commitment. A reader who finds version 1 in an old seal is reading a commitment that did not bind
# the mode, and that is a fact about that seal rather than something to be papered over.
SPEC_HASH_SCHEMA = "arc-specification-hash/2"

MOCK_ANCHOR_LABEL = ("MOCK ANCHOR: no external commitment was made. This receipt was manufactured by "
                     "the runner and proves nothing to anybody outside it. It is acceptable in a "
                     "demonstration or a pilot and is refused at proposition level.")
EXTERNAL_ANCHOR_LABEL = ("EXTERNAL ANCHOR: an identifier issued by a service outside this runner, "
                         "attesting the sealed record's hash.")

# The keys a public bundle never carries. Held-out material and credentials are removed by name on
# every write, rather than by a promise that no caller ever passes them in.
REDACTED_KEYS = frozenset({
    "answer", "answers", "answer_key", "answer_keys", "answer_sha256", "check", "checks",
    "hidden_checks", "solution", "solutions", "api_key", "apikey", "key_env_value", "secret",
    "secrets", "token", "password", "authorization", "credential", "credentials",
})
REDACTED_SUBSTRINGS = ("api_key", "secret", "password", "access_token", "bearer")
# The words that make a key held-out material whatever else the key is called. A key is split on
# every non-alphanumeric character and each part is matched whole, so `hidden_answer`, `gold_solution`
# and `expected_answers` are removed although none of them is in REDACTED_KEYS, which is the gap a
# name list always leaves. `check` and `key` are deliberately NOT words here: `checkpoints` is a
# registered configuration field and `checkpoint_store_root` is part of the run's setup, and a
# boundary that eats the configuration destroys the evidence it exists to protect. They stay as whole
# key names above, where they cannot reach either.
REDACTED_WORDS = frozenset({
    "answer", "answers", "solution", "solutions", "gold", "secret", "secrets", "password", "passwd",
    "credential", "credentials", "apikey", "authorization", "checks",
})
# A credential recognised by its shape rather than by the name it was filed under. These are the
# published prefixes of the common issuers plus the PEM header; each needs a run of characters long
# enough that ordinary prose cannot reach it by accident.
CREDENTIAL_SHAPES = tuple(re.compile(p) for p in (
    r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{20,}",
    r"\bBearer\s+[A-Za-z0-9._\-]{16,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    r"\bxox[baprs]-[A-Za-z0-9\-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
))
CREDENTIAL_PLACEHOLDER = "[redacted: a value shaped like a credential]"


class CustodyRefusal(RuntimeError):
    """The custody chain does not hold. `failures` names every check that failed, not the first."""

    def __init__(self, message: str, failures: Sequence[str] = ()):
        super().__init__(message)
        self.failures = tuple(failures)


# --------------------------------------------------------------------------------------------------
# Portable code identity
# --------------------------------------------------------------------------------------------------

def package_root() -> str:
    """The directory holding `arc_runner` and `arc_instruments`, which is what relative paths are
    relative to."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def package_code_paths(root: Optional[str] = None) -> List[str]:
    """Every source file whose bytes decide what this run did: both packages, sorted."""
    root = root or package_root()
    out = []
    for pkg in ("arc_runner", "arc_instruments"):
        d = os.path.join(root, pkg)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.endswith(".py"):
                out.append(os.path.join(d, n))
    return out


def code_identity(paths: Iterable[str], root: Optional[str] = None) -> str:
    """The portable identity of a set of source files.

    Paths enter the digest RELATIVE to `root` and with forward slashes, so the same bytes in another
    directory, on another machine or under another checkout name produce the same identity. That is
    the property finding A8 asks for: an identity an independent analyst can recompute.

    Every entry is length delimited, path length then path then byte length then bytes. Without the
    lengths the concatenation is ambiguous: a file named `a` holding `bc` and a file named `ab`
    holding `c` produce identical bytes, so a rename could be hidden by a compensating edit.
    """
    items: List[Tuple[str, bytes]] = []
    paths = list(paths)
    if root is None:
        root = os.path.commonpath([os.path.abspath(p) for p in paths]) if paths else package_root()
    for p in paths:
        rel = os.path.relpath(os.path.abspath(p), root).replace(os.sep, "/")
        with open(p, "rb") as fh:
            items.append((rel, fh.read()))
    h = hashlib.sha256()
    h.update(CODE_IDENTITY_SCHEMA.encode())
    for rel, data in sorted(items):
        rb = rel.encode("utf-8")
        h.update(str(len(rb)).encode()); h.update(b":"); h.update(rb)
        h.update(str(len(data)).encode()); h.update(b":"); h.update(data)
    return h.hexdigest()


def package_code_identity(root: Optional[str] = None) -> str:
    root = root or package_root()
    return code_identity(package_code_paths(root), root)


# --------------------------------------------------------------------------------------------------
# The dependency manifest: the other half of a portable identity
# --------------------------------------------------------------------------------------------------

_STDLIB_ROOTS = tuple(os.path.realpath(p) for p in
                      (sysconfig.get_paths().get("stdlib"), sysconfig.get_paths().get("platstdlib"))
                      if p)


def imported_top_level_names(root: Optional[str] = None) -> List[str]:
    """Every top-level module name the two packages import, read out of the source by the parser.

    Read rather than listed, because a hand-written dependency list drifts from the code the first
    time an import is added and nothing tells anybody. Relative imports are skipped: they are the
    packages' own modules, which the code identity already covers byte for byte.
    """
    names = set()
    for p in package_code_paths(root):
        try:
            with open(p, "rb") as fh:
                tree = ast.parse(fh.read(), filename=p)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:                        # a relative import is our own package
                    continue
                if node.module:
                    names.add(node.module.split(".")[0])
    return sorted(n for n in names if n not in ("arc_runner", "arc_instruments"))


def _module_file(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    return getattr(mod, "__file__", None) or ""


def _is_standard_library(name: str) -> bool:
    """A module that lives under the interpreter's own library directory and not under site-packages.

    Python 3.9 has no `sys.stdlib_module_names`, and this package runs on it, so membership is decided
    by where the module's file actually is rather than by a list that only newer interpreters carry.
    A built-in module has no file at all and is standard library by construction.
    """
    f = _module_file(name)
    if f is None:
        return False
    if f == "":
        return True                                   # built into the interpreter
    real = os.path.realpath(f)
    if "site-packages" in real or "dist-packages" in real:
        return False
    return any(real.startswith(r) for r in _STDLIB_ROOTS)


def _declared_version(name: str) -> Tuple[Optional[str], str]:
    """This distribution's version, and how it was found. A version that could not be found is
    recorded as absent with the reason, never guessed: an unknown version stated as a number is worse
    than an unknown version stated as unknown."""
    try:
        from importlib import metadata                # 3.8+, and the authority where it answers
        return str(metadata.version(name)), "importlib.metadata"
    except Exception:
        pass
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None, "the module is not importable in this environment"
    v = getattr(mod, "__version__", None)
    if v is not None:
        return str(v), "module.__version__"
    return None, "the module declares no version"


def environment_manifest(root: Optional[str] = None) -> Dict[str, Any]:
    """A fresh copy of the cached record, because the caller puts it in a manifest and a manifest is
    edited. A shared mutable behind an lru_cache is a defect waiting for the first caller who changes
    one field and silently changes it for every later run in the process."""
    return json.loads(json.dumps(_environment_manifest(root)))


@functools.lru_cache(maxsize=8)
def _environment_manifest(root: Optional[str] = None) -> Dict[str, Any]:
    """The interpreter, the platform and every third-party dependency the packages import, with the
    version of each.

    FINDING A8 ASKS FOR THIS BY NAME: "make code identity portable with relative-path,
    length-delimited entries AND A MANIFEST OF ALL REQUIRED DEPENDENCIES". The relative paths make the
    package's own bytes reproducible; they say nothing about the code the results actually went
    through. The route fits call into scipy's optimiser and every resampling stream comes from numpy's
    generator, so a table regenerated under unstated versions of those has been regenerated under an
    unstated method, and the acceptance case "an independent analyst regenerates every result" is not
    met by the package hash alone.

    Cached on the root because a run builds several manifests and the answer cannot change inside one
    process: the source is read once and the installed distributions do not move under a running
    interpreter.
    """
    root = root or package_root()
    deps: Dict[str, Any] = {}
    for name in imported_top_level_names(root):
        if _is_standard_library(name):
            continue
        version, source = _declared_version(name)
        deps[name] = {"version": version, "version_source": source}
    record = {
        "schema": DEPENDENCY_SCHEMA,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full_version": sys.version.replace("\n", " "),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "dependencies": deps,
        "discovered_by": "the import statements of arc_runner and arc_instruments, read with ast",
        "note": "the platform and the interpreter are recorded beside the dependencies because a "
                "numerical result reproduced on another platform is reproduced, and one that differs "
                "there is a fact worth having rather than a mystery",
    }
    record["environment_sha256"] = sealing.sha256_of(
        {k: v for k, v in record.items() if k != "note"})
    return record


def environment_differences(recorded: Optional[Dict[str, Any]],
                            live: Optional[Dict[str, Any]] = None) -> List[str]:
    """How the environment in hand differs from the one the run recorded, one line per difference.

    CONSERVATIVE READING, NAMED AS OPEN. The contract requires the dependency manifest to be RECORDED
    and does not say that a later analyst running a different scipy may not re-score. The reading
    taken here is that a difference is reported and never silently refused: it is off by default in
    `custody_failures` and is asked for by an analyst who wants to know. If the author settles that a
    deciding run must also refuse to be re-scored under changed dependencies, this becomes a default
    and only that flag changes.
    """
    if not recorded:
        return ["no dependency manifest was recorded for this run, so the environment it ran in "
                "cannot be compared with anything"]
    live = live or environment_manifest()
    out: List[str] = []
    for field_name in ("python_version", "platform_system", "platform_machine"):
        if recorded.get(field_name) != live.get(field_name):
            out.append("%s is %r here and was %r when the run was made"
                       % (field_name, live.get(field_name), recorded.get(field_name)))
    rec_deps = recorded.get("dependencies") or {}
    live_deps = live.get("dependencies") or {}
    for name in sorted(set(rec_deps) | set(live_deps)):
        if name not in live_deps:
            out.append("the dependency %r the run recorded is not imported by the code here" % name)
        elif name not in rec_deps:
            out.append("the dependency %r is imported here and was not in the run's manifest" % name)
        elif (rec_deps[name] or {}).get("version") != (live_deps[name] or {}).get("version"):
            out.append("the dependency %r is version %r here and was %r when the run was made"
                       % (name, (live_deps[name] or {}).get("version"),
                          (rec_deps[name] or {}).get("version")))
    return out


def _verifier_callables(ladder: Any) -> List[Tuple[str, Any]]:
    """Everything that decides whether an item passed. A ladder that reads a simulated latent
    capability has none, which is itself recorded rather than left blank."""
    out: List[Tuple[str, Any]] = []
    for name in ("verifier", "batch_runner"):
        fn = getattr(ladder, name, None)
        if callable(fn):
            out.append((name, fn))
    mapping = getattr(ladder, "verifiers", None)
    if isinstance(mapping, dict):
        for k in sorted(mapping):
            if callable(mapping[k]):
                out.append(("verifier:" + str(k), mapping[k]))
    return out


def _source_of(fn: Any) -> Optional[str]:
    """The source that decides this callable's behaviour: its defining module where one can be read,
    otherwise the callable's own source. A closure's behaviour is its module's behaviour, which is
    why the module comes first."""
    mod = inspect.getmodule(fn)
    for get in (lambda: inspect.getsource(mod) if mod is not None else None,
                lambda: inspect.getsource(fn)):
        try:
            src = get()
        except (OSError, TypeError):
            src = None
        if src:
            return src
    return None


def verifier_identity(ladder: Any) -> Dict[str, Any]:
    """A hash binding the implementations that decide items, not only the pool that names them.

    Finding A8: a ladder spec says which items were asked. Two runs can ask the same items and count
    different things as passing them, and the pool hash cannot tell them apart. The qualified name
    enters the digest beside the source so that two verifiers living in one module are distinguished
    from each other.
    """
    entries = []
    h = hashlib.sha256()
    h.update(b"arc-verifier-identity/1")
    for name, fn in _verifier_callables(ladder):
        src = _source_of(fn)
        qual = getattr(fn, "__qualname__", None) or type(fn).__name__
        module = getattr(inspect.getmodule(fn), "__name__", None)
        src_sha = hashlib.sha256(src.encode("utf-8")).hexdigest() if src else None
        entries.append({"slot": name, "qualname": qual, "module": module, "source_sha256": src_sha,
                        "bound": bool(src_sha),
                        # WHAT THIS CHECK DECLARED ITSELF TO BE (finding A4). The source digest says
                        # which implementation decided the items and cannot say whether its author
                        # held it out as a measurement or as a wiring smoke test. Both marks are read
                        # through a partial, a wrapper and a bound method, and both are recorded so
                        # that a reader of the bundle sees what the gate saw. They are deliberately
                        # not folded into the digest below: the mark lives in the source that is
                        # already hashed, and hashing it twice would make a comment on a check look
                        # like a change of check.
                        "attests_exact_check": _marked(fn, "exact_check"),
                        "declares_substring_smoke_test": _marked(fn, "substring_smoke_test")})
        for part in (name, qual, module or "", src_sha or "unbound"):
            pb = part.encode("utf-8")
            h.update(str(len(pb)).encode()); h.update(b":"); h.update(pb)
    return {"verifier_sha256": h.hexdigest() if entries else None, "verifiers": entries,
            "all_bound": bool(entries) and all(e["bound"] for e in entries)}


def attest_exact_check(fn: Any) -> Any:
    """Mark a callable as a check that decides whether an item was SOLVED, and return it.

    WHY A POSITIVE ATTESTATION EXISTS BESIDE THE NEGATIVE MARKER (finding A4). The first repair marked
    the substring check and refused anything carrying that mark. The mark lives on the callable, and
    the reasoning was that it therefore travels wherever the callable is reused. That is false for
    every ordinary way of reusing a callable: `functools.partial(ladder.contains_answer)` and a
    one-line `lambda t, i: contains_answer(t, i)` are both new objects carrying no mark, and either
    one put a registered pool scored by a substring check straight through the gate. `_marker_chain`
    now follows a partial, a `functools.wraps` wrapper and a bound method to what they delegate to, so
    those two evasions are closed; an opaque wrapper that calls the marked function in its body is not
    reachable that way, and no reading of the wrapper's source would be reliable either.

    So the deciding path stops asking whether a check declares itself unfit and asks whether it
    declares itself fit. An unattested callable is refused, and wrapping an attested check in a lambda
    produces an unattested callable: the wrapper's author re-attests, which is the deliberate act the
    custody of a deciding run is made of. Silence fails closed, as it does everywhere else here.
    """
    fn.exact_check = True
    return fn


def mark_substring_smoke_test(fn: Any) -> Any:
    """Mark a callable as a wiring smoke test that is not a measurement of whether a task was solved,
    and return it. Refused on the deciding path wherever it is attached, and through a partial or a
    wrapper; see `attest_exact_check` for why the negative mark alone is not enough."""
    fn.substring_smoke_test = True
    return fn


def _marker_chain(fn: Any, limit: int = 16) -> List[Any]:
    """`fn` and everything it visibly delegates its behaviour to.

    A marker read off the outermost object alone is read off whatever wrapper happened to be applied
    last, which is how `functools.partial(contains_answer)` escaped the substring refusal. The chain
    follows the three delegations that are visible without guessing: a partial's function, a
    `functools.wraps` wrapper's `__wrapped__`, and a bound method's underlying function. `limit` stops
    a self-referential wrapper from looping.
    """
    chain: List[Any] = []
    seen: set = set()
    cur = fn
    while cur is not None and id(cur) not in seen and len(chain) < limit:
        seen.add(id(cur))
        chain.append(cur)
        if isinstance(cur, functools.partial):
            cur = cur.func
        elif getattr(cur, "__wrapped__", None) is not None:
            cur = cur.__wrapped__
        elif inspect.ismethod(cur):
            cur = cur.__func__
        else:
            cur = None
    return chain


def _marked(fn: Any, attribute: str) -> bool:
    """True when `fn` or anything it delegates to carries the marker."""
    return any(bool(getattr(link, attribute, False)) for link in _marker_chain(fn))


def smoke_verifiers(ladder: Any) -> List[str]:
    """The slots of this ladder whose verifier declares itself a smoke test rather than a measurement.

    Finding A4: the reference arithmetic ladder was scored by a substring check, which a response
    enumerating candidate answers satisfies without solving anything. Barring the reference POOL from
    the deciding path does not bar the CHECK, because a substring verifier attached to a registered
    pool is the same defect with a better pool behind it. The marker is read through a partial, a
    wrapper and a bound method, because a mark read off the outermost object alone is defeated by
    binding one option to the check.
    """
    return [name for name, fn in _verifier_callables(ladder)
            if _marked(fn, "substring_smoke_test")]


def unattested_verifiers(ladder: Any) -> List[str]:
    """The slots of this ladder whose check does not attest that it decides whether an item was solved.

    The complement of `smoke_verifiers` and the reason that list is not enough on its own: a wrapper
    around a substring check carries neither mark, and a deciding run cannot tell an unmarked wrapper
    from an unmarked measurement. See `attest_exact_check`.
    """
    return [name for name, fn in _verifier_callables(ladder)
            if not _marked(fn, "exact_check")]


def ladder_identity(ladder: Any) -> Dict[str, Any]:
    """The ladder as the seal should carry it: its own hash, its spec, and for a checkable ladder the
    verifier binding. A ladder passed as a bare hash string is recorded as exactly that and nothing
    is invented for the fields that were never supplied."""
    if ladder is None:
        return {"ladder_sha256": None, "spec": None, "spec_sha256": None, "verifier_sha256": None,
                "verifiers": [], "all_bound": False, "simulated": None, "smoke_only": None,
                "note": "no ladder object was supplied to this run"}
    if isinstance(ladder, str):
        return {"ladder_sha256": ladder, "spec": None, "spec_sha256": None, "verifier_sha256": None,
                "verifiers": [], "all_bound": False, "simulated": None, "smoke_only": None,
                "note": "the ladder was supplied as a hash only, so no verifier could be bound"}
    spec = getattr(ladder, "spec", None)
    vid = verifier_identity(ladder)
    return {"ladder_sha256": getattr(ladder, "sha256", None), "spec": spec,
            "spec_sha256": sealing.sha256_of(spec) if spec is not None else None,
            "simulated": bool(getattr(ladder, "simulated", False)),
            "smoke_only": bool(getattr(ladder, "smoke_only", False)), **vid}


def same_ladder(one: Any, other: Any) -> bool:
    """Whether two ladders are the same ladder for every purpose this package records.

    Identity and not object identity: two objects built over one frozen pool with one set of
    verifiers are interchangeable in the manifest, the seal and every check, and a caller who builds
    the object twice has not changed the setup. Comparison is on canonical bytes for the reason
    `ladder_differences` uses them, so a tuple that came back from a file as a list is not read as a
    difference.
    """
    return (sealing.canonical_bytes(ladder_identity(one))
            == sealing.canonical_bytes(ladder_identity(other)))


def attach_read_log(ladder: Any) -> Optional[List[Dict[str, Any]]]:
    """Turn on per-read recording for this ladder and return the list it writes into.

    Off by default: a run that is not saving evidence should not pay for a list that grows with every
    read, and a ladder shared between runs must not accumulate another run's reads.
    """
    if ladder is None or not hasattr(ladder, "read_log"):
        return None
    ladder.read_log = []
    return ladder.read_log


def adapter_metadata(adapter: Any) -> Dict[str, Any]:
    """What the provider said, kept rather than discarded.

    Finding A8: the real adapter returned the response text and threw away the returned model
    identifier and the usage figures, so a saved run could not show which model answered or what it
    consumed. An adapter that declares `metadata()` is asked; one that does not is described by what
    it is willing to say about itself, and never by a guess.
    """
    if adapter is None:
        return {"adapter": None, "note": "no adapter was supplied"}
    fn = getattr(adapter, "metadata", None)
    if callable(fn):
        try:
            return dict(fn())
        except Exception as exc:                      # an adapter's bookkeeping never stops a run
            return {"adapter": getattr(adapter, "name", type(adapter).__name__),
                    "note": "the adapter's metadata() raised: %s" % exc}
    return {"adapter": getattr(adapter, "name", type(adapter).__name__),
            "records": list(getattr(adapter, "provider_metadata", []) or []),
            "note": "this adapter declares no metadata(); only what it exposed is recorded"}


# --------------------------------------------------------------------------------------------------
# The anchor receipt
# --------------------------------------------------------------------------------------------------

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def receipt(identifier: str, attests_sha256: str, service: str, issued_utc: Optional[str] = None,
            mock: bool = False, note: str = "") -> Dict[str, Any]:
    """One receipt: an identifier, the sha256 it attests, and who issued it. Nothing else is a
    receipt: an identifier with no digest attests nothing, and a digest with no identifier cannot be
    looked up by anyone."""
    if not identifier or not attests_sha256:
        raise CustodyRefusal("a receipt needs both an identifier and the sha256 it attests",
                             ("anchor-receipt",))
    return {"anchor_identifier": str(identifier), "attests_sha256": str(attests_sha256),
            "service": str(service), "issued_utc": issued_utc or utc_now(), "mock": bool(mock),
            "label": MOCK_ANCHOR_LABEL if mock else EXTERNAL_ANCHOR_LABEL, "note": note}


def mock_anchor(digest: str) -> Dict[str, Any]:
    """The only anchor this package ships, and it is a mock.

    It exists so that a demonstration and a pilot carry a receipt in the same shape a deciding run
    carries, and so that the shape is exercised by every test rather than only by the one run that
    matters. It is labelled a mock in three places because the failure mode being prevented is a
    reader taking a rehearsal's receipt for a commitment.
    """
    return receipt("mock-anchor:" + str(digest)[:16], digest,
                   service="none (no external anchor was performed)", mock=True,
                   note="manufactured by arc_runner.custody.mock_anchor")


def seal_digest(seal: Dict[str, Any]) -> str:
    """The value an anchor attests: the sealed record's own hash, taken over everything the sealer
    committed to and excluding the fields the anchoring step adds afterwards. Anchoring the
    predictions hash alone would leave the specification hash and the sealing time unattested."""
    body = {k: v for k, v in seal.items() if k not in ("anchor_receipt", "seal_sha256")}
    return sealing.sha256_of(body)


def anchor_seal(manifest: Dict[str, Any], anchor: Optional[Callable[[str], Dict[str, Any]]] = None,
                ) -> Dict[str, Any]:
    """Hand the seal's digest to the operator's anchoring service and record what comes back.

    An absent service is not an error here and is not silently ignored either: a mock receipt is
    attached and labelled, and `require_scoreable` refuses it at proposition level. The alternative,
    leaving the seal unreceipted, produces a run that looks finished and cannot be checked.
    """
    seal = manifest.get("seal")
    if not seal:
        raise CustodyRefusal("there is no seal to anchor", ("anchor-receipt",))
    digest = seal_digest(seal)
    rec = (anchor or mock_anchor)(digest)
    if not isinstance(rec, dict):
        raise CustodyRefusal("the anchoring service returned %r rather than a receipt"
                             % type(rec).__name__, ("anchor-receipt",))
    rec = dict(rec)
    if rec.get("attests_sha256") != digest:
        raise CustodyRefusal(
            "the anchoring service attested %r, which is not this seal's digest %r; a receipt for "
            "another document is not a receipt for this one"
            % (rec.get("attests_sha256"), digest), ("anchor-receipt",))
    if not rec.get("anchor_identifier"):
        raise CustodyRefusal("the receipt carries no identifier, so nobody can look it up",
                             ("anchor-receipt",))
    rec.setdefault("mock", False)
    rec.setdefault("label", MOCK_ANCHOR_LABEL if rec["mock"] else EXTERNAL_ANCHOR_LABEL)
    seal["seal_sha256"] = digest
    seal["anchor_receipt"] = rec
    manifest["anchor_identifier"] = rec["anchor_identifier"]
    return rec


# --------------------------------------------------------------------------------------------------
# The prior-inspection attestation
# --------------------------------------------------------------------------------------------------

HELDOUT_ATTESTATION_SCHEMA = "arc-heldout-attestation/1"

UNSEEN_STATEMENT = (
    "The held-out material whose digest is recorded here had not been inspected, in whole or in "
    "part, by anybody who chose the predictions or the configuration this seal covers, at any time "
    "before this seal was taken.")

MOCK_ATTESTATION_LABEL = (
    "PLACEHOLDER ATTESTATION: nobody attested anything. This record was manufactured by the runner "
    "so that the shape a deciding run must carry is present in every run, and it says nothing "
    "whatever about what was or was not inspected. It is honest in a demonstration or a pilot and "
    "is refused at proposition level.")
ATTESTATION_LABEL = (
    "ATTESTATION: a named party states that the material whose digest is recorded here was unseen "
    "when the predictions this seal covers were fixed.")


def attestation(attester: str, heldout_sha256: str, material: str = "",
                statement: str = UNSEEN_STATEMENT, attested_utc: Optional[str] = None,
                note: str = "") -> Dict[str, Any]:
    """One attestation: who says it, which material they are speaking about, and the sentence.

    THE THIRD QUESTION, WHICH NO DIGEST CAN ANSWER. Finding A8 names three things `require_scoreable`
    did not do, and the first two are done by the anchor receipt and the recomputed specification
    hash: were these the predictions that were sealed, and is this the configuration they were sealed
    under. The third is whether the material that decides them had already been looked at when they
    were chosen, and nothing computable can settle it. A hash proves that material has not moved
    since it was hashed; it says nothing about who had read it before. Only a person can say that, so
    this record is a person's statement, recorded verbatim, and this module composes none: the one
    attestation it ships is `mock_attestation`, which attests nothing and says so.

    AND WHY THIS CONSTRUCTOR MAY HOLD A DEFAULT SENTENCE WHERE `attach_attestation` MAY NOT. The two
    look like the same act and are not. Whoever calls this is the attester: they adopt the shipped
    form of words by calling it, they may pass their own instead, and the time recorded is the moment
    they made the record, which is a true statement about it. `attach_attestation` receives a record
    somebody else built, possibly read out of a file or returned by a module named on the command
    line, and a field filled in there is this code finishing another party's sentence and dating it
    for them. The first is a form of words being signed; the second was this package writing the
    answer to the question it exists to ask, which is why it now refuses instead.

    THE DIGEST IS WHAT MAKES THE SENTENCE CHECKABLE. A statement naming no material attests nothing,
    because anything at all can be said not to have been read. With the digest, a scorer holding the
    material can ask whether the artefact attested is the artefact in hand, and `custody_failures`
    refuses when the two disagree. `material` is the attester's own description of what they are
    speaking about, kept beside it because a digest is not a name and a reader of a public bundle
    cannot recompute one from material they do not hold.
    """
    if not attester or not heldout_sha256:
        raise CustodyRefusal(
            "an attestation needs the party making it and the sha256 of the material it speaks "
            "about: a sentence with neither attests nothing to anybody", ("prior-inspection",))
    if not statement:
        raise CustodyRefusal("an attestation with no statement states nothing", ("prior-inspection",))
    return {"schema": HELDOUT_ATTESTATION_SCHEMA, "attester": str(attester),
            "heldout_sha256": str(heldout_sha256), "material": str(material),
            "statement": str(statement), "attested_utc": attested_utc or utc_now(),
            "mock": False, "label": ATTESTATION_LABEL, "note": note}


def mock_attestation(note: str = "") -> Dict[str, Any]:
    """The only attestation this package ships, and it attests nothing.

    It exists for the reason `mock_anchor` does: a demonstration and a pilot carry the shape a
    deciding run carries, so the shape is exercised by every test rather than by the one run that
    matters, and the reader is told in three places that it is a placeholder. It carries NO digest,
    because the runner does not know which material anybody did or did not read, and a digest written
    in here by the code would be the party being checked answering the question that was asked of it.
    """
    return {"schema": HELDOUT_ATTESTATION_SCHEMA, "attester": None, "heldout_sha256": None,
            "material": None, "statement": UNSEEN_STATEMENT, "attested_utc": utc_now(),
            "mock": True, "label": MOCK_ATTESTATION_LABEL,
            "note": note or "manufactured by arc_runner.custody.mock_attestation"}


def attach_attestation(manifest: Dict[str, Any], record: Optional[Dict[str, Any]] = None,
                       ) -> Dict[str, Any]:
    """Record the attestation on the seal, BEFORE the seal is anchored.

    The order is the whole of its strength. `seal_digest` covers everything the sealer committed to
    and excludes only the fields the anchoring step adds afterwards, so an attestation written here
    is inside the digest the receipt attests: one added, removed or edited after the anchor makes the
    recomputed digest disagree with the receipt and refuses. An attestation kept beside the seal
    instead would be a claim anybody could write at any time, including after the material had been
    read.

    An absent record is not an error and is not silently nothing either: the placeholder is attached
    and labelled, exactly as an absent anchoring service produces a mock receipt, and the deciding
    path refuses it. An INCOMPLETE record is a third thing and is refused outright, in every mode:
    the runner cannot supply what is missing from it without becoming its author, and a run that
    could not attach an attestation is better stopped here, before the seal, than sealed with one
    this module wrote and a reader would read as somebody's word.
    """
    seal = manifest.get("seal")
    if not seal:
        raise CustodyRefusal("there is no seal to attest", ("prior-inspection",))
    if record is None:
        rec = mock_attestation()
    elif not isinstance(record, dict):
        raise CustodyRefusal("an attestation is a record and %r is not one; build one with "
                             "arc_runner.custody.attestation" % type(record).__name__,
                             ("prior-inspection",))
    else:
        rec = dict(record)
        rec.setdefault("schema", HELDOUT_ATTESTATION_SCHEMA)
        rec.setdefault("mock", False)
        rec.setdefault("label", MOCK_ATTESTATION_LABEL if rec["mock"] else ATTESTATION_LABEL)
        # AND NOTHING BELOW IS FILLED IN, WHICH IS THE REPAIR. The statement and the time were
        # `setdefault` calls here, and both are the attester's own words: a record carrying an
        # attester and a digest and nothing else reached the seal with the shipped sentence written
        # into it and a time this module had read off its own clock, and `custody_failures` then
        # reported it as a named party's attestation, because its one statement check can only fire
        # on a record this line had already completed. That is the act the constructor above refuses
        # when it is handed a blank sentence, and it was reachable from the shipped command line,
        # where `--attestation-module` hands whatever the module returns straight to the run. A
        # record short of any of the four is refused instead, which is the only answer this code may
        # give: what is missing belongs to somebody outside it.
        if not rec["mock"]:
            absent = [f for f in ("attester", "heldout_sha256", "statement", "attested_utc")
                      if not rec.get(f)]
            if absent:
                raise CustodyRefusal(
                    "an attestation needs the party making it, the sha256 of the material it speaks "
                    "about, the sentence they are making and the time they made it; this one is "
                    "missing %s. None of the four is written in here: a sentence composed by this "
                    "code, or a time it read off its own clock, is the party being checked answering "
                    "a question that was asked of somebody else. Build the record with "
                    "arc_runner.custody.attestation" % ", ".join(absent), ("prior-inspection",))
    seal["heldout_attestation"] = rec
    return rec


# --------------------------------------------------------------------------------------------------
# Verification at scoring time
# --------------------------------------------------------------------------------------------------

def config_record(config: Any) -> Dict[str, Any]:
    """A configuration as a plain mapping, whether it arrived as a dataclass or as JSON."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    return dict(getattr(config, "__dict__", {}) or {})


def config_differences(sealed: Dict[str, Any], live: Any) -> List[str]:
    """Which fields of the live configuration disagree with the one the seal was made under.

    Only fields the live object has are compared: the sealed record also carries the seed, which the
    configuration object does not, and a missing key on the live side is not a change. Comparison is
    on canonical JSON so that a tuple that came back from a file as a list is not read as a change.
    """
    out = []
    for k, v in config_record(live).items():
        if k not in sealed:
            out.append("configuration field %r was not in the sealed configuration" % k)
            continue
        try:
            same = sealing.canonical_bytes(v) == sealing.canonical_bytes(sealed[k])
        except (TypeError, ValueError):
            same = repr(v) == repr(sealed[k])
        if not same:
            out.append("configuration field %r changed after sealing" % k)
    return out


def custody_failures(manifest: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None,
                     config: Any = None, external_anchor_required: bool = False,
                     verify_code: bool = False, ladder: Any = None,
                     verify_environment: bool = False,
                     heldout_sha256: Optional[str] = None) -> List[str]:
    """Every custody check that fails, in one pass, so that a setup is fixed once and not four times.

    `heldout_sha256` is the digest of the held-out material the scorer holds, where the scorer holds
    it. Supplied, it is compared with the digest the prior-inspection attestation speaks about, and a
    disagreement refuses: an attestation for another artefact is not an attestation for this one. Not
    supplied, the comparison is not made and is reported as not made, because the runner does not
    decide what an attester was speaking about."""
    out: List[str] = []
    seal = manifest.get("seal")
    if not seal:
        return ["no seal on this manifest; a verdict without a prior seal is not a registered result"]

    rec = seal.get("anchor_receipt")
    if not rec:
        out.append("no anchor receipt on the seal; the commitment was never attested by anything "
                   "outside this runner")
    else:
        digest = seal_digest(seal)
        if rec.get("attests_sha256") != digest:
            out.append("the anchor receipt attests %r but the seal's digest is now %r; the sealed "
                       "record changed after it was anchored"
                       % (rec.get("attests_sha256"), digest))
        if not rec.get("anchor_identifier"):
            out.append("the anchor receipt carries no identifier")
        if external_anchor_required and rec.get("mock"):
            out.append("the anchor receipt is a mock (%s); a deciding run needs an anchor issued "
                       "outside this runner" % rec.get("anchor_identifier"))

    # THE THIRD QUESTION (finding A8). The receipt says the sealed record has not moved and the
    # specification hash says the configuration has not; neither says whether the material deciding
    # these predictions had already been read when they were chosen. Nothing computable can say it,
    # so what is required is a named party's sentence, made before the seal and anchored with it. The
    # absence of the record is a failure in every mode, because `seal_predictions` attaches at least
    # the placeholder to every seal it makes: a seal reaching here without one was assembled by hand.
    att = seal.get("heldout_attestation")
    if not att:
        out.append("no prior-inspection attestation on the seal; nothing states that the material "
                   "deciding these predictions was unseen when they were fixed. Build one with "
                   "arc_runner.custody.attestation and pass it to the run as attestation=, which is "
                   "the only way one is ever written: this code composes none")
    else:
        if not att.get("statement"):
            out.append("the prior-inspection attestation carries no statement, so nothing was "
                       "attested by it")
        if external_anchor_required:
            if att.get("mock"):
                out.append("the prior-inspection attestation is the runner's own placeholder, which "
                           "attests nothing to anybody outside it; a deciding run needs the sentence "
                           "of a named party outside this code")
            else:
                if not att.get("attester"):
                    out.append("the prior-inspection attestation names no attester, so nobody stands "
                               "behind it")
                if not att.get("heldout_sha256"):
                    out.append("the prior-inspection attestation names no material: a sentence that "
                               "does not say which material was unseen cannot be checked against the "
                               "material in hand")
                if not att.get("attested_utc"):
                    # WHEN IT WAS SAID IS PART OF WHAT IS SAID. The requirement is that the material
                    # was unseen when the predictions were fixed, so an undated sentence cannot be
                    # placed before or after the thing it speaks about. `attach_attestation` refuses
                    # to date one itself, so a record reaching here without a time was assembled by
                    # hand, exactly as an attestation reaching here at all without one was.
                    out.append("the prior-inspection attestation carries no time, so nothing says "
                               "when it was made, and a sentence that cannot be placed before the "
                               "seal cannot say the material was unseen when the predictions were "
                               "fixed")
        if heldout_sha256 is not None and str(heldout_sha256) != att.get("heldout_sha256"):
            out.append("the prior-inspection attestation speaks about the material whose digest is "
                       "%s and the material in hand hashes to %s; an attestation for another "
                       "artefact is not an attestation for this one"
                       % (str(att.get("heldout_sha256"))[:16], str(heldout_sha256)[:16]))

    # The specification hash is recomputed from the manifest's own fields. If the configuration, the
    # ladder hash or the code identity has been edited since the seal was made, the recomputation
    # disagrees and the run refuses. This is the check that catches a configuration moved to fit a
    # result, which is the defect finding A8 names.
    expect = spec_hash_of(manifest)
    if seal.get("spec_sha256") != expect:
        out.append("the specification hash recomputed from this manifest (%s) is not the one sealed "
                   "(%s); the ladder, the code identity or the configuration changed after sealing"
                   % (expect[:16], str(seal.get("spec_sha256"))[:16]))

    if predictions is not None:
        got = sealing.sha256_of(predictions)
        if got != seal.get("predictions_sha256"):
            out.append("the predictions being scored hash to %s and the sealed predictions hash to "
                       "%s; these are not the same predictions"
                       % (got[:16], str(seal.get("predictions_sha256"))[:16]))

    if config is not None:
        out += config_differences(manifest.get("config") or {}, config)

    # THE MANIFEST'S TWO LADDER FIELDS, AGAINST EACH OTHER. `ladder_sha256` is the pool a reader of
    # the manifest reads first, and `ladder_identity` is what the verifier binding is recorded
    # against. They were only ever compared with a live object and never with one another, and while
    # the identity was derived from that same string they agreed by construction; a caller holding
    # the object now supplies both, so a run naming one pool at the top and handing another object
    # would seal the disagreement and leave no reader able to say which pool had been read. Both
    # fields are inside `spec_hash_of`, so the seal preserves the contradiction rather than catching
    # it.
    ident = manifest.get("ladder_identity") or {}
    if ident.get("ladder_sha256") and manifest.get("ladder_sha256") \
            and ident["ladder_sha256"] != manifest["ladder_sha256"]:
        out.append("this manifest records the ladder %s and its ladder identity is of %s: the two "
                   "name different pools, and nothing in the record says which one was read"
                   % (str(manifest["ladder_sha256"])[:16], str(ident["ladder_sha256"])[:16]))

    # THE LIVE LADDER, RECOMPUTED RATHER THAN READ BACK. A ladder whose checking rule was replaced
    # after the seal leaves the recorded identity untouched, so the specification hash cannot see it
    # and only a recomputation from the object in hand can.
    out += ladder_differences(manifest, ladder)

    if verify_code:
        live = package_code_identity()
        if manifest.get("code_sha256") and live != manifest["code_sha256"]:
            out.append("the analysis code on disk (%s) is not the code this run recorded (%s)"
                       % (live[:16], str(manifest["code_sha256"])[:16]))

    if verify_environment:
        out += ["the environment here is not the one the run recorded: %s" % d
                for d in environment_differences(manifest.get("environment"))]
    return out


def spec_hash_of(manifest: Dict[str, Any]) -> str:
    """The one recipe for the sealed specification hash, used to make it and to check it.

    It is here rather than in `manifest.py` so that the making and the checking cannot drift apart:
    two copies of this recipe would eventually differ, and the check would then pass for the wrong
    reason.

    WHAT THE RECIPE COVERS, AND WHY THE RUN'S KIND IS IN IT. Version 1 hashed the ladder, the code,
    the configuration and the ladder identity. That left `execution_mode` and everything that travels
    with it outside the commitment, and the consequence was reachable with the shipped command line: a
    demonstration against the simulated system takes a genuine anchor receipt from the operator's
    service, and rewriting two fields in its own bundle afterwards turns it into a confirmatory run
    that passes every custody check. Nothing in the chain noticed, because nothing in the chain had
    ever been shown the mode. The kind of run is the first thing a commitment must fix, since it
    decides whether the commitment may be read at all, so `experiment`, `execution_mode`, `pilot`,
    `simulated`, `scoreable_at_proposition_level`, `mode_label`, the adapter and the whole
    `confirmatory_inputs` record are inside it now, along with the environment the run ran in.

    Every one of these fields exists in the manifest BEFORE the seal is taken, which is what makes
    them sealable: `new_manifest` writes all of them and `seal_predictions` runs afterwards.
    """
    return sealing.sha256_of({"schema": SPEC_HASH_SCHEMA,
                              "ladder": manifest.get("ladder_sha256"),
                              "code": manifest.get("code_sha256"),
                              "config": manifest.get("config"),
                              "ladder_identity": manifest.get("ladder_identity"),
                              "experiment": manifest.get("experiment"),
                              "execution_mode": manifest.get("execution_mode"),
                              "pilot": bool(manifest.get("pilot")),
                              "simulated": bool(manifest.get("simulated")),
                              "scoreable_at_proposition_level":
                                  bool(manifest.get("scoreable_at_proposition_level")),
                              "mode_label": manifest.get("mode_label"),
                              "adapter": manifest.get("adapter"),
                              "confirmatory_inputs": manifest.get("confirmatory_inputs"),
                              "environment": manifest.get("environment")})


def ladder_differences(manifest: Dict[str, Any], ladder: Any) -> List[str]:
    """How the ladder in hand differs from the one the manifest recorded, recomputed and not read.

    THE DEFECT THIS CLOSES. The ladder identity was computed once, when the manifest was made, and
    was thereafter only ever compared with ITSELF through the specification hash. That catches
    somebody hand-editing the manifest's stored record and catches nothing else: sealing under one
    verifier and then swapping the ladder's own checking rule for a permissive one left the recorded
    identity untouched, the specification hash intact and every custody check passing, which is a
    documentary binding and not an enforced one. The recorded identity is therefore recomputed here
    from the live object at scoring time.

    A caller with no ladder in hand, which is every re-scoring from a saved bundle, passes None and
    gets no differences: the check is reported as not performed rather than reported as passed, and
    `require_custody` says which of the two happened.
    """
    if ladder is None:
        return []
    recorded = manifest.get("ladder_identity")
    if not recorded:
        return ["this manifest carries no ladder identity, so the live ladder cannot be compared "
                "with the one that was sealed"]
    live = ladder_identity(ladder)
    if sealing.canonical_bytes(live) == sealing.canonical_bytes(recorded):
        return []
    out = []
    if live.get("ladder_sha256") != recorded.get("ladder_sha256"):
        out.append("the live ladder hashes to %s and the sealed ladder hashed to %s"
                   % (str(live.get("ladder_sha256"))[:16], str(recorded.get("ladder_sha256"))[:16]))
    if live.get("verifier_sha256") != recorded.get("verifier_sha256"):
        out.append("the live ladder's verifier implementations hash to %s and the sealed ones hashed "
                   "to %s; the pool may be identical and what counts as passing it is not"
                   % (str(live.get("verifier_sha256"))[:16],
                      str(recorded.get("verifier_sha256"))[:16]))
    if not out:
        out.append("the live ladder's identity differs from the sealed one in a field other than the "
                   "pool hash or the verifier hash")
    return out


def require_custody(manifest: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None,
                    config: Any = None, external_anchor_required: bool = False,
                    verify_code: bool = False, ladder: Any = None,
                    verify_environment: bool = False,
                    heldout_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Refuse unless the custody chain holds. Returns the report when it does.

    The report says WHICH checks were performed and not only that the chain held, because a check
    that could not run and a check that passed are different statements and a reader of the report is
    entitled to tell them apart. `ladder_checked` is false for every re-scoring from a saved bundle,
    where no ladder object exists to recompute.
    """
    failures = custody_failures(manifest, predictions, config, external_anchor_required, verify_code,
                               ladder=ladder, verify_environment=verify_environment,
                               heldout_sha256=heldout_sha256)
    if failures:
        raise CustodyRefusal("custody is not established: %d check(s) failed.\n  - %s"
                             % (len(failures), "\n  - ".join(failures)), failures)
    seal = manifest["seal"]
    rec = seal.get("anchor_receipt") or {}
    att = seal.get("heldout_attestation") or {}
    return {"verified": True, "seal_sha256": seal_digest(seal),
            "anchor_identifier": rec.get("anchor_identifier"), "anchor_is_mock": bool(rec.get("mock")),
            "predictions_checked": predictions is not None, "configuration_checked": config is not None,
            "code_checked": bool(verify_code), "ladder_checked": ladder is not None,
            "environment_checked": bool(verify_environment),
            # WHO SAID THE MATERIAL WAS UNSEEN, AND WHETHER ANYBODY DID. A placeholder and a person
            # are different statements, and so are a digest that was compared with the material in
            # hand and a digest that nothing here could compare with anything.
            "prior_inspection_attested": bool(att and not att.get("mock")),
            "attester": att.get("attester"),
            "heldout_material_checked": heldout_sha256 is not None,
            # The mode is inside the sealed specification hash, which was recomputed above, so saying
            # so here is a report of what held rather than a claim made beside it.
            "execution_mode_sealed": manifest.get("execution_mode")}


# --------------------------------------------------------------------------------------------------
# Redaction: the boundary between the public bundle and the held-out material
# --------------------------------------------------------------------------------------------------

def _redacted_key(key: str) -> bool:
    """Whole name, substring, or any WORD of the name.

    The word rule is the part that closes the gap a name list always leaves: `hidden_answer` and
    `gold_solution` are neither in REDACTED_KEYS nor caught by any substring, and both are held-out
    material. Splitting on non-alphanumerics and matching each part whole is what keeps `checkpoints`
    and `checkpoint_store_root` out of it, which matters because both are registered configuration
    and removing them would break the specification hash the boundary exists to protect.
    """
    k = str(key).lower()
    if k in REDACTED_KEYS or any(s in k for s in REDACTED_SUBSTRINGS):
        return True
    return any(w in REDACTED_WORDS for w in re.split(r"[^a-z0-9]+", k) if w)


def _redacted_value(value: Any) -> bool:
    """A credential recognised by its shape rather than by the key it arrived under.

    A key list cannot see a token filed under `note` or appended to a URL, and the bundle is the
    public artefact. The shapes are the issuers' published prefixes with a minimum run length, so
    ordinary prose cannot reach them by accident.
    """
    return isinstance(value, str) and any(p.search(value) for p in CREDENTIAL_SHAPES)


def redact(obj: Any, removed: Optional[List[str]] = None, path: str = "") -> Any:
    """Remove held-out material and anything shaped like a credential, on every write.

    The bundle is the artefact an independent analyst receives. An answer key inside it turns the
    hidden suite into a published one and every later reading of that ladder into a reading of a
    ladder the systems could have seen. Removing by name is crude and is meant to be: a boundary that
    depends on every caller remembering is not a boundary. The name rule is now three rules (whole
    name, substring, word) and is joined by a value rule, because a key list can only ever remove what
    it was told to expect.

    A key matched by name is REMOVED; a value matched by shape is REPLACED with a placeholder. The
    difference is deliberate: a removed key says the field was never public, and a replaced value says
    a field that should have carried something innocuous carried a credential, which is a fact the
    operator needs to see rather than a field to make disappear.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _redacted_key(k):
                if removed is not None:
                    removed.append((path + "/" + str(k)).lstrip("/"))
                continue
            out[k] = redact(v, removed, path + "/" + str(k))
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v, removed, path + "[]") for v in obj]
    if _redacted_value(obj):
        if removed is not None:
            removed.append((path or "/").lstrip("/") or "/")
        return CREDENTIAL_PLACEHOLDER
    return obj


# --------------------------------------------------------------------------------------------------
# The evidence bundle
# --------------------------------------------------------------------------------------------------

class EvidenceBundle:
    """A directory holding one run's evidence, written in two acts.

    The first act happens at the seal: the manifest and the sealed record with its receipt go to disk
    before the first continuation is generated. That is what makes the acceptance case "a crash
    immediately after seal" produce evidence rather than nothing, and it is the only ordering under
    which the commitment is on disk before the data that could decide it exist.

    The SECOND act happens as the collection proceeds, and is the acceptance case "a crash midway
    through collection", which the first act does not cover. A commitment on disk with no collected
    rows beside it says what the run promised and nothing about what it had already measured when it
    stopped, so a run that dies in the middle of a paid continuation loses everything it paid for. The
    progress record is appended one line at a time, at each collection milestone, with the ladder
    reads taken since the previous line, so what has been collected is on disk at every point at which
    a stop is possible.

    The third act happens at the end: the complete record, under an explicit schema, with the bank
    rows, the reads, the replicate series, the margin series, the verdicts and the provider metadata.
    """

    MANIFEST = "manifest.json"
    SEAL = "seal.json"
    BUNDLE = "bundle.json"
    PROGRESS = "progress.jsonl"

    def __init__(self, root: str):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self.written: List[str] = []
        self._reads: Optional[List[Dict[str, Any]]] = None
        self._reads_flushed = 0

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def attach_reads(self, reads: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """The live read log, so that each progress line can carry the reads taken since the last one.

        The reads accumulate in memory until the bundle is written, so a run that stops loses every
        reading it had taken. Flushing them with the progress lines means the readings survive the
        stop, and the slice is taken by index so that no reading is written twice.
        """
        self._reads = reads
        self._reads_flushed = 0 if reads is None else len(reads)
        return reads

    def record_progress(self, stage: str, payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """One line of collected evidence, appended and flushed, with the new reads beside it.

        Appended rather than rewritten because rewriting a whole record at every milestone is what
        makes a run stop persisting evidence when the record gets big; flushed because an unflushed
        buffer is not on disk, and being on disk at the moment of the stop is the entire point.
        """
        new_reads: List[Dict[str, Any]] = []
        if self._reads is not None:
            new_reads = list(self._reads[self._reads_flushed:])
            self._reads_flushed = len(self._reads)
        line = {"schema": BUNDLE_SCHEMA, "stage": str(stage), "written_utc": utc_now(),
                "reads": new_reads, **(payload or {})}
        removed: List[str] = []
        clean = redact(line, removed)
        if removed:
            clean["redacted_paths"] = sorted(set(removed))
        p = self.path(self.PROGRESS)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean, sort_keys=True, default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if p not in self.written:
            self.written.append(p)
        return p

    def _write(self, name: str, payload: Any) -> str:
        removed: List[str] = []
        clean = redact(payload, removed)
        if isinstance(clean, dict) and removed:
            clean = dict(clean)
            clean["redacted_paths"] = sorted(set(removed))
        p = self.path(name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=1, sort_keys=True, default=_jsonable)
        if p not in self.written:
            self.written.append(p)
        return p

    def write_manifest(self, manifest: Dict[str, Any]) -> str:
        return self._write(self.MANIFEST, manifest)

    def write_seal(self, manifest: Dict[str, Any]) -> str:
        """Called immediately after the seal, before any continuation exists."""
        self.write_manifest(manifest)
        seal = manifest.get("seal") or {}
        return self._write(self.SEAL, {
            "schema": BUNDLE_SCHEMA, "experiment": manifest.get("experiment"),
            "execution_mode": manifest.get("execution_mode"), "seal": seal,
            "anchor_receipt": seal.get("anchor_receipt"),
            "seal_sha256": seal.get("seal_sha256"),
            "note": "written at seal time, before the held-out continuation was generated; a run that "
                    "stops after this point still leaves its commitment on disk"})

    def write_bundle(self, payload: Dict[str, Any]) -> str:
        return self._write(self.BUNDLE, payload)


def as_bundle(bundle: Any) -> Optional[EvidenceBundle]:
    """A path, a bundle or nothing. Callers take `bundle=` and never care which they were given."""
    if bundle is None:
        return None
    if isinstance(bundle, EvidenceBundle):
        return bundle
    return EvidenceBundle(str(bundle))


def _jsonable(value: Any) -> Any:
    """Anything numpy hands us becomes a plain number or a plain list; anything else becomes its
    repr rather than stopping the write. Losing a run's whole record to one unserialisable field is
    the wrong trade when the record is the evidence."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return repr(value)


# WHAT AN EMPTY READ LIST MEANS, SAID RATHER THAN LEFT TO THE READER. `reads` was written as
# `list(reads or [])`, so a run that attached no log and a run whose log recorded nothing left the
# same empty list, and neither could be told from a run that genuinely read nothing. The provider
# block beside it already says in words when it was given a name instead of an adapter; the readings
# now say the same kind of thing about themselves.
READS_RECORDED_NOTE = ("every reading this run took, written by the ladder's own read log as it was "
                       "taken")
READS_NO_LOG_NOTE = ("no read log was attached to this run, so this list is empty because nothing "
                     "was recorded and not because nothing was read: a run holding only the ladder's "
                     "hash cannot record what the ladder was asked")
READS_LOG_EMPTY_NOTE = ("a read log was attached to the ladder this run named and it recorded "
                        "nothing. Where the run collected a bank or any arm, that means the object "
                        "named here is not the object its source reads, and the identity sealed "
                        "beside it describes a pool nothing in this run ever read")


def build_bundle(manifest: Dict[str, Any], experiment: str, config: Any, *,
                 sealed: Optional[Dict[str, Any]] = None, routes: Optional[Dict[str, Any]] = None,
                 bank: Optional[Dict[str, Any]] = None, heldout: Optional[Dict[str, Any]] = None,
                 arms: Optional[Sequence[Dict[str, Any]]] = None,
                 verdicts_: Optional[Dict[str, Any]] = None,
                 replication: Optional[Dict[str, Any]] = None,
                 reads: Optional[Sequence[Dict[str, Any]]] = None,
                 provider: Optional[Dict[str, Any]] = None,
                 manipulations: Optional[Sequence[Dict[str, Any]]] = None,
                 ladder: Any = None) -> Dict[str, Any]:
    """The bundle's explicit schema, assembled in one place so that a missing part is visible.

    Every field a later analyst needs is named here even when this run has nothing to put in it: an
    absent key and an empty one are different statements, and only the second one can be read.
    """
    seal = manifest.get("seal") or {}
    return {
        "schema": BUNDLE_SCHEMA,
        "written_utc": utc_now(),
        "experiment": experiment,
        "execution_mode": manifest.get("execution_mode"),
        "scoreable_at_proposition_level": bool(manifest.get("scoreable_at_proposition_level")),
        "mode_label": manifest.get("mode_label"),
        "manifest": manifest,
        "seal": seal,
        "anchor_receipt": seal.get("anchor_receipt"),
        "seal_sha256": seal.get("seal_sha256") or (seal_digest(seal) if seal else None),
        "config": config_record(config) or config_record(manifest.get("config")),
        "sealed_predictions": sealed,
        "code_identity": {"code_sha256": manifest.get("code_sha256"),
                          "schema": CODE_IDENTITY_SCHEMA,
                          # The dependency manifest travels INSIDE the identity block rather than
                          # beside it, because the package's own bytes are only half of what decided
                          # the numbers: the fits go through scipy and the streams through numpy, and
                          # a code hash that omits their versions is an identity of the analysis
                          # script and not of the analysis. Finding A8 names it as part of the same
                          # action as the relative paths.
                          "environment": manifest.get("environment") or environment_manifest(),
                          "ladder_identity": manifest.get("ladder_identity") or ladder_identity(ladder)},
        "bank": bank or {},
        # The independent capability manipulations this run measured, each with its own bank rows and
        # its own fit, so that an analyst can re-estimate the second channel's elasticity as well as
        # the first. An empty list is a statement: this run supplied none, and its identification
        # judgement will say NOT ESTABLISHED for that reason. Finding A6.
        "capability_manipulations": list(manipulations or []),
        "routes": routes or {},
        "heldout": heldout or {},
        "arms": list(arms or []),
        # The fresh-data repetition record, where the run carries one. An absent key and an empty one
        # are different statements, and only the second can be read: a bundle with no repetition is a
        # provisional single run, which the verdicts recomputed from it will say.
        "replication": replication,
        "reads": list(reads or []),
        "reads_note": (READS_NO_LOG_NOTE if reads is None
                       else (READS_RECORDED_NOTE if len(reads) else READS_LOG_EMPTY_NOTE)),
        "provider": provider or {},
        "verdicts": verdicts_,
        "access_boundary": "PUBLIC. This bundle carries no answer key, no hidden check and no "
                           "credential; those live with the held-out material and never travel with "
                           "the evidence. Verification of what passed is by the ladder and verifier "
                           "hashes recorded here. Where `redacted_paths` is present, a hash "
                           "recomputed from the redacted object will not match the hash recorded "
                           "beside it: the recorded hash was taken over the whole object and it is "
                           "the one that governs.",
        "how_to_check": "recompute arc_runner.custody.package_code_identity() and compare with "
                        "code_identity.code_sha256; compare your environment with the run's using "
                        "arc_runner.custody.environment_differences(bundle['code_identity']"
                        "['environment']), which names every dependency whose version differs; "
                        "recompute arc_runner.custody.spec_hash_of(bundle['manifest']) and compare "
                        "with seal.spec_sha256, which is what binds the run's execution mode to its "
                        "commitment; recompute the seal digest with "
                        "arc_runner.custody.seal_digest(bundle['seal']) and compare with the anchor "
                        "receipt; then recompute the verdicts with "
                        "arc_runner.custody.recompute_verdicts(bundle). The progress record beside "
                        "this file, read with arc_runner.custody.load_progress(directory), holds what "
                        "had been collected at each milestone and is the only record a run that "
                        "stopped part-way leaves.",
    }


def load_bundle(path: str) -> Dict[str, Any]:
    """Read a bundle from a directory or from the bundle file itself."""
    p = path
    if os.path.isdir(p):
        p = os.path.join(p, EvidenceBundle.BUNDLE)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_progress(path: str) -> List[Dict[str, Any]]:
    """The progress record, in the order it was written. Empty where a run wrote none.

    A partial line at the end is dropped rather than raising: the record exists precisely because runs
    stop unexpectedly, and a reader that cannot open a record written by a run that was killed
    mid-write is a reader that fails in the one case the record is for.
    """
    p = path
    if os.path.isdir(p):
        p = os.path.join(p, EvidenceBundle.PROGRESS)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _config_from(cls, record: Dict[str, Any]):
    fields = getattr(cls, "__dataclass_fields__", {})
    return cls(**{k: v for k, v in (record or {}).items() if k in fields})


def recompute_verdicts(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute this run's verdicts from the saved bundle alone.

    This is the acceptance case that matters most: an analyst who has the bundle and the code, and
    who was not present for the run, reaches the same table. Nothing here reads the live run, the
    terminal summary or any file the bundle does not name.
    """
    exp = str(bundle.get("experiment", "")).upper()
    man = bundle["manifest"]
    if exp == "P5":
        from . import p5
        cfg = _config_from(p5.P5Config, bundle.get("config") or {})
        return p5.verdicts(man, bundle["routes"], bundle["heldout"], cfg)
    if exp == "P16":
        from . import p16
        cfg = _config_from(p16.P16Config, bundle.get("config") or {})
        return p16.verdicts(man, bundle["sealed_predictions"], bundle["arms"], cfg,
                            bundle.get("replication"))
    raise CustodyRefusal("no analysis is registered for experiment %r" % bundle.get("experiment"))


def reanalyse_bank(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Re-estimate the two routes from the saved bank rows rather than from the saved fit.

    The point estimates do not depend on the resampling stream, so an analyst who re-runs this on the
    bundle's rows must get the run's coupling back exactly. Anything else means the saved rows are
    not the rows the run analysed.
    """
    import numpy as np
    from . import p5
    cfg = _config_from(p5.P5Config, bundle.get("config") or {})
    bank = bundle.get("bank") or {}
    if not bank.get("rows"):
        raise CustodyRefusal("this bundle carries no bank rows, so the routes cannot be re-estimated")
    seed = int((bundle.get("config") or {}).get("seed", 0) or 0)
    # The manipulations the run measured travel back into the reanalysis, so that a bundle re-scores
    # to the same identification judgement. Without them a reanalysis of a run that DID intervene
    # twice would silently report NOT ESTABLISHED, which is the opposite of the error finding A6
    # names but is the same error: a label that does not follow the evidence.
    #
    # AND THE JUDGEMENT RE-DERIVES THEIR ADMISSIBILITY FROM THEIR OWN DESCRIPTIONS rather than
    # reading the `documentation_failures` key the bundle carries. A bundle is a file, and a file
    # whose key says the manipulation was documented is not the manipulation's documentation: see
    # `arc_runner.p5_identification.failures_in_record`. A bundle whose manipulation block has been
    # removed re-scores to NOT ESTABLISHED, which is what a record nobody can check deserves.
    manipulations = (bundle.get("capability_manipulations")
                     or (bundle.get("routes") or {}).get("capability_manipulations") or [])
    return p5.estimate_routes(bank, cfg, np.random.default_rng(seed),
                              manipulation_estimates=list(manipulations))
