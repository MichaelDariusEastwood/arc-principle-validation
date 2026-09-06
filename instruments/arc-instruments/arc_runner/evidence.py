"""Local, allow-listed evidence bundles. No secrets, uploads or scientific status promotion.

The bundle preserves all returned run evidence, not just a manifest. Hashes detect
accidental alteration; an attacker rewriting both data and local hashes is not excluded.
An independently held root commitment is a separate requirement for a deciding study.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys
from . import manifest as M

FILES = {"run.json", "environment.json", "bundle.json"}

def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"$arc_nonfinite"}:
            return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[value["$arc_nonfinite"]]
        return {k:_decode(v) for k,v in value.items()}
    if isinstance(value, list): return [_decode(v) for v in value]
    return value

def _load(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError("duplicate JSON key")
            out[k]=v
        return out
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError("non-standard JSON number")))

def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("bundle members must be regular files")
    data=path.read_bytes()
    return {"sha256":hashlib.sha256(data).hexdigest(), "bytes":len(data)}

def write_bundle(result, directory):
    M.require_integrity(result["manifest"])
    if result.get("empirical_verdict") != "NOT TESTED":
        raise ValueError("development export may not assert an empirical verdict")
    target=Path(directory)
    if any(p.is_symlink() for p in [target, *target.parents]):
        raise ValueError("symlink output paths are not allowed")
    target.mkdir(parents=True, exist_ok=False)
    try:
        (target/"run.json").write_bytes(M.canonical_bytes(result)+b"\n")
        # Deliberately no environment variables or absolute machine paths.
        import numpy, scipy
        env={"python":platform.python_version(), "implementation":sys.implementation.name,
             "platform":platform.system(), "numpy":numpy.__version__, "scipy":scipy.__version__}
        (target/"environment.json").write_bytes(M.canonical_bytes(env)+b"\n")
        members={name:_identity(target/name) for name in sorted(FILES-{"bundle.json"})}
        receipt={"schema":"arc-evidence/1", "members":members,
                 "root_sha256":M.sha256_of(members), "visibility":"local research evidence, not a public release",
                 "evidence_status":result["evidence_status"], "empirical_verdict":"NOT TESTED"}
        (target/"bundle.json").write_bytes(M.canonical_bytes(receipt)+b"\n")
        return verify_bundle(target, current_code=True)["receipt"]
    except Exception:
        # Keep partial output for inspection. Absence of a valid receipt prevents use.
        raise

def verify_bundle(directory, *, expected_root=None, current_code=True):
    root=Path(directory)
    if any(p.is_symlink() for p in [root,*root.parents]) or not root.is_dir():
        raise ValueError("bundle root must be a real directory")
    if {p.name for p in root.iterdir()} != FILES:
        raise ValueError("bundle membership mismatch")
    for name in FILES: _identity(root/name)
    receipt=_load(root/"bundle.json")
    if receipt.get("schema") != "arc-evidence/1": raise ValueError("unsupported bundle schema")
    members={name:_identity(root/name) for name in sorted(FILES-{"bundle.json"})}
    if receipt.get("members") != members or receipt.get("root_sha256") != M.sha256_of(members):
        raise ValueError("bundle digest mismatch")
    if expected_root is not None and expected_root != receipt["root_sha256"]:
        raise ValueError("bundle differs from externally retained root")
    result=_decode(_load(root/"run.json"))
    M.require_integrity(result["manifest"], current_code=current_code)
    if result.get("empirical_verdict") != "NOT TESTED" or receipt.get("empirical_verdict") != "NOT TESTED":
        raise ValueError("development bundle claims an empirical verdict")
    if receipt.get("evidence_status") != result.get("evidence_status"):
        raise ValueError("evidence labels disagree")
    return {"receipt":receipt, "run":result}

def _config(cls, saved):
    """Rebuild a configuration object from a saved manifest's configuration record.

    A manifest's `config` is the run's configuration PLUS what the run committed to beside it: the
    seed, the assigned held-out panel, the capability manipulations, whether the deciding quantities
    were candidates. Those are sealed with the configuration on purpose, and they are not fields of
    the dataclass, so the record is filtered to the dataclass's own fields here rather than splatted
    into it. Nothing is dropped from the record itself: it stays in the manifest, inside the seal,
    where the checks above read it.
    """
    fields=set(getattr(cls,"__dataclass_fields__",{}))
    return cls(**{k:v for k,v in saved.items() if k in fields})


def _arms_with_series(run):
    """The saved arms with their measured series joined back on.

    A returned P16 result carries each arm's SUMMARY on the arm and each arm's SERIES beside it, so
    that a printed summary is not a wall of numbers. A re-scoring needs both: the whole point of it
    is to recompute the readings from the measurements rather than to read the stored readings back.
    The join is on the arm label and the replicate identity, which is why both travel with the
    series, and a series that matches no arm is a bundle whose two halves disagree.
    """
    import copy
    arms=copy.deepcopy(run["arms"])
    index={(a["arm"],a.get("replicate_id")):a for a in arms}
    if len(index)!=len(arms): raise ValueError("duplicate arm identity")
    for rec in run.get("arm_series",()):
        key=(rec["arm"],rec.get("replicate_id"))
        if key not in index: raise ValueError("a saved series belongs to no saved arm")
        for name in ("margin","readings"):
            if name not in rec: continue
            # A bundle that carries a series in both places must carry the same series in both. The
            # join would otherwise overwrite one with the other and a disagreement between the two
            # halves of a record would be repaired on the way past, which is the one thing a
            # verification step must never do.
            if name in index[key] and index[key][name]!=rec[name]:
                raise ValueError("a saved arm and its saved series disagree")
            index[key][name]=copy.deepcopy(rec[name])
    missing=[k for k,a in index.items() if "margin" not in a]
    if missing: raise ValueError("saved arms have no measured series to re-score")
    return arms


def replay_bundle(directory, *, expected_root=None):
    """Reanalyse saved observations; never call a model or regenerate a trajectory."""
    verified=verify_bundle(directory, expected_root=expected_root)
    run=verified["run"]; man=run["manifest"]
    if man["mode"] != "demo": raise ValueError("pilot has no proposition verdict to replay")
    cfg=man["config"]
    if man["experiment"] == "P5":
        from . import p5
        import copy
        import numpy as np
        from .trajectory import loglog_slope
        config=_config(p5.P5Config, cfg)
        bank=run["bank"]
        ids=[r["observation_id"] for r in bank["rows"]]
        if len(set(ids))!=len(ids): raise ValueError("duplicate bank observation identity")
        n_ctrl=max(1,int(round(config.reps*config.control_fraction_of_bank)))
        expected={(state,fraction,rep,control) for state in config.states for fraction in config.fractions
                  for control in (False,True) for rep in range(n_ctrl if control else config.reps)}
        actual=[(r["state"],r["fraction"],r["rep"],r["control"]) for r in bank["rows"]]
        if set(actual)!=expected or len(actual)!=len(expected): raise ValueError("bank assignment universe changed")
        for row in bank["rows"]:
            if (row["available"] != row["fraction"]*row["before"]
                    or row["increment"] != row["after"]-row["before"]
                    or row["read_sd"] != float(np.hypot(row["before_sd"],row["after_sd"]))):
                raise ValueError("bank derived measurements differ from before/after readings")
        rng=np.random.default_rng()
        rng.bit_generator.state=run["analysis_rng_state"]
        routes=p5.estimate_routes(bank,config,rng)
        saved_base={k:v for k,v in run["routes"].items() if k not in
                    ("beta_used_for_prediction","beta_se_used_for_prediction")}
        if M.sha256_of(routes)!=M.sha256_of(saved_base):
            raise ValueError("bank analysis does not reproduce")
        heldout=copy.deepcopy(run["heldout"])
        for system,fit in heldout["fitted"].items():
            slopes=[]
            if len(fit["replicates"])!=config.replicates:
                raise ValueError("assigned continuation count changed")
            if {r["replicate_id"] for r in fit["replicates"]}!=set(range(config.replicates)):
                raise ValueError("continuation identities changed")
            for rep in fit["replicates"]:
                if rep["depths"]!=list(config.checkpoints): raise ValueError("checkpoint grid changed")
                if rep["scores"]!=[c["reading"]["passes"] for c in rep["checkpoints"]]:
                    raise ValueError("checkpoint readings and fitted input disagree")
                slope=loglog_slope(rep["depths"],rep["scores"])
                if M.sha256_of(slope)!=M.sha256_of(rep["fitted_exponent"]):
                    raise ValueError("continuation fit differs from saved readings")
                if np.isfinite(slope): slopes.append(slope)
                rep["headroom_ok"]=not any(c["reading"]["at_ceiling"] for c in rep["checkpoints"])
            fit["headroom_ok"]=all(r["headroom_ok"] for r in fit["replicates"])
            arr=np.asarray(slopes,float)
            fit["fitted_exponent"]=float(arr.mean()) if len(arr) else float("nan")
            fit["fitted_se"]=float(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr)>=2 else float("nan")
            fit["n_replicates"]=len(arr)
        got=p5.verdicts(man,run["routes"],heldout,config)
    elif man["experiment"] == "P16":
        from . import p16
        import copy
        import numpy as np
        config=_config(p16.P16Config, cfg)
        arms=_arms_with_series(run)
        expected={(label,k) for label in run["sealed"]["signs"] for k in range(config.systems_per_arm)}
        actual=[(a["arm"],a["replicate_id"]) for a in arms]
        if set(actual)!=expected or len(actual)!=len(expected):
            raise ValueError("assigned arm universe changed")
        for arm in arms:
            series=arm["margin"]
            if len(series)!=config.horizon: raise ValueError("observation horizon changed")
            if np.all(np.isfinite(series)):
                arm.update(p16.detect_reversal(series,config))
                arm["window_slope"]=p16.window_slope(series,config)
                arm["observation_status"]="COMPLETE"
            else:
                arm.update(declared_round=None,slope=float("nan"),se=float("nan"),falling=False,
                           window_slope=float("nan"),observation_status="MISSING OR INVALID")
        got=p16.diagnostic_verdicts(arms,run["sealed"],config,man=man,
                                    replication=run.get("replication"))
    else: raise ValueError("unsupported experiment")
    if M.sha256_of(got) != M.sha256_of(run.get("diagnostics")):
        raise ValueError("saved diagnostics do not reproduce from the evidence")
    return {"reproduced":True,"empirical_verdict":"NOT TESTED", "evidence_status":run["evidence_status"],
            "root_sha256":verified["receipt"]["root_sha256"], "diagnostics":got}
