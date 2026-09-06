"""Model adapters: one interface, two implementations, and a strict rule about keys.

An adapter takes an artefact, the retained context the round is allowed to see, and a task, and
returns the revised artefact. That is the only thing the experiments need a system to do. Everything
else (scoring, sealing, verdicts) is done by code the system never sees.

MockCouplingAdapter simulates a system obeying the registered growth equation: one round from an
available capability of f * U produces an increment a * (f * U) ** beta, with a nuisance rate that may
itself grow with capability (theta) so that the rate-confound world the design simulation named can
be reproduced here too. Its artefact is a dict carrying a latent capability. The ladder for the mock
measures that capability with binomial noise, so measurement error is real rather than assumed away.

WHAT THE MOCK CAN NOW BE, AND WHY (finding A6). Writing the increment as a power of f * U builds one
assumption into every clean world the pipeline was ever tested against: that the elasticity in the
retention fraction and the elasticity in the capability state are the SAME NUMBER. The runner's two
routes are those two elasticities, so a mock that cannot separate them cannot tell a run that
recovered the coupling from a run that recovered an assumption. `retention_exponent` frees the first
of them, so the general process a * f ** theta * U ** beta can be simulated and the estimator can be
asked to return beta when theta is something else. `available_rate_exponent` is the other half of the
same finding: a nuisance rate that scales with the AVAILABLE capability adds equally to both
elasticities, so the two routes agree exactly while both are measuring beta plus the nuisance. That
world is agreement without identification, and it is the world an independent capability manipulation
exists to catch.

OpenAICompatibleAdapter speaks to any endpoint that accepts the chat-completions shape. The base
URL, model name and key come from the environment and nothing in this file ever prints, logs or
returns the key. If the key is missing the adapter refuses to construct rather than failing later.

WHAT AN ADAPTER NOW KEEPS, AND WHY (finding A8). The real adapter used to return the response text
and throw away everything else the provider said: the model identifier the endpoint actually served,
the token usage, the response identifier. A saved run therefore could not show which model answered
or what it consumed, and a model substituted mid-run left no trace at all. Every call now appends one
record to `provider_metadata` and `metadata()` returns the totals for the evidence bundle. The key is
not in any of it, and never will be: the record holds what came back, not what was sent.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np


class ModelAdapter(Protocol):
    name: str

    def revise(self, artefact: Dict[str, Any], retained: Dict[str, Any], task: str,
               rng: np.random.Generator) -> Dict[str, Any]: ...

    def metadata(self) -> Dict[str, Any]:
        """What this adapter is willing to say about the calls it made, for the evidence bundle.
        Optional on the protocol so that an experimental adapter still satisfies it; a caller reaches
        it through `arc_runner.custody.adapter_metadata`, which copes with its absence."""


@dataclass
class MockCouplingAdapter:
    """A simulated system with a known coupling. Used to prove the pipeline before a real run."""
    beta: float = 0.5
    a: float = 0.70             # in ladder units of 20 to 90 this is about a tenth per round at the middle state,
                                # which is the rate the design documents assume in their own units
    theta: float = 0.0          # rate confound: a grows as U ** theta; zero is the clean world
    retention_exponent: Optional[float] = None
                                # the elasticity of the increment in the RETENTION fraction. None
                                # means the registered clean world, in which it equals beta and the
                                # increment is a power of the available capability f * U. A number
                                # makes it the general crossed process a * f ** theta_f * U ** beta,
                                # in which the capability elasticity is still beta and the two
                                # regression directions of the crossed bank no longer estimate one
                                # quantity between them.
    available_rate_exponent: float = 0.0
                                # a nuisance rate scaling as the available capability to this power.
                                # It adds to BOTH elasticities equally, so the two routes agree at
                                # beta plus this number while neither of them is beta: the shared
                                # omitted rate factor of finding A6.
    noise: float = 0.05         # multiplicative noise on the increment
    control_leak: float = 0.0   # a negative-control cell (retained material the round cannot use) should show no
                                # coupling in the retained fraction; a positive value here makes the mock leak one,
                                # which is the world in which the registered control must fail the identification
    saturation: Optional[float] = None   # H3's rival: the increment saturates in available capability at this scale
    control_floor: float = 0.05 # what the round can still do from its own state when the retained material is unusable
    name: str = "mock-coupling"
    calls: int = 0               # counted so that a demonstration's bundle reports the same shape a real run's does

    def metadata(self) -> Dict[str, Any]:
        """A simulated system has no provider metadata, and says so rather than leaving the field
        empty for a reader to interpret."""
        return {"adapter": self.name, "simulated": True, "calls": int(self.calls),
                "returned_models": [], "usage_totals": {}, "records": [],
                "note": "a simulated system: there was no provider, so there is no provider metadata"}

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        U = float(artefact["capability"])
        f = float(retained.get("fraction", 1.0))
        if retained.get("control") == "unusable_retention":
            available = max(self.control_floor * U, 1e-9) * (f ** self.control_leak)
        else:
            available = max(f * U, 1e-9)
        rate = self.a * (U ** self.theta) * (available ** float(self.available_rate_exponent))
        if self.retention_exponent is None or retained.get("control") == "unusable_retention":
            # The clean world, and every control cell: the control's retained material is unusable by
            # construction, so there is no retention elasticity in it to be different.
            response = available ** self.beta
        else:
            response = (max(f, 1e-9) ** float(self.retention_exponent)) * (U ** self.beta)
        if self.saturation is not None:
            response = response / (1.0 + available / float(self.saturation))
        increment = rate * response * float(np.exp(rng.normal(0.0, self.noise)))
        new = dict(artefact)
        new["capability"] = U + increment
        new["rounds"] = int(artefact.get("rounds", 0)) + 1
        return new


@dataclass
class OpenAICompatibleAdapter:
    """A real system behind an OpenAI-compatible endpoint. Configured from the environment only."""
    uses_remote_endpoint = True
    model: str = field(default_factory=lambda: os.environ.get("ARC_RUNNER_MODEL", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("ARC_RUNNER_BASE_URL", ""))
    key_env: str = "ARC_RUNNER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 3000
    timeout_s: float = 120.0
    name: str = "openai-compatible"
    provider_metadata: list = field(default_factory=list)

    def __post_init__(self):
        if not os.environ.get(self.key_env):
            raise RuntimeError("no API key in %s; the adapter will not construct without one" % self.key_env)
        if not self.model or not self.base_url:
            raise RuntimeError("ARC_RUNNER_MODEL and ARC_RUNNER_BASE_URL must both be set")
        self.name = "openai-compatible:" + self.model

    def _record(self, out: Dict[str, Any], prompt: str = "", response: str = "") -> None:
        """One line of provider metadata per call. The model identifier is recorded as returned and
        not as requested, because those two differing is the case worth catching: a silent
        substitution changes the experiment as well as the bill.

        AND THE CALL IS BOUND TO ITS OWN TEXT (finding A8, which asks for a raw-response record). The
        text itself does not go in: a response here is an artefact that passes the hidden suite's
        checks, so publishing it in the evidence bundle publishes solutions to the held-out material,
        which is the one thing the access boundary forbids. What goes in is the digest of the prompt
        and the digest and length of the response, which is enough for a holder of the artefacts to
        show that a given response came from a given call, and enough for anybody to see that a call
        which returned nothing returned nothing. The artefacts themselves live in the checkpoint
        store, where they are already kept and are not public.
        """
        usage = out.get("usage") or {}
        self.provider_metadata.append({
            "response_id": out.get("id"), "model_requested": self.model,
            "model_returned": out.get("model"), "created": out.get("created"),
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                      if k in usage},
            "finish_reason": (out.get("choices") or [{}])[0].get("finish_reason"),
            "system_fingerprint": out.get("system_fingerprint"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "response_chars": len(response)})

    def metadata(self) -> Dict[str, Any]:
        """The totals the evidence bundle carries, plus every per-call record. No key, no prompt text
        and no response text: the bundle records what the provider charged for, what it says it was,
        and the DIGEST of each prompt and response, so a call can be bound to an artefact held in the
        checkpoint store without the held-out solutions travelling in a public bundle."""
        totals: Dict[str, int] = {}
        for r in self.provider_metadata:
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + int(v)
        returned = sorted({r.get("model_returned") for r in self.provider_metadata
                           if r.get("model_returned")})
        return {"adapter": self.name, "simulated": False, "calls": len(self.provider_metadata),
                "model_requested": self.model, "returned_models": returned,
                "model_substituted": bool([m for m in returned if m != self.model]),
                "usage_totals": totals, "records": list(self.provider_metadata)}

    def _call(self, messages):
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature, "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ[self.key_env]})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            out = json.loads(r.read().decode())
        text = (((out.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        # The record is taken from what came back rather than from what was expected, and it is taken
        # BEFORE the indexing below, so a malformed response is still an accounted call.
        self._record(out, prompt="\n".join(m.get("content", "") for m in messages), response=text)
        return out["choices"][0]["message"]["content"]

    def revise(self, artefact, retained, task, rng):
        prior = retained.get("text", "")
        prompt = ("Task:\n%s\n\n" % task +
                  ("Your prior attempt, as much of it as you are permitted to see:\n%s\n\n" % prior if prior else "") +
                  "Produce the complete revised artefact and nothing else.")
        text = self._call([{"role": "user", "content": prompt}])
        new = dict(artefact)
        new["text"] = text
        new["rounds"] = int(artefact.get("rounds", 0)) + 1
        return new
