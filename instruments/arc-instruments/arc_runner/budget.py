"""A hard spending controller for paid runs, and a rule about what it may never do.

THE ONE RULE THAT MATTERS MOST HERE IS NOT FINANCIAL. A financial stop is not a scientific stopping
rule. Running out of money must never quietly reduce the depth, drop a registered repetition, switch
the model, or relax a margin. When the reserve cannot cover the next operation this controller raises
`BudgetExhausted`, the run halts with its observations preserved, and the result is reported under the
registered interrupted-run outcome. Nothing here can change a verdict rule, and the verdict code never
imports this module.

HOW IT BOUNDS SPENDING. Before a request is dispatched the caller reserves that request's
conservatively estimated maximum charge, which stays held while the request is in flight so that
concurrent dispatch cannot spend the same pound twice. On completion the caller settles the actual
charge and the unused remainder returns to the pool. Retries are capped per operation and every retry
reserves again, so a retry storm exhausts the allowance instead of the account.

WHAT IT CANNOT PROMISE, SAID PLAINLY. This is a local controller in front of a vendor that bills on
its own terms. It bounds what this runner asks for; it cannot bound what a vendor charges. Where
billing cannot be independently verified, a run must not advertise an absolute guarantee, and this
module refuses to describe itself as one. The allowance also has to say what it includes: tax,
payment fees, rented compute, storage and paid human raters either sit inside the approved figure or
are named as outside it, and unpaid human work stays in the resource account even when it is outside
the cash budget.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BudgetExhausted(RuntimeError):
    """Raised when the remaining reserve cannot cover the next operation. The caller stops and
    reports the registered interrupted-run outcome; it never continues on a cheaper setting."""


class PremiumFallbackRefused(RuntimeError):
    """Raised when a caller asks for a model outside the approved list. An unapproved fallback is a
    change to the experiment as well as to the bill."""


@dataclass
class Allowance:
    """The approved figure and what it covers. `includes` is a statement, not a calculation: it is
    recorded so that a reader can tell whether the number is a whole vendor bill or only part of one."""
    limit_gbp: float
    reserve_gbp: float = 0.0
    includes: Dict[str, bool] = field(default_factory=lambda: {
        "vendor_inference": True, "tax": False, "payment_fees": False,
        "rented_compute": False, "storage": False, "paid_human_raters": False})
    approved_models: Optional[List[str]] = None
    max_retries_per_operation: int = 2
    unpaid_human_hours: float = 0.0          # outside the cash budget, never outside the account

    def total_gbp(self) -> float:
        return float(self.limit_gbp) + float(self.reserve_gbp)


class BudgetController:
    """Reserve, settle, release. Thread-safe, because concurrent dispatch is the case that breaks a
    naive running total."""

    def __init__(self, allowance: Allowance):
        self.allowance = allowance
        self._lock = threading.Lock()
        self._committed = 0.0                 # settled charges
        self._held = 0.0                      # reserved for requests in flight
        self._retries: Dict[str, int] = {}
        self.ledger: List[Dict[str, Any]] = []
        self.halted_reason: Optional[str] = None

    @property
    def committed_gbp(self) -> float:
        return self._committed

    @property
    def held_gbp(self) -> float:
        return self._held

    @property
    def available_gbp(self) -> float:
        return self.allowance.total_gbp() - self._committed - self._held

    def reserve(self, operation: str, estimated_max_gbp: float, model: Optional[str] = None) -> str:
        """Hold the conservative maximum for one request. Raises rather than overspending."""
        approved = self.allowance.approved_models
        if approved is not None and model is not None and model not in approved:
            raise PremiumFallbackRefused(
                "model %r is not in the approved list %r; an unapproved fallback changes the experiment "
                "as well as the bill" % (model, approved))
        with self._lock:
            if self.halted_reason:
                raise BudgetExhausted(self.halted_reason)
            if estimated_max_gbp < 0:
                raise ValueError("a negative reservation is not a reservation")
            if estimated_max_gbp > self.available_gbp:
                self.halted_reason = (
                    "the remaining reserve (%.2f) cannot cover the next operation %r (%.2f). The run "
                    "halts with its observations preserved and reports the registered interrupted-run "
                    "outcome. The depth, the repetitions, the model and the margin do not move."
                    % (self.available_gbp, operation, estimated_max_gbp))
                raise BudgetExhausted(self.halted_reason)
            self._held += float(estimated_max_gbp)
            token = "%s#%d" % (operation, len(self.ledger))
            self.ledger.append({"token": token, "operation": operation, "reserved": float(estimated_max_gbp),
                                "model": model, "settled": None})
            return token

    def settle(self, token: str, actual_gbp: float) -> None:
        """Record what was actually charged and return the unused remainder to the pool."""
        with self._lock:
            row = next(r for r in self.ledger if r["token"] == token)
            if row["settled"] is not None:
                raise ValueError("operation %s settled twice" % token)
            self._held -= row["reserved"]
            self._committed += float(actual_gbp)
            row["settled"] = float(actual_gbp)

    def release(self, token: str) -> None:
        """A request that never billed: the hold returns and nothing is committed."""
        self.settle(token, 0.0)

    def retry_allowed(self, operation: str) -> bool:
        with self._lock:
            n = self._retries.get(operation, 0)
            if n >= self.allowance.max_retries_per_operation:
                return False
            self._retries[operation] = n + 1
            return True

    def report(self) -> Dict[str, Any]:
        with self._lock:
            a = self.allowance
            return {"allowance_gbp": a.total_gbp(), "limit_gbp": a.limit_gbp, "reserve_gbp": a.reserve_gbp,
                    "committed_gbp": round(self._committed, 4), "held_gbp": round(self._held, 4),
                    "available_gbp": round(self.available_gbp, 4), "operations": len(self.ledger),
                    "halted": self.halted_reason, "includes": dict(a.includes),
                    "unpaid_human_hours": a.unpaid_human_hours,
                    "guarantee": "this bounds what the runner asks for and cannot bound what a vendor "
                                 "charges; it is not an absolute spending guarantee"}


class MeteredAdapter:
    """An adapter that cannot dispatch a call the remaining allowance has no room for.

    WHY THIS OBJECT EXISTS (finding A9). The approved ceiling was required by the deciding gate,
    recorded in the manifest, and then consulted by nothing: no run reserved against it, so the figure
    bounded the record and not the spending, while the record's own sentence said it bounded what the
    runner asked for. Every paid mode now dispatches through this wrapper, so the reserve, settle and
    halt rules at the top of this module are the rules the run actually obeys: when the remainder
    cannot cover the next call, `BudgetExhausted` is raised, the run stops with its observations
    preserved, and the depth, the repetitions, the model and the margin do not move.

    CONSERVATIVE SETTLEMENT, AND IT IS AN OPEN DECISION. No price per token is registered anywhere in
    this contract, so this runner cannot convert a provider's reported usage into a charge. Each
    reservation is therefore settled at the figure it reserved rather than at an actual charge: the
    ledger counts the maximum every call could have cost and never less, which is the direction that
    stops early rather than the direction that overspends. If the author registers a price, settlement
    becomes the priced usage and this method is the one place that changes.

    It wraps any adapter, including a simulated one, so that the metering can be exercised without a
    provider. It adds nothing to the call it wraps: the artefact that comes back is the inner
    adapter's, untouched.
    """

    def __init__(self, inner: Any, controller: BudgetController, estimated_max_call_gbp: float,
                 operation: str = "revise"):
        if not (float(estimated_max_call_gbp) > 0):
            raise ValueError("a per-call maximum of %r bounds nothing; a meter that reserves zero is "
                             "not a meter" % (estimated_max_call_gbp,))
        self.inner = inner
        self.controller = controller
        self.estimated_max_call_gbp = float(estimated_max_call_gbp)
        self.operation = operation
        self.name = getattr(inner, "name", "metered")

    def revise(self, artefact, retained, task, rng):
        token = self.controller.reserve(self.operation, self.estimated_max_call_gbp,
                                        getattr(self.inner, "model", None))
        try:
            out = self.inner.revise(artefact, retained, task, rng)
        except BaseException:
            # A call that raised may still have billed, and this runner cannot tell. The hold is
            # released rather than committed because a failed call that also consumed the allowance
            # would stop the run twice for one fault; the vendor's own account remains the record of
            # what was charged, which is what the guarantee sentence has always said.
            self.controller.release(token)
            raise
        self.controller.settle(token, self.estimated_max_call_gbp)
        return out

    def metadata(self) -> Dict[str, Any]:
        """The inner adapter's account, with the ledger this run spent under beside it."""
        fn = getattr(self.inner, "metadata", None)
        md = dict(fn()) if callable(fn) else {"adapter": self.name}
        md["budget"] = self.controller.report()
        md["budget"]["estimated_max_call_gbp"] = self.estimated_max_call_gbp
        return md


def pilot_allowance() -> Allowance:
    """The illustrative planning allowance for the pilot: a proposed allocation, not a vendor quote."""
    return Allowance(limit_gbp=40.0, reserve_gbp=10.0)


def decisive_allowance() -> Allowance:
    """The illustrative planning allowance for the decisive run: a proposed allocation, not a quote."""
    return Allowance(limit_gbp=240.0, reserve_gbp=60.0)
