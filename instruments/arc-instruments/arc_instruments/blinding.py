"""Blinding audit kit for Condition A2 and the scorer gates.

Three measurements, each on calibration material and never on the deciding outputs:

1. Blinding indices. James (1996): (1 + P_dontknow + P_incorrect - P_correct) / 2, from 1 (everyone
   says they do not know) through 0.5 (guessing at chance) to 0 (complete unblinding). Bang (2004),
   per arm: (2 p_correct - 1) times the fraction who ventured a guess, from -1 (opposite guessing)
   through 0 (chance) to +1 (complete unblinding). Bootstrap intervals over respondents.

2. The same-answer label experiment. Identical answers shown under randomised labels (the judge's own
   family, another family, or hidden), crossed with judge family; the label-induced score shift is
   the difference in mean score between label conditions, paired by item, with a bootstrap interval
   over items. It isolates the effect of a declared label on a fixed answer; it does not detect a
   keyed watermark, which is a separate channel.

3. The provenance challenge. An attacker trained on held-out calibration outputs with known source
   family predicts the family of fresh outputs; accuracy with a Wilson interval against chance, and a
   positive control (a deliberately leaked marker) that the attacker must detect, so that "found no
   leak" is never "the audit is weak". The attacker here is a hashed character n-gram multinomial
   model, a baseline; a deciding unit registers stronger classes beside it.

Nothing in this module removes, paraphrases or rewrites any output.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]
DONT_KNOW = "DK"


# --------------------------------------------------------------------------------- blinding indices
def james_index(actual: Sequence[str], guess: Sequence[str]) -> float:
    """The chance-adjusted James blinding index over paired (actual arm, guessed arm or DK) responses.

    BI = (1 + P_DK + (1 - P_DK) K_D) / 2, where K_D = (P_do - P_de) / (1 - P_de) is the disagreement
    kappa among the non-DK responses: P_do the observed proportion of wrong guesses and P_de the
    proportion expected under independence of guess and allocation from their marginals (unweighted,
    two or more arms). The earlier shortcut (one minus the correct fraction) coincides with this only in
    balanced special cases: with 90 assignments to A, 10 to B and every guess A, the shortcut returns
    0.10 and this returns 0.50, because the guesser only exploited the allocation imbalance. Zero DK
    and complete correct guessing give 0; complete opposite guessing gives 1; all DK gives 1.
    """
    if len(actual) != len(guess) or not actual:
        raise ValueError("actual and guess must be non-empty and equal length")
    n = len(actual)
    non = [(a, g) for a, g in zip(actual, guess) if g != DONT_KNOW]
    p_dk = 1.0 - len(non) / n
    if not non:
        return 1.0
    m = len(non)
    p_do = sum(1 for a, g in non if a != g) / m
    arms = set(actual) | set(g for _, g in non)
    ta = {x: sum(1 for a, _ in non if a == x) / m for x in arms}
    tg = {x: sum(1 for _, g in non if g == x) / m for x in arms}
    p_de = 1.0 - sum(ta[x] * tg[x] for x in arms)
    if p_de <= 0.0 or p_de >= 1.0:
        k_d = 0.0
    else:
        k_d = (p_do - p_de) / (1.0 - p_de)
    return (1.0 + p_dk + (1.0 - p_dk) * k_d) / 2.0


def bang_index(actual: Sequence[str], guess: Sequence[str], arm: str) -> float:
    """Bang blinding index for one arm: (2 p_correct - 1) times the share who guessed."""
    idx = [i for i, a in enumerate(actual) if a == arm]
    if not idx:
        raise ValueError("no respondents in arm %r" % arm)
    guessed = [i for i in idx if guess[i] != DONT_KNOW]
    if not guessed:
        return 0.0
    p_correct = sum(1 for i in guessed if guess[i] == arm) / len(guessed)
    return (2.0 * p_correct - 1.0) * (len(guessed) / len(idx))


def bootstrap_interval(stat, actual: Sequence[str], guess: Sequence[str], n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 20260905, **kw) -> Interval:
    rng = np.random.default_rng(seed)
    a = np.asarray(actual); g = np.asarray(guess); n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(stat(list(a[idx]), list(g[idx]), **kw))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


# ------------------------------------------------------------------------- same-answer experiment
def label_effect(rows: Iterable[Dict[str, object]], reference_label: str = "hidden",
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 20260905) -> Dict[str, Dict[str, object]]:
    """Label-induced score shift per judge family and displayed label, relative to the reference label.

    rows: dicts with keys item, judge_family, label, score. The same answer must appear under every
    label for the same item and judge family (the design randomises which judge session sees which
    label). The estimate is the mean over items of (score under label - score under reference), with a
    bootstrap interval over items.
    """
    table: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        table[(str(r["judge_family"]), str(r["label"]), str(r["item"]))].append(float(r["score"]))
    families = sorted({k[0] for k in table}); labels = sorted({k[1] for k in table})
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, object]] = {}
    for fam in families:
        out[fam] = {}
        for lab in labels:
            if lab == reference_label:
                continue
            items = sorted({k[2] for k in table if k[0] == fam and k[1] == lab} & {k[2] for k in table if k[0] == fam and k[1] == reference_label})
            if not items:
                continue
            diffs = np.array([np.mean(table[(fam, lab, it)]) - np.mean(table[(fam, reference_label, it)]) for it in items])
            boots = [float(np.mean(diffs[rng.integers(0, len(diffs), len(diffs))])) for _ in range(n_boot)]
            out[fam][lab] = {"estimate": float(diffs.mean()), "interval": (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))), "n_items": len(items)}
    return out


def simulate_label_experiment(n_items: int, families: Sequence[str], own_label_bias: Dict[str, float],
                              score_sd: float = 1.0, seed: int = 1) -> List[Dict[str, object]]:
    """Synthetic rows: each judge family scores every item under labels 'own', 'other' and 'hidden';
    own_label_bias[family] is the shift a judge gives when the displayed label is its own family."""
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(n_items):
        quality = rng.normal(5.0, 1.0)
        for fam in families:
            for lab in ("own", "other", "hidden"):
                shift = own_label_bias.get(fam, 0.0) if lab == "own" else 0.0
                rows.append({"item": "item%03d" % it, "judge_family": fam, "label": lab,
                             "score": quality + shift + rng.normal(0.0, score_sd)})
    return rows


# ----------------------------------------------------------------------------- provenance attacker
class HashedNGramNB:
    """Multinomial model on hashed character n-grams. Deterministic, dependency-free, a baseline."""

    def __init__(self, n: int = 4, buckets: int = 1 << 16, alpha: float = 0.5):
        self.n = n; self.buckets = buckets; self.alpha = alpha
        self.classes: List[str] = []
        self.log_prior: Dict[str, float] = {}
        self.log_lik: Dict[str, np.ndarray] = {}

    def _features(self, text: str) -> np.ndarray:
        v = np.zeros(self.buckets, dtype=np.float64)
        t = text.lower()
        for i in range(max(len(t) - self.n + 1, 1)):
            h = int(hashlib.blake2b(t[i:i + self.n].encode("utf-8"), digest_size=4).hexdigest(), 16)
            v[h % self.buckets] += 1.0
        return v

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "HashedNGramNB":
        self.classes = sorted(set(labels))
        counts = {c: np.zeros(self.buckets) for c in self.classes}
        n_c = {c: 0 for c in self.classes}
        for t, l in zip(texts, labels):
            counts[l] += self._features(t); n_c[l] += 1
        for c in self.classes:
            tot = counts[c].sum() + self.alpha * self.buckets
            self.log_lik[c] = np.log((counts[c] + self.alpha) / tot)
            self.log_prior[c] = np.log(n_c[c] / len(labels))
        return self

    def predict(self, text: str) -> str:
        f = self._features(text)
        scores = {c: self.log_prior[c] + float(f @ self.log_lik[c]) for c in self.classes}
        return max(scores, key=scores.get)


def wilson(successes: int, n: int, z: float = 1.96) -> Interval:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def provenance_challenge(train_texts: Sequence[str], train_labels: Sequence[str],
                         test_texts: Sequence[str], test_labels: Sequence[str],
                         positive_control_marker: Optional[str] = None,
                         materiality_margin: float = 0.10) -> Dict[str, object]:
    """Train the attacker on calibration outputs, score it on held-out outputs, and run the positive
    control by appending a leaked marker to a copy of the test set. Returns accuracy with a Wilson
    interval, chance, whether the lower bound exceeds chance plus the registered margin, and the
    positive-control accuracy. Nothing here is applied to the deciding outputs."""
    clf = HashedNGramNB().fit(train_texts, train_labels)
    correct = sum(1 for t, l in zip(test_texts, test_labels) if clf.predict(t) == l)
    n = len(test_texts); chance = 1.0 / max(len(set(train_labels)), 1)
    lo, hi = wilson(correct, n)
    result: Dict[str, object] = {"accuracy": correct / n if n else float("nan"), "interval": (lo, hi),
                                 "chance": chance, "leak_material": bool(lo > chance + materiality_margin),
                                 "n_test": n}
    if positive_control_marker is not None:
        leaked_train = [t + " " + positive_control_marker + l for t, l in zip(train_texts, train_labels)]
        leaked_test = [t + " " + positive_control_marker + l for t, l in zip(test_texts, test_labels)]
        clf2 = HashedNGramNB().fit(leaked_train, train_labels)
        pc = sum(1 for t, l in zip(leaked_test, test_labels) if clf2.predict(t) == l)
        result["positive_control_accuracy"] = pc / n if n else float("nan")
        result["positive_control_detected"] = bool(wilson(pc, n)[0] > chance + materiality_margin)
    return result


# ------------------------------------------------------------------------------- marking status
KNOWN_MARKED = "known-marked"
KNOWN_UNMARKED = "known-unmarked"
UNKNOWN_MARKING = "unknown"


def marking_status(provider_states_marked: Optional[bool]) -> str:
    """Per-artefact provider marking status. Unknown is recorded as unknown and never as unmarked."""
    if provider_states_marked is True:
        return KNOWN_MARKED
    if provider_states_marked is False:
        return KNOWN_UNMARKED
    return UNKNOWN_MARKING


# ------------------------------------------------------------------- transformation as intervention
def transformation_preservation(raw_properties: Sequence[Dict[str, object]],
                                transformed_properties: Sequence[Dict[str, object]],
                                properties: Sequence[str], floor: float = 0.95) -> Dict[str, object]:
    """A rewrite is a measurement intervention, never a neutral anonymisation step.

    For each named outcome property (refusal, negation, quantifier, commitment, factual proposition,
    uncertainty, relevant omission) compare the adjudicated value on the raw item with the value on the
    transformed item. The preservation rate per property carries a Wilson interval; the transformed view
    is admissible as a primary endpoint only where every property's lower bound clears the registered
    floor. Anything less makes transformation a labelled sensitivity analysis."""
    if len(raw_properties) != len(transformed_properties):
        raise ValueError("raw and transformed items must be paired one to one")
    n = len(raw_properties)
    out: Dict[str, object] = {"n": n, "per_property": {}}
    admissible = n > 0
    for p in properties:
        kept = sum(1 for r, t in zip(raw_properties, transformed_properties) if r.get(p) == t.get(p))
        lo, hi = wilson(kept, n)
        ok = bool(lo > floor)
        out["per_property"][p] = {"preserved": kept, "rate": (kept / n if n else float("nan")), "interval": (lo, hi), "clears_floor": ok}
        admissible = admissible and ok
    out["admissible_as_primary"] = bool(admissible)
    return out


# ------------------------------------------------------------------- allocation and genuine effects
def genuine_effect_classifier_accuracy(effect: float, sd: float) -> float:
    """The best achievable treatment classification from the outcome alone under equal-variance normal
    outcomes: Phi(effect / (2 sd)). A shift of 0.4 at sd 0.4 gives about 69 per cent with no source cue at
    all, so a gate demanding chance classification of outcome-bearing content would reject a real effect."""
    from scipy.stats import norm
    return float(norm.cdf(effect / (2.0 * sd)))


def allocation_confounding(true_effect: float = 0.0, source_bias: float = 0.4, noise: float = 0.4, n: int = 80,
                           reps: int = 2000, seed: int = 0) -> Dict[str, float]:
    """False-positive fractions for a treatment contrast when source is completely confounded with
    treatment, when source is balanced across arms and ignored, and when it is balanced and adjusted.
    With zero true effect and a source-related score shift, complete confounding produces a false result
    almost always; balanced allocation with adjustment holds the nominal rate. A design diagnostic, not
    an estimate of any real provenance bias."""
    from scipy.stats import t as tdist
    rng = np.random.default_rng(seed)
    half = n // 2
    T = np.r_[np.zeros(half), np.ones(n - half)]
    P_bal = np.tile(np.r_[np.zeros(half // 2), np.ones(half - half // 2)], 2)[:n]
    P_conf = T.copy()
    Xb = np.column_stack([np.ones(n), T])
    Xadj = np.column_stack([np.ones(n), T, P_bal])
    out: Dict[str, float] = {}
    for label, P, X in (("confounded", P_conf, Xb), ("balanced_unadjusted", P_bal, Xb), ("balanced_adjusted", P_bal, Xadj)):
        false = 0
        for r in range(reps):
            y = true_effect * T + source_bias * P + rng.normal(0.0, noise, n)
            coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            df = n - X.shape[1]
            sigma2 = (float(res[0]) if len(res) else float(np.sum((y - X @ coef) ** 2))) / df
            se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
            false += int(abs(coef[1]) > tdist.ppf(0.975, df) * se)
        out[label] = false / reps
    return out
