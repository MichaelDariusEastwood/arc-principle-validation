"""ARC instruments: reference code for the deciding units of registration v1.93.

Pure numpy and scipy, Python 3.9. The registration governs; this code exists so that every decision
rule, sensitivity claim and blinding audit the registration promises has one executable form with
tests. See README.md for the module map and the adoption path.
"""

__all__ = [
    "verdicts",
    "form_discrimination",
    "precision",
    "blinding",
    "diversity",
    "capacity",
    "coupling_identification",
    "dependence",
    "balance",
    "conversion",
    "resampling",
    "regions",
    "resources",
    "burden_identification",
    "identification_adversaries",
    "sealing",
    "parity",
]
