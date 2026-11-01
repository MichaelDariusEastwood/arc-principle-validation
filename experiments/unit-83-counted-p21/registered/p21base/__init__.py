"""The base carried by 41-D-COUNTED (DESIGN-FREEZE.md, section 12, package A).

Four modules of the compact Screen's version 0.3 stage, as audited in BASE-AUDIT.md: `common.py` unchanged (sha256
37b9a91ce907c74d2ecfd9fa06145f000cabecdcaeb5b2f6629ba8e2ac1ee192), `ledger.py`, `providers.py` and
`registration_gate.py` changed as package A says. Nothing of the Screen's sandbox, tasks, arms, analysis or
command-line tool is carried or loaded. Nothing here is imported or executed before both registrations read
REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of each file's text.
"""
