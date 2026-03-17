# OSF Component Registration Wrapper

Title: `Preregistered empirical extension for Paper VII: next-domain structured comparison under the ARC/Cauchy scaling framework`

Status: `draft_local_ready_for_upload`

## Short description

This OSF component is the preregistration wrapper for the next empirical extension of the Paper VII Cauchy-unification programme. It does **not** preregister the current 50-domain run retroactively. Instead, it locks a fresh extension domain set, operator classifications, predicted families, fitter set, inclusion/exclusion rules, and the primary endpoint **before** new data extraction begins.

## Contribution claim

The target claim is not Cauchy's mathematics itself, but the hypothesis that Cauchy-type functional equations have a physically testable consequence for scaling-law family classification across domains.

## Contributors

- Michael Darius Eastwood

## Scientific packet included in this component

Primary preregistration files:

- `next_extension_protocol.md`
- `next_extension_manifest.json`

Supporting attachments:

- `osf_attachment_manifest.json`
- `file_checksums.txt`
- `../data/blinded_operator_classification_packet.json`
- `../data/blinded_operator_classification_template.json`
- `../results/blinded_operator_classification_instructions.md`
- `../results/cross_library_replication_status.md`

## Analysis version anchor

- Repository: `arc-principle-validation`
- Head commit at wrapper creation: `6b06321d590ce7c7cc4192e96fe97f371741af0a`
- Canonical runner: `experiments/cauchy-unification__Paper-VII/scripts/arc_50_domain_universal_test.py`
- Canonical manifest: `experiments/cauchy-unification__Paper-VII/data/canonical_50_domain_manifest.json`

## Data location once collection begins

Extracted numeric datasets for the preregistered extension should be placed in:

- `experiments/cauchy-unification__Paper-VII/preregistration/extracted_data/`

Do not modify the preregistration packet after OSF timestamping. New data should be added only under `extracted_data/` and referenced from a separate execution log.

## Upload instructions

1. Upload this wrapper and the scientific packet files unchanged.
2. Use the title above or an equivalent title with the same scope.
3. Mark the component as a preregistered extension of the Paper VII empirical programme.
4. Do not paste older parent-project OSF update text into this component.
5. Timestamp the packet before collecting or transcribing any new extension-domain data.
