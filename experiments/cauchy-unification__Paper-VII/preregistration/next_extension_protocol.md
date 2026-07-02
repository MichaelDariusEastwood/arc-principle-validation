# Preregistered Extension Packet for Paper VII

Status: `archived_local_packet_exercised_before_timestamp`

This packet was originally drafted as the next-step strengthening protocol for the Cauchy-unification empirical programme. However, local data extraction and a dry-run execution were performed on 17 March 2026 before any OSF timestamp. It can no longer serve as a clean prospective preregistration and should be treated as an archived locked pilot extension instead.

## Integrity note

- Do not upload this packet unchanged as a preregistration.
- Use it only as an audit trail for the first locked 12-domain dry run.
- Create a fresh packet with new domains before any future OSF-timestamped extension.

## Purpose

Extend the empirical curve-fit cohort with a fresh preregistered set of new domains rather than retrofitting the current 25-domain cohort.

## Scope

- New empirical domains only.
- No reuse of the current 25 empirical datasets as headline evidence.
- No tier blending with published-exponent or analytic-identity cases.

## Locked primary endpoint

Strict family match on the new empirical extension domains only:

- operator class is locked before data extraction
- predicted family is locked before data extraction
- best model is selected by the same AICc-based harness as the canonical 50-domain suite
- no `R^2 within 0.05` rescue
- no blended p-value across secondary tiers

## Locked fitter set

The extension must use the same candidate model set as the canonical 50-domain runner:

- `power_law`
- `exponential`
- `saturation_exp`
- `michaelis_menten`
- `logistic`
- `hill`
- `hyperbolic_decay`

The same saturation guardrail must remain active:

- bounded models only count as valid winners if the observed data approach at least 50% of the fitted asymptote

## Inclusion rules

- empirical dataset must come from a published paper, official dataset, or traceable laboratory/field dataset
- at least 8 usable data points unless the domain is a historically canonical dataset with unavoidable sparsity
- source and preprocessing notes must be archived alongside the extracted numeric data
- one domain, one predeclared operator class, one predeclared predicted family

## Exclusion rules

- do not include purely analytic identities in the empirical extension headline
- do not include datasets whose key variable definitions are still being argued about after extraction
- do not replace a miss inside the current 25-domain cohort and call it a correction; treat replacements as new extension domains
- if the dataset obviously mixes incompatible physical regimes, either preregister a regime split up front or exclude the domain

## Secondary checks to run with the extension

- blinded operator classification by an independent assessor
- shuffled-y null control using the current stricter harness
- cross-library replication using the R scaffold once R is available

## Success thresholds

- minimum publishable signal: new extension empirical cohort beats chance at `p < 0.01`
- strong reinforcement: new extension empirical cohort beats chance at `p < 0.001`
- decisive reinforcement: a fresh preregistered extension plus independent operator agreement plus cross-library replication all succeed

## Interpretation guardrails

- do not combine extension results with the existing 25-domain cohort into one retroactive headline p-value
- do not change existing canonical misses after seeing extension results
- report exact matches and misses, not only p-values

## Manifest

The locked candidate list for the next extension is in:

- `preregistration/next_extension_manifest.json`
