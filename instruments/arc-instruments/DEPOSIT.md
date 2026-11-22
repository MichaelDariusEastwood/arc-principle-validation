# Deposit checklist for the kit (ruling 19, 5 September 2026)

The kit is deposited with a citable DOI at the freeze of the theory-level registration, so that anyone who claims to test these laws either runs the registered design or is seen not to. The deposit is the author's act; this file lists what goes in and what is confirmed first.

## What is deposited
- The kit at the commit named in the registration's manifest: `arc_instruments/`, `tests/`, `README.md`, `CITATION.cff`, `.zenodo.json`, this file.
- The registration's decision regions as exported by `python3 -c "from arc_instruments import regions; print(regions.to_json())"` at that commit.
- The design-sensitivity outputs the charters cite, regenerated at the deposited commit.

## Confirmed by the author before upload
1. The ORCID: the site and the validation repository carry different identifiers; one is chosen and written into `CITATION.cff` and the deposit form.
2. The licence: `.zenodo.json` carries the Apache License 2.0, the licence of the validation repository's code, experiments, scripts, data and results from 22 November 2026 (its `LICENCE`, `NOTICE` and `LICENCE-CODE.md`); that repository's `CITATION.cff` carries `license: Apache-2.0`. Confirmed against that repository before the deposit points at it.
3. The version string and the date match the frozen commit and the registration's manifest.
4. `related_identifiers` gains the theory-level registration's identifier after the submission click.

## After the deposit
- The DOI is written into `CITATION.cff` (`doi:` field), into the registration's references by the applying session, and into the replication terms, the Registered Report and the website's result-chain record by their owners.
- Every later change to the kit ships as a new deposit version; the deposited bytes are never edited in place.
