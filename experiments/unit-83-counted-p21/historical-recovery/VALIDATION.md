# Recovery Amendment 01 — validation record

Prepared 28 October 2026 UTC. This validates the supplemental helper, not the experiment's outcome or the legitimacy of a provider policy rejection.

## Checks completed before delivery

- 21 synthetic tests passed under `/opt/homebrew/bin/python3.13` using temporary fixtures only. The tests import no live study module, read no live bank, model reply, score or interim result, and call no network endpoint. All OSF exchanges in the tests are simulated.
- The launcher passes `zsh -n` syntax checking.
- The existing registered manifest still has SHA-256 `013d7008aa92a7867385d20f15e6754988e1d83d20eebc6096b20b9e5505e026`, and each of its 43 listed source/configuration files was checked by static hashing and matched. This is not a claim that the live helper has run or that its runtime checks have passed.
- The existing journal's six-field recovery schema and numbered-attempt recovery path were inspected as static source. They accept a nonempty descriptive cause with the existing `reissue` action. A new cause is still a post-registration deviation even though the unchanged parser accepts its format.
- An unauthenticated OSF metadata GET for source component `zwnc5` returned HTTP 401 during preparation. The operator-run deposit uses the existing component token, records actual visibility, and never changes sharing settings.

## Behaviour exercised by the synthetic tests

1. One authorisation only; repeated execution does not duplicate release, deposit or audit records.
2. The helper leaves the journal database bytes unchanged.
3. Changed request content is refused by its canonical hash.
4. Changed registered source is refused.
5. A different provider request ID or error is refused.
6. Another unresolved hold is refused, including a later policy rejection after the first authorisation.
7. Missing verified deposit and wrong downloaded capsule bytes prevent release.
8. A successful upload followed by failed verification can be retried without a second named upload.
9. Initial application after a completed first look is refused.
10. Original evidence changed between preparation and application is refused.
11. Invalid recovery entries and a recorded abandonment are refused.
12. An active study lock blocks the helper.
13. Existing charges are never reduced below their amount or the original reservation.
14. Interrupted audit bookkeeping can finish without issuing a second authorisation.
15. Final disclosure is blocked until the report is complete, and preserves the original report bytes.
16. The OSF token is removed on redirects to other hosts; insecure redirects are refused.
17. The capsule contains hashes/metadata rather than actual request or reply files.
18. Restricted component visibility is recorded and disclosed without pretending it is public preregistration.

Several tests cover multiple behaviours above; the executable suite contains 21 test methods.

## Limits

No real rejected prompt has been retried during preparation. No live recovery entry has been written, no live journal row has been changed, and no amendment capsule has yet been uploaded by the assistant. Only the operator's launch command performs those actions. The helper does not determine whether the policy filter was correct. Network failures, insufficient funding, changed source, another policy rejection or a provider change can still stop the run. Synthetic tests do not guarantee a successful scientific result or completion before Saturday.

Re-run the helper tests without any provider credentials by running `python3.13 -m unittest -v test_recovery_amendment` from this directory with `PYTHONDONTWRITEBYTECODE=1`. This recreates only synthetic temporary fixtures.
