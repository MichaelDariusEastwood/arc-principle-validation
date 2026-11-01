# Unit 83: inspecting the source and preparing an authorised repetition

## First distinguish the two tasks

**Source inspection is available now.** Read the files and compare their hashes with the [public registration](https://doi.org/10.17605/OSF.IO/MN73W). The original report and recovery disclosure remain together on [OSF](https://osf.io/98hkr/). No study module needs to run to check the supplied file hashes.

**A new live run needs its own explicit execution authority.** The commands below describe the original run-sheet order. They are not an authorisation to execute, an unattended launcher, or an instruction to resume the completed run. A reader must not claim to have clicked the author's registrations, reuse his run workspace or use his credentials. A separately controlled replication needs a new, versioned registration route; changing the frozen configuration in place would break its identity.

## Exact requirements

1. Author permission under the all-rights-reserved code licence and the applicable published registration. The code checks Unit 83 registration `mn73w`, the configured theory registration `p8ckq`, and source component `zwnc5`. The source component requires authorised access; its public metadata request returned HTTP 401 at this package's check.
2. CPython 3.13, with **3.13.13** recorded for the original run. The frozen implementation imports the standard library and its own modules. `fcntl` requires a POSIX environment. The original long commands use macOS `caffeinate`; the historical recovery launcher uses `zsh`. The original orchestration bench was made by bench factory 1.2 from Unit 06 bench 1.1; its SHA-256 is `1fa82fe465bfa955180f3f61e274bf379d21e58eba7924dfe2025fb56afad3e1`. Its source is included for inspection, without its full kit or private lab state. No newer Eden Lab engine is substituted.
3. The exact `DESIGN-FREEZE.md`, revision 4.2.1, SHA-256 `82c9e3cba25eeea7bf4055ef9b4cf4814783f80f6d2666060fce4e4cf2d9071d`. The authored charter test requires it through `P21LANE_FREEZE_FILE`. The original design file and author-specific run sheet are not bundled in this source directory; obtain the approved exact-byte copies before attempting the command sequence. Do not replace the design with a transcription of a registration form.
4. The configured provider identifiers and request settings: `claude-haiku-5-5` on Anthropic's standard synchronous route and `gpt-6-luna` on OpenAI's Flex synchronous route, with the frozen low-reasoning settings. Availability, prices, balances and identity must pass the original preflight. A provider alias does not guarantee identical model behaviour on a later day. The frozen configuration permits no call after 31 December 2026.
5. Provider credentials and a token authorised for the original OSF component, supplied by the operator in an ordinary terminal. The runtime freeze and bank commitment must be deposited there and read back before main collection. No credentials are supplied by this package. Never paste them into a transcript or a source file.
6. A new, empty run directory. The original completed workspace remains untouched. The original run sheet and its run-start record, admission read-back and stopping rules continue to govern; this guide does not replace them.

## Preparation commands that run no study code

Run from this source directory. Set `UNIT83_RUN` to a new directory outside this checkout, and `UNIT83_DESIGN` to the exact approved design file. The placeholders are deliberately not defaults.

```sh
shasum -a 256 -c MANIFEST.sha256
UNIT83_SOURCE="$PWD"
UNIT83_RUN=/absolute/path/to/a/new/unit83-run
UNIT83_DESIGN=/absolute/path/to/the/approved/DESIGN-FREEZE.md
UNIT83_PY=/absolute/path/to/python3.13
[ ! -e "$UNIT83_RUN" ] || { printf 'STOP: choose a new run directory.\n'; exit 1; }
mkdir -p "$UNIT83_RUN/stage" "$UNIT83_RUN/run" "$UNIT83_RUN/workspace"
cp -R "$UNIT83_SOURCE/registered/." "$UNIT83_RUN/stage/"
shasum -a 256 "$UNIT83_DESIGN"
"$UNIT83_PY" --version
export P21LANE_WORKSPACE="$UNIT83_RUN/workspace"
export P21LANE_FREEZE_FILE="$UNIT83_DESIGN"
export PYTHONDONTWRITEBYTECODE=1
```

The design hash must match the value above. The original registration manifest must remain unchanged. Credentials are loaded separately by the authorised operator, without echoing them. The original run-start record must be completed before the first command in the next section; it names the registration, the public-state receipt, manifest and preflight hashes, current prices and credential presence without values.

## Original study-command order, only after the requirements are met

The preflight is `"$UNIT83_PY" -m p21lane.lane preflight --registration mn73w` from the copied stage. Complete the original preflight and run-start record first. The sequence then follows run-sheet sections 4.1 to 4.7:

```sh
set -eu
cd "$UNIT83_RUN/stage"
"$UNIT83_PY" -m p21lane.lane attest --registration mn73w
caffeinate -i "$UNIT83_PY" tests/run_tests.py > "$UNIT83_RUN/run/tests-output.txt" 2>&1
```

Stop on a failed command or any failed test. Do not run later steps unless the authored acceptance finishes with zero failures. The next commands are:

```sh
caffeinate -i "$UNIT83_PY" -m p21lane.lane init
"$UNIT83_PY" -m p21lane.lane preflight --registration mn73w
caffeinate -i "$UNIT83_PY" -m p21lane.lane admit
```

Stop to read admission's required operational figures before the runtime freeze, as the original run sheet requires. This is not a chance to inspect outcomes or change registered choices. Refresh preflight whenever its six-hour limit requires it, including a restart. Then:

```sh
caffeinate -i "$UNIT83_PY" -m p21lane.lane run --registration mn73w
"$UNIT83_PY" -m p21lane.lane status
```

`run` performs the runtime freeze/deposit and remaining collection/look/report steps in order. It does not replace the separate authored tests or the admission reading. Preserve the report, journal, deviations and original outputs, then seal and timestamp them using the approved run sheet. Commands shown here have not been executed for this package.

## Recovery and replication limits

The original bank seed is derived from the registered seed, the OSF registration identifier and its `date_registered` value, with the prescribed retry counter only for a failed draw-dependent property. A new workspace using the same registration therefore rebuilds the same deterministic sequence. A new bank or a different provider/model/configuration needs a separately documented successor; it must not be labelled an unchanged repetition of this registered implementation.

Recovery Amendment 01 authorised one identified unchanged-request retry after a policy rejection in the original run. It was a post-registration deviation. Its helper checks that historical incident and its existing evidence. Do not run `historical-recovery/resume-unit83.zsh` or apply its authorisation to another rejection. Its full historical capsule and incident/deposit/application records are preserved with the original results on OSF; the source subset here is for inspection.

The runtime freeze recorded tier 3 and four pairs in flight on each of the two admitted engines. These are facts about the original admission, not replacement settings for a new run. The original statistical report must always travel with `RECOVERY-AMENDMENT-DISCLOSURE.md`.
