#!/bin/zsh
# Run only by Michael in his ordinary Terminal. Never source this file.
set +x
set -eu
set -o pipefail
umask 077

unit83_py=/opt/homebrew/bin/python3.13
unit83_amendment="${0:A:h}"
unit83_unit="${unit83_amendment:h:h:h}"
unit83_bench="${unit83_unit:h}/bench/bench.py"

export P21LANE_WORKSPACE="$unit83_unit/workspace"
export PYTHONDONTWRITEBYTECODE=1
export P21LANE_FREEZE_FILE="${unit83_unit:h}/kit/frozen-design/DESIGN-FREEZE.md"

# Check the amendment and original evidence before any network operation.
"$unit83_py" "$unit83_amendment/recovery_amendment.py" prepare

[[ -n "${ANTHROPIC_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" && -n "${OSF_COMPONENT_TOKEN:-}" ]] || {
  print 'STOP: a required credential is absent. Use the original Terminal with the credentials already exported.'
  exit 1
}

cd "$unit83_unit/stage"
print 'Refresh preflight: answer using the current balances and settings.'
"$unit83_py" -m p21lane.lane preflight --registration mn73w
"$unit83_py" "$unit83_bench" done 4.4 --exit 0

print 'Deposit the recovery addendum on OSF and verify its downloaded bytes.'
"$unit83_py" "$unit83_amendment/recovery_amendment.py" deposit
"$unit83_py" "$unit83_amendment/recovery_amendment.py" apply
"$unit83_py" "$unit83_bench" note 'Unit83-RA01: operator-authorised post-registration recovery amendment deposited on OSF zwnc5 before releasing one identified OpenAI invalid_prompt hold. Frozen study code and prompt/model unchanged. See unit/run/amendments/2026-10-28-recovery-01/. Final results require amended-protocol disclosure.'

print 'Resume the existing study from its journal. No automatic policy-hold release loop is enabled.'
unit83_exit=0
caffeinate -i "$unit83_py" -m p21lane.lane run --registration mn73w || unit83_exit=$?
printf 'Study run exit code: %s\n' "$unit83_exit"
if (( unit83_exit != 0 )); then
  print 'STOP: preserve all files and send the STOP/error line. This script will not release another rejection.'
  exit "$unit83_exit"
fi

"$unit83_py" "$unit83_amendment/recovery_amendment.py" disclose
"$unit83_py" "$unit83_bench" done 4.7 --exit 0
print 'Report complete. Read REPORT.md together with RECOVERY-AMENDMENT-DISCLOSURE.md, then seal the results.'
