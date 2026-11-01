"""The lane's interface: the entry points the v7 harness and the author call (41-D-COUNTED, section 12, package H;
sections 7.4, 11.1 to 11.3 and 14.5).

Package H of the build list (DESIGN-FREEZE.md, section 12). Nothing here is imported or executed before both
registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

Every command is run by the author in an ordinary terminal logged into no AI, from the stage folder (the folder that
holds p21base and p21lane), with the workspace named in P21LANE_WORKSPACE; the bench prints each command exactly:

  python3 -m p21lane.lane preflight --registration ID     the pre-flight read-back (section 14.5, item 5)
  caffeinate -i python3 -m p21lane.lane run --registration ID
                                                         the one command (section 14.5, item 6)
  python3 -m p21lane.lane recover                         a hold the harness does not release itself
  python3 -m p21lane.lane stop_at --date YYYY-MM-DD       a posted price change (section 3.4)
  python3 -m p21lane.lane status                          the steps done and the next one; no result
  python3 -m p21lane.lane <step>                          one registered step on its own (the v7 harness's call)

`run` calls the registered steps in their order: attest, init, admit, freeze, collect_stage_one, look,
collect_stage_two and report; each is also an entry point of its own name. Each step refuses to run out of order or before the registration gate is satisfied;
`run` applies the first look's decision itself (section 7.4), stops at a failed step with the step named, and, given
again, resumes at the first step not done. The harness prints at the first look only the decision to stop or continue
and the SHA-256 of the sealed look record; no command prints a look record until the report (sections 6.4 and 7.4).

Money and keys. Only the steps that call a provider (admit, collect_stage_one, collect_stage_two) read a key, and only
inside the provider client, which takes the study lock and journals every attempt against this lane's caps. Before the
first of them in an invocation of `run`, a pre-flight that passed within six hours, recorded after both registrations
read REGISTERED, must exist; the first paid step writes the run-start record (docs/EDEN_LAB_STANDARD.md). The pre-flight
reads each key once, against the provider's free model-retrieval endpoint. Nothing else here reads a key or calls a
network address, apart from the keyless reads of the registry's public interface (attest, init, the pre-flight) and the
deposit of the runtime freeze with the component token (freeze).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from p21base import ledger, providers
from p21base import registration_gate as gate
from p21base.common import GateError, digest, load, now, save

# The modules admit and freeze are imported under other names, so that the lane's entry points may carry the
# names the build list gives them (section 12, package H): attest, init, admit, freeze, collect_stage_one, look,
# collect_stage_two and report.
from . import admit as admitting
from . import freeze as freezing
from . import analysis, collect, preflight, score, seal, sentences

STEPS = ('attest', 'init', 'admit', 'freeze', 'collect_stage_one', 'look', 'collect_stage_two', 'report')
PAID_STEPS = ('admit', 'collect_stage_one', 'collect_stage_two')
STATE = 'lane_state.json'
RUN_START = 'run_start.json'
LOOKS = 'looks'

# The fixed list from which the author chooses a recovery entry's cause (section 11.2, item 6), so that no line is
# typed by hand. Each cause carries its action.
RECOVERY_CAUSES = (
    ('transport_outage_ended', 'reissue',
     'the provider\'s outage, capacity shortage or rate limiting has ended (a transport hold the harness did not release)'),
    ('request_repaired_before_freeze', 'reissue',
     'a rejected request was put right by a logged defect repair before the runtime freeze'),
    ('tariff_or_usage_reconciled', 'reissue',
     'a tier, usage or charge hold was reconciled with the provider\'s own statement'),
    ('abandon_study', 'abandon_study',
     'the study is abandoned: no further paid call is made'),
)


class LaneStop(GateError):
    """A step could not finish; `step` names it. Given again, `run` resumes at the first step not done."""
    def __init__(self, step, message):
        super().__init__(message)
        self.step = step


# ---- The workspace, the state and the configuration.

def workspace():
    """The workspace the author named in P21LANE_WORKSPACE."""
    value = os.environ.get(gate.WORKSPACE_ENV)
    if not value:
        raise LaneStop('workspace', 'name the workspace in ' + gate.WORKSPACE_ENV + ' as the bench prints it')
    root = Path(value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def raw_config():
    """The configuration as it stands in the stage, before the attestation exists (attest and the pre-flight). Every
    later step reads it through the gate, which checks its bytes against the registered manifest."""
    return gate._read(gate.stage_root() / gate.CONFIG_REL)


def read_state(root):
    path = Path(root) / STATE
    if path.exists():
        return load(path)
    return {'study_id': gate.STUDY_ID, 'done': {}, 'decision': None, 'stop_detail': None}


def write_state(root, state):
    save(Path(root) / STATE, state)


def mark(root, state, step, status, detail=None):
    state['done'][step] = {'status': status, 'utc': now(), 'detail': detail}
    write_state(root, state)


def require_order(state, step):
    """Every earlier step is done or skipped; nothing runs out of order."""
    for earlier in STEPS[:STEPS.index(step)]:
        if earlier not in state['done']:
            raise LaneStop(step, 'the step %s is not done; steps run in the registered order: %s' % (earlier, ', '.join(STEPS)))


def gated(root):
    """The registration gate (section 11.3, step 1 onwards), then the registered configuration."""
    named = gate.require_registered(root)
    return gate.load_config(named)


class Session:
    """What one invocation shares: the workspace, the configuration and one provider client, created when the first
    paid step needs it (the client holds the study lock for the life of the process)."""

    def __init__(self, root, config=None, *, ask=input, show=print):
        self.root = Path(root)
        self.config = config
        self.ask, self.show = ask, show
        self.started = datetime.datetime.now(datetime.timezone.utc)
        self._client = None
        self.preflight_checked = False

    def client(self, expected=None):
        if self._client is None:
            self._client = providers.Client(self.root, self.config, live=True, expected_models=expected or {},
                                            on_event=lambda record: collect.append_deviation(self.root, record))
        elif expected:
            self._client.expected.update(expected)
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


def require_preflight(session, step):
    """A pre-flight that passed within the six hours before this invocation began, recorded after both registrations
    read REGISTERED, before the first paid call of the invocation (docs/EDEN_LAB_STANDARD.md)."""
    if session.preflight_checked:
        return
    if not preflight.current(session.root, at=session.started):
        raise LaneStop(step, 'no pre-flight passed within the six hours before this command: run the pre-flight the '
                             'bench prints (preflight --registration ID), then this command again')
    session.preflight_checked = True


def write_run_start(session):
    """The run-start record, written once before the first paid call (docs/EDEN_LAB_STANDARD.md)."""
    path = session.root / RUN_START
    if path.exists():
        return load(path)
    latest = preflight.latest(session.root)
    record = {'study_id': gate.STUDY_ID, 'utc': now(), 'preflight_utc': latest['utc'],
              'preflight_sha256': digest(json.dumps(latest, sort_keys=True).encode('utf-8')),
              'attestation_sha256': digest((session.root / gate.ATTESTATION).read_bytes()),
              'bank_commitment_sha256': digest((session.root / seal.COMMITMENT).read_bytes())}
    save(path, record, exclusive=True)
    return record


# ---- The steps, in the registered order (section 11.3).

def step_attest(session, registration_id):
    """Step 1: read both registrations from the registry's public interface, show what was read and ask the author to
    confirm it; then the gate checks every registered file's bytes."""
    root = session.root
    state = read_state(root)
    if 'attest' in state['done']:
        return state['done']['attest']['detail']
    if not (root / gate.ATTESTATION).exists():
        if not registration_id:
            raise LaneStop('attest', 'give the unit\'s registration identifier as the bench prints it (--registration ID)')
        gate.attest(root, registration_id, raw_config(), confirm=session.ask, show=session.show)
    session.config = gated(root)
    record = gate.read_attestation(root)
    mark(root, state, 'attest', 'done', {'registration_id': record['registration_id'],
                                         'date_registered': record['date_registered']})
    return record


def step_init(session):
    """Step 3: the three tiers' banks, schedules and public sets, sealed (section 4.9). It cannot be repeated in the
    same workspace; a failure of a draw-dependent property moves to the next bank of the registered sequence."""
    root = session.root
    state = read_state(root)
    require_order(state, 'init')
    session.config = gated(root)
    if not (root / seal.COMMITMENT).exists():
        seal.init(root, session.config)
    commitment = seal.verify(root)
    mark(root, state, 'init', 'done', {'counter': commitment['derivation']['counter'],
                                       'commitment_sha256': digest((root / seal.COMMITMENT).read_bytes())})
    return commitment


def _final_not_admitted(root, receipt):
    # Each excluded engine with the set it failed, so that the sentence names the engine (section 10).
    excluded = {e: row.get('failed') for e, row in receipt['engines'].items() if not row.get('admitted')}
    record = {'study_id': gate.STUDY_ID, 'look': 1, 'mark': analysis.NOT_EVALUABLE, 'reason': 'not_admitted',
              'reason_text': 'fewer than two engines were admitted', 'admitted': receipt['admitted'],
              'excluded': excluded, 'utc': now()}
    path = root / LOOKS / 'look1_incomplete.json'
    if not path.exists():
        save(path, record, exclusive=True)
    return record


def step_admit(session):
    """Steps 4 and 5: admission and calibration on the public sets, and the rule for pairs in flight (section 11.3).
    With fewer than two engines admitted the lane does not collect, and P21 is NOT EVALUABLE at the cost of admission
    alone (section 11.1)."""
    root = session.root
    state = read_state(root)
    require_order(state, 'admit')
    session.config = config = gated(root)
    seal.verify(root)
    require_preflight(session, 'admit')
    write_run_start(session)
    client = session.client()
    ctx = collect.Context(root, config, client, tier=None, in_flight={}, generating={})
    sets = seal.load_public(root)
    folder = root / 'admission'
    receipt = load(folder / 'admission_receipt.json') if (folder / 'admission_receipt.json').exists() else admitting.run(root, config, ctx, sets)
    if not receipt['enough_engines']:
        _final_not_admitted(root, receipt)
        state['decision'] = 'stop'
        for step in ('freeze', 'collect_stage_one', 'look', 'collect_stage_two'):
            state['done'][step] = {'status': 'skipped', 'utc': now(), 'detail': 'fewer than two engines were admitted'}
        mark(root, state, 'admit', 'done', {'admitted': receipt['admitted'], 'enough_engines': False})
        return receipt
    calibration = (load(folder / 'calibration_receipt.json') if (folder / 'calibration_receipt.json').exists()
                   else admitting.calibrate(root, config, ctx, sets, receipt['admitted']))
    flights = load(folder / 'in_flight.json') if (folder / 'in_flight.json').exists() else admitting.in_flight(root, config, receipt)
    mark(root, state, 'admit', 'done', {'admitted': receipt['admitted'], 'tier': calibration['tier'],
                                        'pairs_in_flight': {e: row['pairs_in_flight'] for e, row in flights['engines'].items()}})
    return receipt


def step_freeze(session):
    """Step 6: the runtime freeze record, deposited on the unit's OSF component and read back by hash, before the first
    study call. If the upload fails, the record is uploaded by hand and the same command resumes."""
    root = session.root
    state = read_state(root)
    require_order(state, 'freeze')
    session.config = config = gated(root)
    if not (root / freezing.FREEZE).exists():
        freezing.write(root, config)
    freezing.check_frozen(root)
    try:
        record = freezing.deposit(root, config)
    except Exception as exc:
        raise LaneStop('freeze', 'the runtime freeze could not be deposited and read back (%s: %s); upload %s and %s to the '
                                 'unit\'s component by hand, then run the same command' % (type(exc).__name__, exc,
                                                                                          freezing.FREEZE, seal.COMMITMENT))
    freezing.authorisation(root)
    mark(root, state, 'freeze', 'done', {'freeze_sha256': record['freeze_sha256']})
    return record


def _collection_setup(session, step):
    root = session.root
    config = session.config
    freezing.check_frozen(root)
    freezing.authorisation(root)
    seal.verify(root)
    receipt = load(root / 'admission' / 'admission_receipt.json')
    calibration = load(root / 'admission' / 'calibration_receipt.json')
    flights = load(root / 'admission' / 'in_flight.json')
    require_preflight(session, step)
    client = session.client(expected=receipt['expected_models'])
    in_flight = {e: flights['engines'][e]['pairs_in_flight'] for e in receipt['admitted']}
    ctx = collect.Context(root, config, client, tier=calibration['tier'], in_flight=in_flight,
                          generating=receipt['generating_admitted'])
    return ctx, receipt['admitted'], calibration['tier']


def print_projection(session, first, last):
    """The projection the harness prints before collection begins (section 11.3): from admission's median call time
    and the pairs in flight on each engine."""
    flights = load(session.root / 'admission' / 'in_flight.json')
    rule = session.config['pairs_in_flight']
    for engine, row in sorted(flights['engines'].items()):
        median = row.get('median_call_seconds') or rule['assumed_call_seconds']
        n = row['pairs_in_flight']
        days = -(-(last - first + 1) // n) * rule['rounds'] * median / 86400.0
        session.show('Projection for %s: pairs %d to %d, %d in flight, a median call of %.0f seconds: about %.1f days.'
                     % (session.config['display_names'].get(engine, engine), first, last, n, median, days))


def abandoned(root):
    """Whether the recovery log records that the author abandoned the study (section 11.2, item 6): the provider client
    then makes no further paid call, and the run's stop is a stop before a look (section 7.4)."""
    log = ledger.read_recovery_log(Path(root) / 'private' / ledger.RECOVERY_LOG)
    return any(entry['action'] == 'abandon_study' for entry in log['entries'])


def _stage(session, step, first, last, look_number):
    """One collection stage and what follows from where each engine ended (sections 7.4 and 11.1)."""
    root = session.root
    state = read_state(root)
    require_order(state, step)
    session.config = gated(root)
    if abandoned(root):
        # An abandoned run is a stop before a look (section 7.4). No client is built, since it refuses once the study
        # is abandoned; each admitted engine short of the stage's size is stopped for good, and the record of the look
        # not taken follows below, so that the report can still be written.
        freezing.check_frozen(root)
        engines = load(root / 'admission' / 'admission_receipt.json')['admitted']
        status = {'utc': now(), 'first': first, 'last': last, 'engines': {
            e: {'engine': e, 'completed_prefix': collect.completed_prefix(root, e), 'stopped': collect.STOP_ABANDONED,
                'target': last} for e in engines}}
    else:
        ctx, engines, tier = _collection_setup(session, step)
        print_projection(session, first, last)
        status = collect.stage(ctx, first, last)
    short = {e: r for e, r in status['engines'].items() if r['completed_prefix'] < last}
    if not short:
        mark(root, state, step, 'done', {'first': first, 'last': last})
        return status
    permanent = {e: r for e, r in short.items() if r['stopped'] in collect.PERMANENT_STOPS}
    if not permanent:
        detail = '; '.join('%s stopped (%s) at %d pairs' % (e, r['stopped'], r['completed_prefix']) for e, r in sorted(short.items()))
        raise LaneStop(step, detail + '; release any hold with recover, then run the same command to resume')
    # A stop short of a look's size takes no look on fewer pairs (section 7.4): before the first look the run is NOT
    # EVALUABLE, administratively incomplete; between the looks the first look's INCONCLUSIVE stands with the reason
    # that the second look was not reached.
    engine, row = sorted(permanent.items())[0]
    stop_detail = {'engine': engine, 'completed_prefix': row['completed_prefix'], 'reason': row['stopped']}
    stops = collect.read_engine_stops(root)
    completed = {e: collect.completed_prefix(root, e) for e in engines}
    record = analysis.incomplete(session.config, look_number, stops, completed)
    path = root / LOOKS / ('look%d_incomplete.json' % look_number)
    if not path.exists():
        save(path, record, exclusive=True)
    state['decision'] = 'stop'
    state['stop_detail'] = stop_detail
    if look_number == 1:
        state['done']['look'] = {'status': 'skipped', 'utc': now(), 'detail': 'administratively incomplete'}
    state['done']['collect_stage_two'] = state['done'].get('collect_stage_two') or {'status': 'skipped', 'utc': now(), 'detail': 'stopped'}
    mark(root, state, step, 'stopped', stop_detail)
    return status


def step_collect_stage_one(session):
    """Step 7: pairs 1 to the first look's size on each engine."""
    sizes = session.config['looks']['sizes'] if session.config else gated(session.root)['looks']['sizes']
    return _stage(session, 'collect_stage_one', 1, sizes[0], 1)


def _look_failure(root, number, exc):
    """The look-failure rule (section 7.4): the error trace, with no result in it, kept where no command prints it,
    and its SHA-256 recorded for the repair, which is made from the trace alone."""
    trace = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    folder = root / LOOKS
    folder.mkdir(parents=True, exist_ok=True)
    name = 'look%d_failure' % number
    if (folder / (name + '.json')).exists():
        name = '%s_again_%s' % (name, now().replace(':', '').replace('+0000', 'Z'))
    (folder / (name + '_trace.txt')).write_text(trace, encoding='utf-8')
    record = {'look': number, 'utc': now(), 'exception': type(exc).__name__,
              'error_trace_sha256': hashlib.sha256(trace.encode('utf-8')).hexdigest(), 'trace_file': name + '_trace.txt'}
    save(folder / (name + '.json'), record, exclusive=True)
    return record


def compute_look(root, config, tier, engines, number, name):
    """Score the pairs a look reads, compute the look and seal it; returns (record, sha256). A look already sealed
    under that name (a stop between the sealing and the state's update) is read back, never computed twice. Pure apart
    from the files it writes; the gate and the order are the caller's."""
    sealed = Path(root) / LOOKS / (name + '.json')
    if sealed.exists():
        return load(sealed), digest(sealed.read_bytes())
    size = config['looks']['sizes'][number - 1]
    seal.verify(root)
    score.table(root, config, tier, engines, size)
    score.run(root)
    arrays = analysis.load_arrays(root, config, tier, engines, size)
    record = analysis.look(arrays, size, number, config)
    record['utc'] = now()
    sha = analysis.seal_record(root, record, name)
    return record, sha


def run_first_look(root, config, state, tier, engines, show=print):
    """Step 8, given the gate: the first look, sealed; prints only the decision and the sealed record's SHA-256. A look
    that cannot be computed follows the look-failure rule: collection continues to the second look's size and the first
    look is computed for the record later; it cannot stop the run."""
    try:
        record, sha = compute_look(root, config, tier, engines, 1, 'look1')
    except Exception as exc:
        failure = _look_failure(root, 1, exc)
        state['decision'] = 'continue'
        state['look1_failed'] = True
        show('FIRST LOOK: it could not be computed; collection continues to the second look\'s size (section 7.4). '
             'Error trace SHA-256 %s.' % failure['error_trace_sha256'])
        return 'continue', None
    decision = analysis.decide(record)
    state['decision'] = decision
    state['look1_sha256'] = sha
    show('FIRST LOOK: %s; sealed record SHA-256 %s.' % (decision, sha))
    return decision, sha


def step_look(session):
    """Step 8: the first look, by the harness, at the first look's level."""
    root = session.root
    state = read_state(root)
    require_order(state, 'look')
    session.config = config = gated(root)
    freezing.check_frozen(root)
    receipt = load(root / 'admission' / 'admission_receipt.json')
    tier = load(root / 'admission' / 'calibration_receipt.json')['tier']
    decision, sha = run_first_look(root, config, state, tier, receipt['admitted'], show=session.show)
    if decision == 'stop':
        state['done']['collect_stage_two'] = {'status': 'skipped', 'utc': now(), 'detail': 'stopped at the first look'}
    mark(root, state, 'look', 'done', {'decision': decision, 'sealed_sha256': sha})
    return decision


def step_collect_stage_two(session):
    """Step 9: pairs after the first look's size to the second look's size, unless stopped."""
    state = read_state(session.root)
    if state.get('decision') != 'continue':
        mark(session.root, state, 'collect_stage_two', 'skipped', 'stopped at the first look')
        return None
    sizes = (session.config or gated(session.root))['looks']['sizes']
    return _stage(session, 'collect_stage_two', sizes[0] + 1, sizes[1], 2)


def run_second_look(root, config, state, tier, engines, show=print):
    """Step 10, given the gate: where the first look failed, it is computed now for the record; then the second look,
    sealed. If the second look cannot be computed, the data stand and the look-failure rule applies to the report."""
    if state.get('look1_failed') and not (root / LOOKS / 'look1_record.json').exists():
        try:
            compute_look(root, config, tier, engines, 1, 'look1_record')
        except Exception as exc:
            _look_failure(root, 1, exc)
    try:
        record, sha = compute_look(root, config, tier, engines, 2, 'look2')
    except Exception as exc:
        failure = _look_failure(root, 2, exc)
        raise LaneStop('report', 'the second look could not be computed; the data stand. A repair from the error trace '
                                 'alone (SHA-256 %s) under section 7.4, then the same command' % failure['error_trace_sha256'])
    state['look2_sha256'] = sha
    return record


def step_report(session):
    """Step 10: the second look where the run went on, then the report (sections 10, 11.4 and 11.5)."""
    root = session.root
    state = read_state(root)
    require_order(state, 'report')
    session.config = config = gated(root)
    second = state.get('decision') == 'continue' and state['done'].get('collect_stage_two', {}).get('status') == 'done'
    if second or (state.get('look1_failed') and not (root / LOOKS / 'look1_record.json').exists()):
        freezing.check_frozen(root)
        receipt = load(root / 'admission' / 'admission_receipt.json')
        tier = load(root / 'admission' / 'calibration_receipt.json')['tier']
        if second:
            run_second_look(root, config, state, tier, receipt['admitted'], show=session.show)
        else:
            # The first look failed and the second was not reached: the first is computed for the record (section 7.4).
            try:
                compute_look(root, config, tier, receipt['admitted'], 1, 'look1_record')
            except Exception as exc:
                _look_failure(root, 1, exc)
        write_state(root, state)
    path = sentences.report(root, config)
    mark(root, state, 'report', 'done', {'report': str(path.relative_to(root))})
    session.show('REPORT written: ' + str(path))
    return path


STEP_FUNCTIONS = {
    'init': step_init, 'admit': step_admit, 'freeze': step_freeze, 'collect_stage_one': step_collect_stage_one,
    'look': step_look, 'collect_stage_two': step_collect_stage_two, 'report': step_report,
}


# ---- The commands.

def run(registration_id=None, *, ask=input, show=print):
    """The one command (section 14.5, item 6): every step not yet done, in the registered order; it stops at a failed
    step with the step named and, given again, resumes at the first step not done."""
    root = workspace()
    session = Session(root, ask=ask, show=show)
    try:
        for step in STEPS:
            state = read_state(root)
            if step in state['done']:
                continue
            try:
                if step == 'attest':
                    step_attest(session, registration_id)
                else:
                    STEP_FUNCTIONS[step](session)
            except LaneStop as stop:
                show('STOP at step %s: %s' % (stop.step, stop))
                return 1
            except Exception as exc:
                show('STOP at step %s: %s: %s' % (step, type(exc).__name__, exc))
                return 1
        show('RUN COMPLETE. The report is in report/REPORT.md in the workspace.')
        return 0
    finally:
        session.close()


def preflight_command(registration_id, *, ask=input, show=print):
    """The pre-flight read-back (section 14.5, item 5; package I, item 2): the registrations read from the registry,
    then for each provider the key valid, the model string served, the balance read by hand on the provider's billing
    page against the larger of the planned spend with its margin and this lane's cap, the provider-side spend limit and
    the price pages read on the day. A failed read-back stops the run before its first call."""
    root = workspace()
    if not registration_id:
        show('Give the unit\'s registration identifier as the bench prints it (--registration ID).')
        return 1
    config = raw_config()
    registry = preflight.registry_reads(registration_id, config)
    record = preflight.readback(root, config, ask=ask, show=show, registry=registry)
    return 0 if record.get('passed') else 1


def transport_holds_awaiting(root):
    """Transport holds the harness did not release (a seventh in a UTC day): the attempts the client reported as
    transport_hold in the deviation record that no reissue entry of the recovery log names."""
    released = ledger.reissue_entries(Path(root) / 'private' / ledger.RECOVERY_LOG)
    out = []
    for record in collect.read_deviations(root):
        if record.get('kind') == 'transport_hold' and record.get('request_id') and record['request_id'] not in released:
            if record['request_id'] not in out:
                out.append(record['request_id'])
    return out


def recover(*, choose=input, show=print):
    """`recover` (section 11.2, item 6): for every hold of a class the harness does not release itself, an append-only
    recovery entry written from a cause the author chooses from a fixed list, without reading any score. It takes the
    study lock, so it cannot run while a run does."""
    root = workspace()
    config = gated(root)
    lock = ledger.hold_study_lock(root / 'private' / 'STUDY.lock')
    try:
        caps = {spec['budget_key']: float(spec['cap_usd']) for spec in config['engines'].values()}
        journal = ledger.Ledger(root / 'private' / 'requests.sqlite', caps, sum(caps.values()))
        held = list(journal.holds()) + [r for r in transport_holds_awaiting(root) if r not in journal.holds()]
        if not held:
            show('No hold awaits a recovery entry.')
            return 0
        show('Attempts held:')
        for ident in held:
            row = journal.get(ident)
            detail = json.loads(row['detail']) if row else {}
            slot = (detail.get('reservation_detail') or detail).get('slot') if isinstance(detail, dict) else None
            show('  %s on %s, slot %s, state %s' % (ident[:16], row['provider'] if row else '?', slot, row['state'] if row else '?'))
        show('Causes:')
        for number, (_, _, words) in enumerate(RECOVERY_CAUSES, 1):
            show('  %d. %s' % (number, words))
        answer = str(choose('Number of the cause that applies to every attempt listed: ')).strip()
        if not answer.isdigit() or not 1 <= int(answer) <= len(RECOVERY_CAUSES):
            show('No entry was written.')
            return 1
        cause, action, words = RECOVERY_CAUSES[int(answer) - 1]
        path = root / 'private' / ledger.RECOVERY_LOG
        with path.open('a', encoding='utf-8') as handle:
            for ident in held:
                row = journal.get(ident)
                charge = None if action == 'abandon_study' else (
                    max(row['reservation'], row['charge'] if row['charge'] is not None else 0.0) if row else 0.0)
                entry = {'utc': now(), 'request_id': ident, 'cause': cause, 'action': action,
                         'reconciled_charge_usd': charge, 'written_without_reading_any_score': True}
                handle.write(json.dumps(entry, sort_keys=True) + '\n')
        collect.append_deviation(root, {'kind': 'recovery', 'cause': cause, 'action': action, 'attempts': len(held)})
        show('%d recovery entries written (%s). Run the same run command to resume.' % (len(held), words))
        return 0
    finally:
        os.close(lock)


def stop_at(date_text, *, show=print):
    """`stop_at` (section 3.4): each engine's collection ends at its last completed pair before the date a posted price
    change takes effect."""
    root = workspace()
    gated(root)
    record = collect.write_stop_at(root, date_text)
    collect.append_deviation(root, {'kind': 'stop_at', 'date': record['date']})
    show('Each engine\'s collection ends at its last completed pair before %s.' % record['date'])
    return 0


def status(*, show=print):
    """The steps done or skipped and the next one. It prints no look record; after the first look only the decision
    and the sealed record's SHA-256, which the harness has already printed."""
    root = workspace()
    state = read_state(root)
    for step in STEPS:
        row = state['done'].get(step)
        show('%-18s %s' % (step, row['status'] + ' ' + row['utc'] if row else 'not done'))
    following = [step for step in STEPS if step not in state['done']]
    show('Next: ' + (following[0] if following else 'nothing: the run is complete'))
    if state.get('look1_sha256'):
        show('First look: %s; sealed record SHA-256 %s.' % (state.get('decision'), state['look1_sha256']))
    return 0


def one_step(step, registration_id=None, *, ask=input, show=print):
    """One registered step on its own, as the v7 harness calls it: it refuses to run out of order or before the gate is
    satisfied, and says which step stopped and why."""
    root = workspace()
    session = Session(root, ask=ask, show=show)
    try:
        if step in read_state(root)['done']:
            show('The step %s is done already.' % step)
            return 0
        if step == 'attest':
            step_attest(session, registration_id)
        else:
            STEP_FUNCTIONS[step](session)
        show('Step %s: %s.' % (step, read_state(root)['done'].get(step, {}).get('status', 'not done')))
        return 0
    except LaneStop as stop:
        show('STOP at step %s: %s' % (stop.step, stop))
        return 1
    except Exception as exc:
        show('STOP at step %s: %s: %s' % (step, type(exc).__name__, exc))
        return 1
    finally:
        session.close()


# The entry points of the build list, in their order (section 12, package H); `run` calls them in this order.
def attest(registration_id=None, **kwargs):
    return one_step('attest', registration_id, **kwargs)


def init(**kwargs):
    return one_step('init', **kwargs)


def admit(**kwargs):
    return one_step('admit', **kwargs)


def freeze(**kwargs):
    return one_step('freeze', **kwargs)


def collect_stage_one(**kwargs):
    return one_step('collect_stage_one', **kwargs)


def look(**kwargs):
    return one_step('look', **kwargs)


def collect_stage_two(**kwargs):
    return one_step('collect_stage_two', **kwargs)


def report(**kwargs):
    return one_step('report', **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python3 -m p21lane.lane', description='41-D-COUNTED-v0.1: the lane\'s commands')
    parser.add_argument('command', choices=('run', 'preflight', 'recover', 'stop_at', 'status') + STEPS)
    parser.add_argument('--registration', help='the unit\'s registration identifier, as the bench prints it')
    parser.add_argument('--date', help='for stop_at: the date a posted price change takes effect, YYYY-MM-DD')
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            return run(args.registration)
        if args.command == 'preflight':
            return preflight_command(args.registration)
        if args.command == 'recover':
            return recover()
        if args.command == 'stop_at':
            return stop_at(args.date)
        if args.command in STEPS:
            return one_step(args.command, args.registration)
        return status()
    except LaneStop as stop:
        print('STOP at %s: %s' % (stop.step, stop))
        return 1
    except Exception as exc:
        print('STOP: %s: %s' % (type(exc).__name__, exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
