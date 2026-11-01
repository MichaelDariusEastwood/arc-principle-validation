"""Durable native-USD reservation journal. One local journal per workspace.

Carried from the compact Screen's version 0.3 (`ledger.py`, 171 lines, sha256
6aaa1e02945538b193d0129ac07213d40f93f83caced28ac9bdaae207da6e93d) with its logic unchanged, and changed only as the
design freeze's package A says (DESIGN-FREEZE.md, section 12): one lock serialises every use of the journal, so that
several threads of the one process may use it (`reserve` and `settle` above all); `headroom(key)` gives a budget key's
cap less its settled charges and open reservations; the caps are keyed by the lane's own budget key for each provider,
so that a lane's cap stop leaves the other lanes running; and `settle` takes a zero charge, which the provider client
writes for the failure classes section 11.2 names as uncharged.

Every numbered attempt of a request is one row. A reserved row counts at its full reservation and is never treated as
free. A held row stays unresolved until an entry in the append-only recovery log beside the journal releases it, and a
released row is never recorded below what was already journalled. Actual billing must be reconciled with provider
statements; this journal cannot impose a provider-side hard cap.
"""
from __future__ import annotations
import json, sqlite3, math, os, threading
from datetime import datetime
from pathlib import Path
from typing import Any
from .common import GateError, canonical, now

# Journal states that hold the run until a recovery entry names the row.
HOLD_STATES=('request_rejected','identity_drift','pricing_tier_mismatch','usage_missing','reservation_exceeded')
RECOVERY_LOG='recovery_log.jsonl'
RECOVERY_ACTIONS=('reissue','abandon_study')
RECOVERY_FIELDS=frozenset(('utc','request_id','cause','action','reconciled_charge_usd','written_without_reading_any_score'))

def _unique_fields(pairs: list[tuple[str,Any]]) -> dict[str,Any]:
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate recovery field')
        out[key]=value
    return out

def valid_recovery_entry(value: Any) -> bool:
    """Exactly the six registered fields, a dated cause and action, and the blind-writing statement set to true."""
    if not isinstance(value,dict) or set(value)!=RECOVERY_FIELDS: return False
    if value['written_without_reading_any_score'] is not True: return False
    if not isinstance(value['action'],str) or value['action'] not in RECOVERY_ACTIONS: return False
    for name in ('utc','request_id','cause'):
        if not isinstance(value[name],str) or not value[name].strip(): return False
    try: when=datetime.fromisoformat(value['utc'].replace('Z','+00:00'))
    except ValueError: return False
    if when.tzinfo is None: return False
    charge=value['reconciled_charge_usd']
    if charge is None: return value['action']=='abandon_study'
    return isinstance(charge,(int,float)) and not isinstance(charge,bool) and math.isfinite(charge) and charge>=0

def read_recovery_log(path: str | Path) -> dict[str,Any]:
    """Read the append-only recovery log.

    One JSON object a line: utc, request_id, cause, action (reissue or
    abandon_study), reconciled_charge_usd and written_without_reading_any_score.
    A line that is not a complete valid entry releases nothing. Its line number
    is reported, so that a corrected entry can be appended and nothing edited.
    """
    path=Path(path); entries=[]; invalid=[]
    if path.exists():
        for number,line in enumerate(path.read_bytes().decode('utf-8',errors='replace').split('\n'),1):
            if not line.strip(): continue
            try: value=json.loads(line,object_pairs_hook=_unique_fields)
            except (ValueError,RecursionError): value=None
            if valid_recovery_entry(value): entries.append({**value,'line':number})
            else: invalid.append(number)
    return {'entries':entries,'invalid_lines':invalid}

def reissue_entries(path: str | Path) -> dict[str,dict[str,Any]]:
    """Each identifier named by a valid reissue entry, with that entry. Of several for one identifier, the largest reconciled charge."""
    out={}
    for entry in read_recovery_log(path)['entries']:
        if entry['action']!='reissue': continue
        key=entry['request_id'].strip(); prior=out.get(key)
        if prior is None or entry['reconciled_charge_usd']>prior['reconciled_charge_usd']: out[key]=entry
    return out

def hold_study_lock(path: str | Path) -> int:
    """Take the one-process study lock and keep it for the life of the caller.

    The operating system releases the lock when the process ends, however it
    ends. A run that was stopped therefore cannot block its own recovery, two
    runs cannot share one journal, and a reserved row seen under this lock
    belongs to a process that has stopped.
    """
    try: import fcntl
    except ImportError as exc: raise GateError('The study lock needs a POSIX host') from exc
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.ftruncate(fd,0); os.write(fd,str(os.getpid()).encode())
    except OSError as exc:
        os.close(fd); raise GateError('The study lock is held by another run or cannot be taken') from exc
    return fd

class Ledger:
    def __init__(self, path: str | Path, caps: dict[str,float], total_cap: float):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        # Package A: one lock serialises every use of the one connection, which several threads of the process share.
        self.lock=threading.RLock()
        self.db=sqlite3.connect(self.path,timeout=30,isolation_level=None,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, provider TEXT NOT NULL, reservation REAL NOT NULL, charge REAL, state TEXT NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL, slot TEXT NOT NULL, role TEXT NOT NULL, UNIQUE(provider,role,slot))')
        if [row[1] for row in self.db.execute('PRAGMA table_info(calls)')]!=['id','provider','reservation','charge','state','detail','created','slot','role']:
            raise GateError('Incompatible historical journal; never migrate or overwrite it implicitly')
        # The caps are keyed by the lane's own budget key for each provider (package A).
        self.caps=caps; self.total_cap=total_cap
    def totals(self) -> dict[str,float]:
        with self.lock:
            return dict(self.db.execute('SELECT provider,SUM(COALESCE(charge,reservation)) FROM calls GROUP BY provider'))
    def headroom(self, provider: str) -> float:
        """The budget key's cap less its settled charges and its open reservations (package A)."""
        with self.lock:
            if provider not in self.caps: raise GateError('unknown budget key')
            return self.caps[provider]-self.totals().get(provider,0.0)
    def get(self, key: str) -> dict[str,Any] | None:
        with self.lock:
            row=self.db.execute('SELECT provider,reservation,charge,state,detail,created FROM calls WHERE id=?',(key,)).fetchone()
        return None if row is None else dict(zip(('provider','reservation','charge','state','detail','created'),row))
    def get_slot(self, provider: str, role: str, slot: str) -> dict[str,Any] | None:
        with self.lock:
            row=self.db.execute('SELECT id,state FROM calls WHERE provider=? AND role=? AND slot=?',(provider,role,slot)).fetchone()
        return None if row is None else dict(zip(('id','state'),row))
    def reserve(self,key: str, provider: str, amount: float, detail: dict[str,Any]) -> None:
        if not math.isfinite(amount) or amount<0 or provider not in self.caps: raise GateError('invalid reservation')
        role=detail.get('role'); slot=detail.get('slot')
        if not isinstance(role,str) or not role or not isinstance(slot,str) or not slot:
            raise GateError('Every request needs a fixed role and slot')
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                if self.get(key) is not None: raise GateError('request already journalled; do not reissue')
                if self.get_slot(provider,role,slot) is not None:
                    raise GateError('Request slot already journalled; a different body cannot authorise a repeat')
                totals=self.totals()
                if totals.get(provider,0)+amount>self.caps[provider]+1e-10 or sum(totals.values())+amount>self.total_cap+1e-10:
                    raise GateError('budget reservation would exceed cap')
                self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)',
                    (key,provider,amount,None,'reserved',canonical(detail).decode(),now(),slot,role))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK'); raise
    def settle(self,key: str, charge: float | None, state: str, detail: dict[str,Any]) -> None:
        # A zero charge is accepted: the client writes it for the failure classes section 11.2 names as uncharged.
        with self.lock:
            old=self.get(key)
            if old is None or old['state']!='reserved': raise GateError('cannot settle non-pending request')
            if charge is not None and (not math.isfinite(charge) or charge<0): raise GateError('invalid charge')
            preserved={'reservation_detail':json.loads(old['detail']),'settlement_detail':detail}
            self.db.execute('UPDATE calls SET charge=?,state=?,detail=? WHERE id=?',
                            (charge,state,canonical(preserved).decode(),key))
    def recovery(self) -> dict[str,Any]:
        """The recovery log beside this journal: its valid entries and the numbers of its invalid lines."""
        return read_recovery_log(self.path.with_name(RECOVERY_LOG))
    def released(self) -> dict[str,dict[str,Any]]:
        """Request identifiers that a valid reissue entry names, with the entry."""
        return reissue_entries(self.path.with_name(RECOVERY_LOG))
    def unresolved(self) -> list[str]:
        """Rows still reserved or without a charge, except those a reissue entry names."""
        released=self.released()
        with self.lock:
            rows=[r[0] for r in self.db.execute("SELECT id FROM calls WHERE state='reserved' OR charge IS NULL")]
        return [r for r in rows if r not in released]
    def holds(self) -> list[str]:
        """Settled rows that hold the run: a hold state or no charge, and no reissue entry.

        A reserved row is not a hold. It belongs to a process that stopped, and
        the client completes it or charges it in full on the next start.
        """
        released=self.released()
        marks=','.join('?'*len(HOLD_STATES))
        with self.lock:
            rows=[r[0] for r in self.db.execute("SELECT id FROM calls WHERE state<>'reserved' AND (charge IS NULL OR state IN ("+marks+"))",HOLD_STATES)]
        return [r for r in rows if r not in released]
    def holds_by_key(self) -> dict[str,list[str]]:
        """The holds of `holds`, grouped by budget key, so that a hold on one provider leaves the other running."""
        out: dict[str,list[str]]={}
        for ident in self.holds():
            row=self.get(ident)
            out.setdefault(row['provider'],[]).append(ident)
        return out
    def reconcile(self,key: str, charge: float, state: str, detail: dict[str,Any] | None=None) -> None:
        """Record the reconciled charge of a settled row that a recovery entry releases.

        The charge can only stay or rise: never below the full reservation and
        never below a charge already journalled. The earlier state and charge
        are kept in the row's detail.
        """
        with self.lock:
            old=self.get(key)
            if old is None or old['state']=='reserved': raise GateError('cannot reconcile an unknown or pending request')
            if not isinstance(state,str) or not state or state=='reserved': raise GateError('invalid reconciled state')
            if isinstance(charge,bool) or not isinstance(charge,(int,float)) or not math.isfinite(charge) or charge<0:
                raise GateError('invalid charge')
            floor=max(old['reservation'],old['charge'] if old['charge'] is not None else 0.0)
            if charge<floor-1e-12: raise GateError('A reconciled charge cannot be lower than the amount already journalled')
            preserved=json.loads(old['detail'])
            preserved.setdefault('reconciliations',[]).append({'previous_state':old['state'],'previous_charge':old['charge'],
                                                             'reconciled_utc':now(),'recovery_entry':detail})
            self.db.execute('UPDATE calls SET charge=?,state=?,detail=? WHERE id=?',
                            (charge,state,canonical(preserved).decode(),key))
