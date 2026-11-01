"""Native, non-streaming adapters for the two engines of 41-D-COUNTED, with no silent model fallback.

Carried from the compact Screen's version 0.3 (`providers.py`, 332 lines, sha256
86e2930437a970d47c5922a6f46a81cc55b353d84380208aa12b4db4d9349ecd): its attempt and recovery logic (the base's lines 171
to 332) is kept, and the rest is changed as the design freeze's package A requires (DESIGN-FREEZE.md, sections 3.1, 3.4,
11.2 and 12), with the Anthropic branch following ANTHROPIC-ADAPTER-SPEC.md, sections 2 to 8. Nothing here is imported
or executed before both registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a
parse of this file's text.

Recovery, every rule blind to outcomes. A request is issued as numbered attempts: the first under its slot, attempt k
under slot#attemptk. One journalled attempt is one HTTP exchange. A transport failure (HTTP 408, 409, 425, 429, 5xx or
529, a time-out, a connection error, a body that is not JSON, or an empty reply with no output tokens) is issued again
on the schedule of section 11.2 for the request's tier: on the standard tier after 30, 120 and 600 seconds, on the flex
tier after 30, 120, 600, 1,800 and 3,600 seconds, each wait the larger of that delay and the provider's recorded
`retry-after`, up to 3,600 seconds. A failure the provider documents as uncharged (HTTP 429 or 529, a 5xx with an error
body, flex capacity unavailable, which OpenAI returns as a 429) is settled at zero; a time-out, a disconnection, a body
that is not JSON and every other transport failure is settled at its full reservation; an empty reply with no output
tokens is settled from its usage where it carries one. After the last attempt the call raises TransportHold, and the
collector holds that provider while the other continues. A request the provider rejects as malformed or unauthorised,
a tier other than the registered one, missing usage or a charge above its reservation holds that provider until an
entry in the append-only recovery log names the attempt. An attempt that an earlier process reserved and did not settle
is completed from its saved response, or charged in full and issued again when no response was saved.

A truncated reply and a refusal are usable results that are not `ok`: charged from their usage and returned, so that
the collector records them and asks again under the re-ask of section 11.2, item 8; no text of theirs is read. A reply
whose returned `model` differs from the admitted value is returned with the status `model_changed`, settled at no less
than its reservation, and the collector ends that engine's collection at its last completed pair (section 3.1). No
other model is ever asked.
"""
from __future__ import annotations
import datetime, json, math, os, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any
from .common import GateError, canonical, digest, load, now, save
from .ledger import Ledger, hold_study_lock

ANTHROPIC='anthropic'
OPENAI='openai'
ENDPOINTS={ANTHROPIC:'https://api.anthropic.com/v1/messages', OPENAI:'https://api.openai.com/v1/responses'}
MODEL_ENDPOINTS={ANTHROPIC:'https://api.anthropic.com/v1/models/', OPENAI:'https://api.openai.com/v1/models/'}
IDS={ANTHROPIC:'claude-haiku-5-5', OPENAI:'gpt-6-luna'}
ENVS={ANTHROPIC:'ANTHROPIC_API_KEY', OPENAI:'OPENAI_API_KEY'}
TIERS={ANTHROPIC:'standard', OPENAI:'flex'}
ANTHROPIC_VERSION='2023-06-01'
USER_AGENT='p21-lane-client'

# Seconds waited after a request's first, second, ... transport failure, before its next attempt (section 11.2, item 1).
RETRY_DELAYS={'standard':(30,120,600), 'flex':(30,120,600,1800,3600)}
RETRY_AFTER_CEILING=3600
# HTTP statuses that leave no usable response for a transport reason, with every 5xx.
TRANSPORT_HTTP=(408,409,425,429)
# Transport failures the providers document as uncharged (section 11.2, item 1); a 5xx with an error body is uncharged too.
UNCHARGED_HTTP=(429,529)
# Statuses of a response that arrived and must hold the provider until a recovery entry names the attempt.
HOLD_STATUSES=('pricing_tier_mismatch','usage_missing','reservation_exceeded')
# Statuses that are transport failures without an HTTP error status.
TRANSPORT_STATUSES=('transport_unknown','interrupted_charged_in_full','empty_transport')
NO_RESPONSE=('http_failure','transport_unknown','interrupted_charged_in_full')
MODEL_CHANGED='model_changed'
TRUNCATED='truncated'
REFUSED='refused'

# The registered request fields of section 3.1, with their registered values; the text fields are filled per call.
REGISTERED={
    ANTHROPIC:{'model':'claude-haiku-5-5','max_tokens':16000,'inference_geo':'global','stream':False,
               'output_config':{'effort':'low'},'thinking':{'type':'adaptive'}},
    OPENAI:{'model':'gpt-6-luna','max_output_tokens':16000,'reasoning':{'effort':'low'},'service_tier':'flex',
            'store':False,'prompt_cache_options':{'mode':'explicit'}},
}
# Response headers kept: the request identifier, retry-after and the rate-limit set. No header that identifies an
# account, an organisation, a workspace or a project is ever written to disk.
KEPT_HEADERS={ANTHROPIC:('request-id','retry-after'), OPENAI:('x-request-id','retry-after')}
KEPT_HEADER_PREFIXES={ANTHROPIC:('anthropic-ratelimit-',), OPENAI:('x-ratelimit-',)}


class TransportHold(GateError):
    """Every numbered attempt failed for a transport reason: the provider is held, and the other continues."""
    def __init__(self, message: str, row_id: str, budget_key: str):
        super().__init__(message); self.row_id=row_id; self.budget_key=budget_key


class ProviderHeld(GateError):
    """A recorded hold on this provider's budget key: no new call is made on it until a recovery entry releases it."""


class CapReached(GateError):
    """The journal refused a reservation: this lane's cap for the provider would be passed."""


class EngineStopped(GateError):
    """A posted price change (`stop_at`) ends this engine's collection before its next call."""


def attempt_slot(slot: str, attempt: int) -> str:
    return slot if attempt==1 else slot+'#attempt'+str(attempt)


def _count(value: Any) -> int | None:
    return value if isinstance(value,int) and not isinstance(value,bool) and value>=0 else None


def failure_class(result: dict[str,Any]) -> str | None:
    """'transport' (issued again on the tier's schedule), 'hold' (stops the provider until released) or None (a usable result)."""
    status=result.get('status')
    if status in TRANSPORT_STATUSES: return 'transport'
    if status=='http_failure':
        code=result.get('http_status')
        if isinstance(code,int) and not isinstance(code,bool) and (code in TRANSPORT_HTTP or 500<=code<=599): return 'transport'
        return 'hold'
    return 'hold' if status in HOLD_STATUSES else None


def settlement_class(result: dict[str,Any]) -> str:
    """How a transport failure is settled (section 11.2, item 1): 'uncharged', 'usage' or 'reservation'."""
    status=result.get('status')
    if status=='http_failure':
        code=result.get('http_status')
        if code in UNCHARGED_HTTP: return 'uncharged'
        if isinstance(code,int) and 500<=code<=599 and result.get('error_body_json') is True: return 'uncharged'
        return 'reservation'
    if status=='empty_transport' and result.get('input_tokens') is not None and result.get('output_tokens') is not None:
        return 'usage'
    return 'reservation'


def journal_state(result: dict[str,Any]) -> str:
    # A rejected request keeps http_failure and its code in the result; the journal names the hold.
    if result['status']=='http_failure' and failure_class(result)=='hold': return 'request_rejected'
    return result['status']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise GateError('API redirect refused, credentials are not forwarded')


def payload(provider: str, system: str, user: str, fields: dict[str,Any]) -> dict[str,Any]:
    """The request body from the registered request fields of section 3.1; any field outside them, or a field whose
    value differs from its registered value, is refused. The system text goes to Anthropic's `system` and OpenAI's
    `instructions`; the user turn is one user message (section 3.1)."""
    if provider not in REGISTERED: raise GateError('unknown provider')
    registered=REGISTERED[provider]
    extra=sorted(set(fields)-set(registered))
    if extra: raise GateError('request field outside the registered fields: '+', '.join(extra))
    if fields!=registered: raise GateError('a request field differs from its registered value')
    if not isinstance(system,str) or not isinstance(user,str) or not system or not user:
        raise GateError('the system text and the user turn must be non-empty text')
    body=json.loads(json.dumps(registered))
    if provider==ANTHROPIC:
        body['system']=system; body['messages']=[{'role':'user','content':user}]
    else:
        body['instructions']=system; body['input']=[{'role':'user','content':user}]
    return body


def output_cap(provider: str) -> int:
    return REGISTERED[provider]['max_tokens' if provider==ANTHROPIC else 'max_output_tokens']


def normalise(provider: str, body: Any) -> dict[str,Any]:
    """The provider's reply as one result record. Text is gathered only from text blocks; thinking and reasoning
    items are never read. The status keeps the provider's own word for an abnormal finish, except `truncated` and
    `refused`, which the collector records without passing any text on."""
    empty={'status':'malformed','text':'','returned_model':None,'input_tokens':None,'output_tokens':None,
           'reasoning_tokens':None,'response_id':None,'served_tier':None,'refusal_category':None}
    if not isinstance(body,dict): return empty
    out=dict(empty)
    out['returned_model']=body.get('model') if isinstance(body.get('model'),str) else None
    out['response_id']=body.get('id') if isinstance(body.get('id'),str) else None
    usage=body.get('usage') if isinstance(body.get('usage'),dict) else {}
    if provider==ANTHROPIC:
        blocks=body.get('content') if isinstance(body.get('content'),list) else []
        text=''.join(b.get('text','') for b in blocks if isinstance(b,dict) and b.get('type')=='text' and isinstance(b.get('text'),str))
        base=_count(usage.get('input_tokens')); created=usage.get('cache_creation_input_tokens'); read=usage.get('cache_read_input_tokens')
        created=0 if created is None else _count(created); read=0 if read is None else _count(read)
        inp=None if base is None or created is None or read is None else base+created+read
        outp=_count(usage.get('output_tokens'))
        details=usage.get('output_tokens_details')
        reasoning=_count(details.get('thinking_tokens')) if isinstance(details,dict) else None
        tier=usage.get('service_tier')
        stop=body.get('stop_reason')
        if not isinstance(stop,str): status='malformed'
        elif stop=='end_turn': status='ok'
        elif stop in ('max_tokens','model_context_window_exceeded'): status=TRUNCATED
        elif stop=='refusal':
            status=REFUSED
            details=body.get('stop_details') if isinstance(body.get('stop_details'),dict) else {}
            out['refusal_category']=details.get('category') if isinstance(details.get('category'),str) else None
        else: status=stop
        out['served_tier']=tier if isinstance(tier,str) else None
        if status=='ok' and not text: status='empty_transport' if outp==0 else 'empty_response'
        # A tier other than standard (null allowed), or a cache write, is not the registered tariff (adapter, section 5.4).
        if inp is not None and (tier not in ('standard',None) or (created or 0)>0): status='pricing_tier_mismatch'
    elif provider==OPENAI:
        items=body.get('output') if isinstance(body.get('output'),list) else []
        texts=[]; refused=False
        for item in items:
            if not isinstance(item,dict) or item.get('type')!='message': continue
            for part in item.get('content') if isinstance(item.get('content'),list) else []:
                if not isinstance(part,dict): continue
                if part.get('type')=='output_text' and isinstance(part.get('text'),str): texts.append(part['text'])
                elif part.get('type')=='refusal': refused=True
        text=''.join(texts)
        inp=_count(usage.get('input_tokens')); outp=_count(usage.get('output_tokens'))
        details=usage.get('output_tokens_details')
        reasoning=_count(details.get('reasoning_tokens')) if isinstance(details,dict) else None
        tier=body.get('service_tier')
        state=body.get('status')
        if state=='completed': status='ok'
        elif state=='incomplete':
            why=body.get('incomplete_details') if isinstance(body.get('incomplete_details'),dict) else {}
            reason=why.get('reason')
            status=TRUNCATED if reason=='max_output_tokens' else (REFUSED if reason=='content_filter' else 'incomplete_'+str(reason))
        elif isinstance(state,str): status=state
        else: status='malformed'
        if refused: status=REFUSED
        out['served_tier']=tier if isinstance(tier,str) else None
        if status=='ok' and not text: status='empty_transport' if outp==0 else 'empty_response'
        if inp is not None and tier!='flex': status='pricing_tier_mismatch'
    else:
        raise GateError('unknown provider')
    out.update({'status':status,'text':text,'input_tokens':inp if outp is not None else None,
                'output_tokens':outp if inp is not None else None,'reasoning_tokens':reasoning})
    return out


def reserve_amount(provider_cfg: dict[str,Any], provider: str, request: dict[str,Any]) -> tuple[int,float]:
    # A conservative text-only upper envelope, explicitly a budgeting assumption: a token covers at least one byte.
    n=len(canonical(request))+int(provider_cfg['input_margin_tokens'])
    if n>provider_cfg['max_reserved_input_tokens']: raise GateError('request exceeds frozen input envelope')
    out=math.ceil(output_cap(provider)*provider_cfg.get('output_reservation_multiplier',1))
    return n,(n*provider_cfg['input_usd_per_million_reserve']+out*provider_cfg['output_usd_per_million_reserve'])/1e6


def _kept_headers(provider: str, headers: Any) -> dict[str,str]:
    kept={}
    if headers is None: return kept
    try: items=list(headers.items())
    except Exception: return kept
    for name,value in items:
        low=str(name).lower()
        if low in KEPT_HEADERS[provider] or any(low.startswith(p) for p in KEPT_HEADER_PREFIXES[provider]):
            kept[low]=str(value)
    return kept


def _retry_after(kept: dict[str,str]) -> float | None:
    value=kept.get('retry-after')
    if value is None: return None
    try: seconds=float(value)
    except ValueError: return None
    return seconds if math.isfinite(seconds) and seconds>0 else None


def _error_fields(raw: bytes) -> dict[str,Any]:
    """Whether an HTTP error carried a JSON error body, and its type and code, for the settlement class and the report."""
    try: body=json.loads(raw)
    except Exception: return {'error_body_json':False,'error_type':None,'error_code':None}
    error=body.get('error') if isinstance(body,dict) else None
    if not isinstance(error,dict): return {'error_body_json':False,'error_type':None,'error_code':None}
    details=error.get('details') if isinstance(error.get('details'),dict) else {}
    code=error.get('code') if isinstance(error.get('code'),str) else details.get('error_code')
    return {'error_body_json':True,'error_type':error.get('type') if isinstance(error.get('type'),str) else None,
            'error_code':code if isinstance(code,str) else None}


class Client:
    """One journalled HTTP exchange per numbered attempt, for both engines, from several threads of one process.

    `call(slot, engine, role, system, user)` keeps the base's contract and returns the usable result of one request;
    reading what the result says is the collector's (section 6.1). The read time-out, the input margin, the rates and
    the transport schedule come from the configuration and are checked against the registered values; the output cap
    is the registered 16,000 tokens. The request identifier, `retry-after` and the rate-limit headers are kept.
    """
    def __init__(self, root: str | Path, config: dict[str,Any], *, live: bool=False,
                 expected_models: dict[str,str] | None=None, sleep: Any=None, on_event: Any=None):
        if not live: raise GateError('paid calls disabled; the lane passes live=True only from a step its gate allows')
        self.root=Path(root); self.config=config
        self.engines=config['engines']; self.providers=config['providers']
        self.expected=dict(expected_models or {})
        self.sleep=sleep if sleep is not None else time.sleep
        # Every retry and hold is reported as it happens, for the deviation record of section 11.4.
        self.on_event=on_event
        for engine,spec in self.engines.items():
            provider=spec['provider']; pcfg=self.providers[provider]
            if spec['model']!=IDS[provider] or pcfg['endpoint']!=ENDPOINTS[provider] or pcfg['key_env']!=ENVS[provider]:
                raise GateError('frozen provider identity mismatch: '+engine)
            if tuple(pcfg['transport_delays_seconds'])!=RETRY_DELAYS[TIERS[provider]] or pcfg['tier']!=TIERS[provider]:
                raise GateError('transport schedule or tier differs from the registered one: '+engine)
            payload(provider,'x','x',spec['request_fields'])
        # One process at a time owns the journal, admission and collection alike.
        self.lock_fd=hold_study_lock(self.root/'private'/'STUDY.lock')
        try:
            caps={spec['budget_key']:float(spec['cap_usd']) for spec in self.engines.values()}
            self.ledger=Ledger(self.root/'private'/'requests.sqlite',caps,sum(caps.values()))
            if any(entry['action']=='abandon_study' for entry in self.ledger.recovery()['entries']):
                raise GateError('The recovery log records that the study was abandoned; no further paid call is made')
        except BaseException:
            self.close(); raise
        self.opener=urllib.request.build_opener(NoRedirect())
    def close(self) -> None:
        """Release the study lock and clear the owner's mark. Safe to call more than once."""
        fd=getattr(self,'lock_fd',None)
        if fd is not None:
            self.lock_fd=None
            try: os.ftruncate(fd,0)
            except OSError: pass
            os.close(fd)
    def held(self, engine: str) -> list[str]:
        """The attempts that hold this engine's provider, with no reissue entry yet."""
        return self.ledger.holds_by_key().get(self.engines[engine]['budget_key'],[])
    def headroom(self, engine: str) -> float:
        return self.ledger.headroom(self.engines[engine]['budget_key'])
    def _event(self, record: dict[str,Any]) -> None:
        if self.on_event is not None:
            self.on_event({'utc':now(),**record})
    def _invalid_lines(self) -> str:
        invalid=self.ledger.recovery()['invalid_lines']
        return '; recovery log lines that are not valid entries and release nothing: '+', '.join(map(str,invalid)) if invalid else ''
    def _stop_date(self) -> str | None:
        path=self.root/'stop_at.json'
        if not path.exists(): return None
        value=load(path).get('date')
        return value if isinstance(value,str) else None
    def _check_stop(self, engine: str) -> None:
        when=self._stop_date()
        if when is not None and datetime.datetime.now(datetime.timezone.utc).date().isoformat()>=when:
            raise EngineStopped('a posted price change takes effect on '+when+'; '+engine+' makes no further call')
    def call(self, slot: str, engine: str, role: str, system: str, user: str) -> dict[str,Any]:
        """Return the usable result of one request, issuing numbered attempts under the recovery rules."""
        spec=self.engines[engine]; provider=spec['provider']; pcfg=self.providers[provider]; key=spec['budget_key']
        if datetime.datetime.now(datetime.timezone.utc).date().isoformat()>self.config['pricing_valid_through']:
            raise GateError('Price snapshot expired; re-freeze before new study, never silently change an active design')
        if '#attempt' in slot: raise GateError('A slot name cannot carry an attempt mark')
        if self.held(engine):
            raise ProviderHeld('provider '+provider+' is held; a recovery entry must release it first'+self._invalid_lines())
        self._check_stop(engine)
        reqbody=payload(provider,system,user,spec['request_fields'])
        delays=RETRY_DELAYS[TIERS[provider]]
        attempt=1; limit=len(delays)+1; wait=0.0; failed=0
        while attempt<=limit:
            aslot=attempt_slot(slot,attempt)
            ident=digest(canonical({'study':self.config['study_id'],'engine':engine,'role':role,'slot':aslot,'body':reqbody}))
            # The journalled row of this numbered attempt. Its request can differ from the current one only when the
            # body changed after it was journalled, as a logged repair can do.
            prior_slot=self.ledger.get_slot(key,role,aslot)
            row_id=ident if prior_slot is None else prior_slot['id']; changed=row_id!=ident
            different='The same paid slot has a different request body; preserve and reconcile, never issue twice (journalled request '+row_id+')'
            folder=self.root/'private'/'calls'/row_id; result_path=folder/'result.json'
            prior=self.ledger.get(row_id)
            if prior is None:
                # A new attempt. After a transport failure it waits its registered delay first.
                if wait: self.sleep(wait)
                self._check_stop(engine)
                result=self._issue(engine,spec,provider,pcfg,role,slot,aslot,attempt,ident,folder,reqbody)
            elif prior['state']=='reserved':
                # An earlier process stopped inside this attempt.
                if changed: raise GateError(different)
                result=self._resume(engine,spec,provider,pcfg,role,slot,aslot,attempt,ident,folder,reqbody,prior)
            else:
                if not result_path.exists(): raise GateError('journal/result mismatch; preserve and reconcile')
                result=load(result_path)
            wait=0.0
            if self.ledger.get(row_id)['state']=='held_released':
                # A recovery entry released this attempt earlier: one further attempt follows it.
                limit=max(limit,attempt+1); attempt+=1; continue
            klass=failure_class(result)
            if klass is None:
                # A usable result answers only the request that produced it.
                if changed: raise GateError(different)
                return result
            entry=self.ledger.released().get(row_id)
            label=result['status']+(' '+str(result['http_status']) if 'http_status' in result else '')
            if klass=='hold' or changed:
                # A hold, or a failed attempt made with another body: passed only by a recovery entry that names it.
                if entry is None:
                    if klass!='hold': raise GateError(different)
                    self._event({'kind':'hold','engine':engine,'slot':aslot,'request_id':row_id,'status':label})
                    raise ProviderHeld('provider failure: '+label+'; recorded hold; all records and reservations retained (request '+row_id+')'+self._invalid_lines())
                self._release(row_id,entry); limit=max(limit,attempt+1)
            elif attempt>=limit:
                if entry is None:
                    self._event({'kind':'transport_hold','engine':engine,'slot':aslot,'request_id':row_id,'status':label})
                    raise TransportHold('transport exhausted after '+str(attempt)+' attempts; recorded hold (request '+row_id+')'+self._invalid_lines(),row_id,key)
                self._release(row_id,entry); limit=attempt+1
            else:
                # The request's k-th transport failure: the k-th registered delay, or the recorded retry-after if longer.
                failed+=1; delay=delays[min(failed,len(delays))-1]
                after=result.get('retry_after_seconds')
                wait=float(max(delay,min(after,RETRY_AFTER_CEILING))) if isinstance(after,(int,float)) and not isinstance(after,bool) else float(delay)
                if self.ledger.get_slot(key,role,attempt_slot(slot,attempt+1)) is None:
                    self._event({'kind':'retry','engine':engine,'slot':aslot,'request_id':row_id,'status':label,
                                 'settlement':result.get('settlement_class'),'wait_seconds':wait})
            attempt+=1
        raise GateError('attempt sequence ended without a result; preserve and reconcile')
    def _release(self, ident: str, entry: dict[str,Any]) -> None:
        """Record a reissue entry on its row: the reconciled charge, never below what is already journalled."""
        row=self.ledger.get(ident)
        floor=max(row['reservation'],row['charge'] if row['charge'] is not None else 0.0)
        self.ledger.reconcile(ident,max(floor,float(entry['reconciled_charge_usd'])),'held_released',entry)
    def _headers(self, provider: str, secret: str) -> dict[str,str]:
        if provider==ANTHROPIC:
            return {'x-api-key':secret,'anthropic-version':ANTHROPIC_VERSION,'content-type':'application/json','user-agent':USER_AGENT}
        return {'Authorization':'Bearer '+secret,'Content-Type':'application/json','User-Agent':USER_AGENT}
    def _issue(self, engine: str, spec: dict[str,Any], provider: str, pcfg: dict[str,Any], role: str, slot: str,
               aslot: str, attempt: int, ident: str, folder: Path, reqbody: dict[str,Any]) -> dict[str,Any]:
        """Reserve, send and settle one numbered attempt."""
        secret=os.environ.get(pcfg['key_env'])
        if not secret: raise GateError(pcfg['key_env']+' is absent. Never put secrets in config or chat.')
        n,reservation=reserve_amount(pcfg,provider,reqbody)
        try:
            self.ledger.reserve(ident,spec['budget_key'],reservation,{'slot':aslot,'role':role,'engine':engine,
                                'request_sha256':digest(canonical(reqbody)),'base_slot':slot,'attempt':attempt})
        except GateError as exc:
            if 'exceed cap' in str(exc): raise CapReached('this lane\'s cap for '+provider+' would be passed') from exc
            raise
        save(folder/'request.json',reqbody,exclusive=True)
        started=now(); start=time.monotonic(); kept={}
        req=urllib.request.Request(pcfg['endpoint'],data=canonical(reqbody),headers=self._headers(provider,secret),method='POST')
        try:
            with self.opener.open(req,timeout=pcfg['read_timeout_seconds']) as response:
                kept=_kept_headers(provider,response.headers)
                raw=response.read(8_000_001)
            if len(raw)>8_000_000: raise GateError('response exceeds byte envelope')
            with (folder/'response.json').open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            body=json.loads(raw); result=normalise(provider,body)
        except urllib.error.HTTPError as exc:
            kept=_kept_headers(provider,exc.headers)
            try: raw=exc.read(65536)
            except Exception: raw=b''
            (folder/'http_error.txt').write_bytes(raw)
            result={'status':'http_failure','http_status':exc.code,'text':'','input_tokens':None,'output_tokens':None,
                    **_error_fields(raw)}
        except Exception as exc:
            # A stop by the operator or the system is not caught here: the row stays reserved and the next start
            # applies the interruption rule to it.
            result={'status':'transport_unknown','exception_type':type(exc).__name__,'text':'','input_tokens':None,'output_tokens':None}
        result['headers']=kept; result['retry_after_seconds']=_retry_after(kept)
        return self._finish(engine,spec,provider,pcfg,role,slot,aslot,attempt,ident,folder,n,reservation,result,started,time.monotonic()-start)
    def _resume(self, engine: str, spec: dict[str,Any], provider: str, pcfg: dict[str,Any], role: str, slot: str,
                aslot: str, attempt: int, ident: str, folder: Path, reqbody: dict[str,Any], prior: dict[str,Any]) -> dict[str,Any]:
        """Settle an attempt that an earlier process reserved and did not settle."""
        result_path=folder/'result.json'
        if result_path.exists():
            # The result was saved and the journal was not: settle the row from the saved record.
            result=load(result_path)
            if result.get('request_id')!=ident or 'journal_charge_usd' not in result:
                raise GateError('journal/result mismatch; preserve and reconcile')
            self.ledger.settle(ident,result['journal_charge_usd'],journal_state(result),
                               {'result_path':str(result_path.relative_to(self.root)),'result_sha256':digest(canonical(result)),
                                'settled_after_restart_utc':now()})
            return result
        n,reservation=reserve_amount(pcfg,provider,reqbody)
        if abs(reservation-prior['reservation'])>1e-12:
            raise GateError('journal reservation differs from the request envelope; preserve and reconcile')
        if (folder/'response.json').exists():
            # The response was saved: complete it exactly as a fresh response is completed.
            try: result=normalise(provider,json.loads((folder/'response.json').read_bytes()))
            except Exception as exc:
                result={'status':'transport_unknown','exception_type':type(exc).__name__,'text':'','input_tokens':None,'output_tokens':None}
            result['completed_from_saved_response']=True
        else:
            # No response was saved: charged in full, and the next numbered attempt follows.
            result={'status':'interrupted_charged_in_full','text':'','input_tokens':None,'output_tokens':None}
        result['headers']={}; result['retry_after_seconds']=None
        result['settled_after_restart_utc']=now()
        return self._finish(engine,spec,provider,pcfg,role,slot,aslot,attempt,ident,folder,n,reservation,result,prior['created'],None)
    def _model_ok(self, engine: str, provider: str, returned: Any) -> bool:
        """The receipt of section 3.1: after admission, exact equality with the admitted value; at admission, exact
        equality with the identifier on Anthropic, and on OpenAI the bare name or a dated name that extends it."""
        if not isinstance(returned,str): return False
        expected=self.expected.get(engine)
        if expected: return returned==expected
        if provider==ANTHROPIC: return returned==IDS[provider]
        return returned==IDS[provider] or (returned.startswith(IDS[provider]+'-') and returned[len(IDS[provider])+1:].replace('-','').isdigit())
    def _finish(self, engine: str, spec: dict[str,Any], provider: str, pcfg: dict[str,Any], role: str, slot: str,
                aslot: str, attempt: int, ident: str, folder: Path, n: int, reservation: float, result: dict[str,Any],
                started: str, elapsed: float | None) -> dict[str,Any]:
        """Check, charge, save and settle one attempt. A fresh and a recovered response take the same path."""
        result_path=folder/'result.json'
        result.update({'engine':engine,'provider':provider,'requested_model':spec['model'],'slot':aslot,'role':role,
                       'request_id':ident,'started_utc':started,'elapsed_seconds':elapsed,'reservation_usd':reservation,
                       'input_reservation':n,'base_slot':slot,'attempt':attempt})
        inp,out=result.get('input_tokens'),result.get('output_tokens')
        charge=None
        if inp is not None and out is not None:
            # Charged at the registered rates (section 3.1); a cache read, were one to occur, at the full input rate.
            charge=(inp*pcfg['input_usd_per_million']+out*pcfg['output_usd_per_million'])/1e6
            result['ledger_charge_usd']=charge
            if result['status'] not in TRANSPORT_STATUSES and (charge>reservation+1e-9 or inp>n or out>math.ceil(output_cap(provider)*pcfg.get('output_reservation_multiplier',1))):
                result['status']='reservation_exceeded'
        elif result['status'] not in NO_RESPONSE: result['status']='usage_missing'
        if result['status'] not in NO_RESPONSE+HOLD_STATUSES and inp is not None and not self._model_ok(engine,provider,result.get('returned_model')):
            result['status']=MODEL_CHANGED
        klass=failure_class(result)
        if klass=='transport':
            settle=settlement_class(result); result['settlement_class']=settle
            charge=0.0 if settle=='uncharged' else (charge if settle=='usage' and charge is not None else reservation)
        elif klass=='hold':
            # An unconfirmed tariff keeps its reservation counting until reconciliation; every other hold is charged at
            # no less than its reservation.
            if result['status']=='pricing_tier_mismatch':
                if charge is not None: result['unconfirmed_registered_tariff_estimate_usd']=result.pop('ledger_charge_usd')
                result['billing_reconciliation_required']=True; charge=None
            elif charge is not None: charge=max(charge,reservation)
        elif result['status']==MODEL_CHANGED:
            charge=max(charge,reservation) if charge is not None else reservation
        if klass is not None: result['failure_class']=klass
        result['journal_charge_usd']=charge
        save(result_path,result,exclusive=True)
        self.ledger.settle(ident,charge,journal_state(result),{'result_path':str(result_path.relative_to(self.root)),
                                                             'result_sha256':digest(canonical(result))})
        return result
