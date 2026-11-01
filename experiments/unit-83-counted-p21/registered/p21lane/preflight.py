"""The pre-flight read-back the lab standard requires (docs/EDEN_LAB_STANDARD.md; DESIGN-FREEZE.md, sections 12,
package I, and 14.5, item 5).

Package H. Run only by the author's own command, in a terminal logged into no AI, after both registrations read
REGISTERED and before the run-start record: for each provider, the key in its environment variable is valid and the
model string is served (read from the provider's free model-retrieval endpoint; no paid call is made outside the
journal, so the flex tier itself is confirmed by admission's journalled transport checks); then the balance the author
reads by hand on the provider's own billing page, which must cover the larger of this lane's planned spend with its
margin and its registered cap; the provider-side spend limit set (and OpenAI's automatic reload off); and the two price
pages read on the day showing the prices of section 3.1. A failed read-back stops the run before its first call. Each
result is written with its UTC time and counts for six hours. No key, account identifier or retrieved page is written.
Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012); before then the only mechanical
check is a parse of this file's text.
"""
import datetime
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from p21base import providers
from p21base.common import GateError, load, now, save

VALID_HOURS = 6
USER_AGENT = 'p21-lane-preflight'


def _model_readback(provider, key_env):
    """(key valid, model served, detail) from the provider's model-retrieval endpoint; the key is read here only."""
    secret = os.environ.get(key_env)
    if not secret:
        return False, False, key_env + ' is not set'
    model = providers.IDS[provider]
    url = providers.MODEL_ENDPOINTS[provider] + model
    if provider == providers.ANTHROPIC:
        headers = {'x-api-key': secret, 'anthropic-version': providers.ANTHROPIC_VERSION, 'user-agent': USER_AGENT}
    else:
        headers = {'Authorization': 'Bearer ' + secret, 'User-Agent': USER_AGENT}
    request = urllib.request.Request(url, headers=headers, method='GET')
    # A redirect is refused, as the provider client refuses it, so that the key goes to the provider's own address
    # alone; a redirected check reads as no reply and the pre-flight fails.
    opener = urllib.request.build_opener(providers.NoRedirect())
    try:
        with opener.open(request, timeout=60) as response:
            body = json.loads(response.read(1_000_001))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, False, 'the key was refused (HTTP %d)' % exc.code
        if exc.code == 404:
            return True, False, 'the model string is not served (HTTP 404)'
        return False, False, 'HTTP %d' % exc.code
    except Exception as exc:
        return False, False, 'no reply: ' + type(exc).__name__
    served = isinstance(body, dict) and body.get('id') == model
    return True, served, 'the endpoint returned ' + str(body.get('id') if isinstance(body, dict) else None)


def _yes(ask, question):
    return str(ask(question + ' Type yes or no: ')).strip().lower() == 'yes'


def _number(ask, question):
    text = str(ask(question)).strip().replace(',', '')
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def registry_reads(registration_id, config):
    """Both registrations (and the suite's, where the configuration names it) read from the registry's public interface
    without a key, each with its problems: a read-back counts only when it is recorded after the registrations read
    REGISTERED (docs/EDEN_LAB_STANDARD.md, the pre-flight). The identifier is the one the bench prints; the attestation
    checks it again against the record it names."""
    from p21base import registration_gate as gate
    out = {'read_utc': now(), 'records': {}, 'problems': []}
    names = [('unit', registration_id), ('theory', config['theory_registration'])]
    if config.get('suite_registration'):
        names.append(('suite', config['suite_registration']))
    for label, identifier in names:
        try:
            record = gate.registration_record(identifier)
        except Exception as exc:
            out['problems'].append('%s %s could not be read: %s' % (label, identifier, type(exc).__name__))
            continue
        out['records'][label] = {'identifier': record['identifier'], 'date_registered': record['date_registered'],
                                 'response_sha256': record['response_sha256']}
        out['problems'] += ['%s %s: %s' % (label, identifier, p) for p in gate.registration_problems(record)]
    out['registered'] = not out['problems']
    return out


def readback(root, config, *, ask=input, show=print, registry=None):
    """The pre-flight. Returns the record and writes it under preflight/; a failed read-back says which item failed.
    `registry` is registry_reads(...), taken first: where the registrations do not read REGISTERED no key is read."""
    root = Path(root)
    results = {'utc': now(), 'providers': {}, 'passed': False, 'registry': registry}
    if registry is None or not registry.get('registered'):
        results['refused'] = 'the registrations do not read REGISTERED on the registry: ' + '; '.join((registry or {}).get('problems') or ['not read'])
        stamp = results['utc'].replace(':', '').replace('+0000', 'Z')
        save(root / 'preflight' / ('readback-%s.json' % stamp), results, exclusive=True)
        show('PRE-FLIGHT FAILED: ' + results['refused'] + '; no key was read')
        return results
    passed = True
    for engine_key in config['engine_order']:
        spec = config['engines'][engine_key]
        provider = spec['provider']
        pcfg = config['providers'][provider]
        show('')
        show('%s (%s, %s tier):' % (config['display_names'][engine_key], provider, pcfg['tier']))
        valid, served, detail = _model_readback(provider, pcfg['key_env'])
        show('  key valid: %s; model %s served: %s (%s)' % ('yes' if valid else 'no', spec['model'], 'yes' if served else 'no', detail))
        need = max(spec['planned_spend_with_margin_usd'], spec['cap_usd'])
        show('  read the balance on %s' % pcfg['billing_page'])
        balance = _number(ask, '  The balance you read there, in US dollars: ')
        funded = balance is not None and balance >= need
        show('  funded for at least USD %.2f: %s' % (need, 'yes' if funded else 'no'))
        limit = _yes(ask, '  Is the provider-side spend limit set (this lane\'s cap, or the run\'s total where the lanes share it)%s?'
                     % (' and is automatic reload off' if provider == providers.OPENAI else ''))
        prices = config['prices_usd_per_million'][provider]
        show('  registered prices: input %s and output %s US dollars a million tokens (%s); page %s'
             % (prices['input'], prices['output'], prices['route'], pcfg['price_page']))
        price_ok = _yes(ask, '  Does the price page, read today, show exactly these prices?')
        row = {'key_valid': valid, 'model_served': served, 'detail': detail, 'balance_read_usd': balance,
               'balance_needed_usd': need, 'funded': funded, 'billing_page': pcfg['billing_page'],
               'spend_limit_set': limit, 'prices_as_registered': price_ok, 'price_page': pcfg['price_page'],
               'read_utc': now()}
        row['passed'] = all((valid, served, funded, limit, price_ok))
        results['providers'][provider] = row
        passed = passed and row['passed']
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    results['pricing_valid'] = today <= config['pricing_valid_through']
    results['passed'] = passed and results['pricing_valid']
    stamp = results['utc'].replace(':', '').replace('+0000', 'Z')
    save(root / 'preflight' / ('readback-%s.json' % stamp), results, exclusive=True)
    show('')
    show('PRE-FLIGHT ' + ('PASSED' if results['passed'] else 'FAILED: the run does not start'))
    return results


def latest(root):
    """The newest pre-flight record, or None."""
    folder = Path(root) / 'preflight'
    if not folder.exists():
        return None
    paths = sorted(folder.glob('readback-*.json'))
    return load(paths[-1]) if paths else None


def current(root, at=None):
    """Whether a pre-flight passed within the six hours before `at` (now by default), recorded after both
    registrations read REGISTERED on the registry (the lab standard)."""
    record = latest(root)
    if record is None or not record.get('passed') or not (record.get('registry') or {}).get('registered'):
        return False
    when = datetime.datetime.fromisoformat(record['utc'].replace('Z', '+00:00'))
    moment = at or datetime.datetime.now(datetime.timezone.utc)
    return datetime.timedelta(0) <= moment - when <= datetime.timedelta(hours=VALID_HOURS)
