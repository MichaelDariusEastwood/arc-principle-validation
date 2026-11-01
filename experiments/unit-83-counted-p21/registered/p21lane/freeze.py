"""The runtime freeze and its deposit (41-D-COUNTED, sections 4.9 and 11.3, step 6).

Package H of the build list (DESIGN-FREEZE.md, section 12). `write` records the hashes of every executed file, the
repair log, the admission and calibration receipts, the tier, the number of pairs in flight on each engine, the bank
commitment and the environment. `deposit` uploads the freeze record and the bank commitment to the unit's OSF
component with a token for that component, reads each back and compares its hash, before the first study call; if it
cannot, the lane stops, the files are uploaded by hand, and the same command finds them by hash and resumes. Nothing
here is executed before both registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is
a parse of this file's text.
"""
import hashlib
import json
import os
import platform
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from p21base import registration_gate as gate
from p21base.common import GateError, file_sha, load, now, save

FREEZE = 'freeze.json'
DEPOSIT = 'freeze_deposit.json'
TOKEN_ENV = 'OSF_COMPONENT_TOKEN'
USER_AGENT = 'p21-lane-deposit'
RECORDS = ('registration_approval.json', 'bank_commitment.json', 'admission/admission_receipt.json',
           'admission/calibration_receipt.json', 'admission/in_flight.json', 'environment.json')


def write(root, config):
    """Step 6: the runtime freeze record. It is written once; after it, nothing changes."""
    root = Path(root)
    if (root / FREEZE).exists():
        raise GateError('Do not overwrite a freeze')
    stage = gate.stage_root()
    manifest = json.loads((stage / gate.MANIFEST).read_text('utf-8'))
    environment = {'python': sys.version, 'implementation': platform.python_implementation(),
                   'platform': platform.platform(), 'utc': now()}
    # Rewritten while no freeze exists, so that a freeze stopped part-way can be written again; its hash is frozen below.
    save(root / 'environment.json', environment)
    calibration = load(root / 'admission' / 'calibration_receipt.json')
    in_flight = load(root / 'admission' / 'in_flight.json')
    admission = load(root / 'admission' / 'admission_receipt.json')
    files = {'stage/' + rel: file_sha(stage / rel) for rel in sorted(manifest['files'])}
    files['stage/' + gate.MANIFEST] = file_sha(stage / gate.MANIFEST)
    for rel in RECORDS:
        files[rel] = file_sha(root / rel)
    if (root / gate.REPAIR_LOG).exists():
        files[gate.REPAIR_LOG] = file_sha(root / gate.REPAIR_LOG)
    record = {'created_utc': now(), 'status': 'REGISTERED_RECIPE_RUNTIME_ADMISSION_FROZEN', 'study_id': config['study_id'],
              'tier': calibration['tier'], 'admitted': admission['admitted'],
              'generating_admitted': admission['generating_admitted'], 'expected_models': admission['expected_models'],
              'pairs_in_flight': {e: row['pairs_in_flight'] for e, row in in_flight['engines'].items()},
              'files': files}
    save(root / FREEZE, record, exclusive=True)
    os.chmod(root / FREEZE, 0o444)
    return record


def look_repaired(root, record):
    """For each analysis file a look-failure repair changed after the freeze (section 7.4), the hash on which its chain
    of look repairs ends, the chain starting from the frozen hash. The gate has already checked each entry against its
    recorded look failure (registration_gate.check_look_repairs)."""
    entries = gate.read_look_repair_log(Path(root) / gate.LOOK_REPAIR_LOG)
    if entries:
        gate.check_look_repairs(entries, Path(root))
    last = {}
    for entry in entries:
        rel = 'stage/' + entry['file']
        if rel not in record['files'] or entry['replaced_sha256'] != last.get(rel, record['files'][rel]):
            raise GateError('Look repair log line %d does not continue the frozen chain of %s' % (entry['line'], entry['file']))
        last[rel] = entry['new_sha256']
    return last


def check_frozen(root):
    """Every frozen file holds its frozen bytes, or for one of the four analysis files the bytes on which its logged
    look repairs end (section 7.4), and the interpreter is the frozen one (the base's check_frozen)."""
    root = Path(root)
    record = load(root / FREEZE)
    stage = gate.stage_root()
    repaired = look_repaired(root, record)
    for rel, expected in record['files'].items():
        path = stage / rel[len('stage/'):] if rel.startswith('stage/') else root / rel
        actual = file_sha(path) if path.is_file() else None
        if actual is None or (actual != expected and actual != repaired.get(rel)):
            raise GateError('Frozen bytes changed: ' + rel)
    if (root / gate.REPAIR_LOG).exists() and gate.REPAIR_LOG not in record['files']:
        if gate.read_repair_log(root / gate.REPAIR_LOG):
            raise GateError('A repair log exists that the runtime freeze did not record')
    environment = load(root / 'environment.json')
    if environment['python'].split()[0] != sys.version.split()[0]:
        raise GateError('Python version differs from the freeze')
    return record


class _TokenStaysWithOSF(urllib.request.HTTPRedirectHandler):
    """A redirect is followed, but the component token goes only to the registry's own hosts over https: a redirect to
    any other address (a storage service's signed address, say) is followed without it, so the token never leaves OSF
    (the provider client refuses redirects for the same reason)."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        parts = urllib.parse.urlsplit(newurl)
        host = (parts.hostname or '').lower()
        if new is not None and not (parts.scheme == 'https' and (host == 'osf.io' or host.endswith('.osf.io'))):
            for store in (new.headers, new.unredirected_hdrs):
                for name in [n for n in store if n.lower() == 'authorization']:
                    del store[name]
        return new


_OPENER = urllib.request.build_opener(_TokenStaysWithOSF())


def _request(url, token, data=None, method='GET'):
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': USER_AGENT}
    if data is not None:
        headers['Content-Type'] = 'application/octet-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with _OPENER.open(request, timeout=120) as response:
        return response.read(50_000_001)


def _held_hashes(node, token):
    """The SHA-256 of each file in the component's storage, with its download address, read with the component token."""
    found = {}
    queue = ['https://api.osf.io/v2/nodes/%s/files/osfstorage/' % node]
    while queue and len(found) < 5000:
        body = json.loads(_request(queue.pop(0), token))
        for item in body.get('data') or []:
            attributes = item.get('attributes') or {}
            if attributes.get('kind') == 'folder':
                related = (((item.get('relationships') or {}).get('files') or {}).get('links') or {}).get('related') or {}
                href = related.get('href') if isinstance(related, dict) else related
                if isinstance(href, str):
                    queue.append(href)
                continue
            sha = ((attributes.get('extra') or {}).get('hashes') or {}).get('sha256')
            link = (item.get('links') or {}).get('download')
            if isinstance(sha, str):
                found[sha.lower()] = link
        following = (body.get('links') or {}).get('next')
        if isinstance(following, str):
            queue.append(following)
    return found


def deposit(root, config):
    """Upload the freeze record and the bank commitment to the unit's component and read each back by hash; or, where
    they were uploaded by hand, find them by hash. Writes freeze_deposit.json."""
    root = Path(root)
    if (root / DEPOSIT).exists():
        return load(root / DEPOSIT)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise GateError(TOKEN_ENV + ' is absent: the component token is needed to deposit the freeze before the first study call')
    node = gate.read_attestation(root)['registered_from']
    targets = {}
    for rel in (FREEZE, 'bank_commitment.json'):
        data = (root / rel).read_bytes()
        targets[rel] = (data, hashlib.sha256(data).hexdigest())
    held = _held_hashes(node, token)
    addresses = {}
    for rel, (data, sha) in targets.items():
        if sha not in held:
            name = '%s-%s%s' % (Path(rel).stem, sha[:12], Path(rel).suffix)
            url = 'https://files.osf.io/v1/resources/%s/providers/osfstorage/?kind=file&name=%s' % (node, urllib.parse.quote(name))
            body = json.loads(_request(url, token, data=data, method='PUT'))
            link = ((body.get('data') or {}).get('links') or {}).get('download')
            back = _request(link, token) if isinstance(link, str) else b''
            if hashlib.sha256(back).hexdigest() != sha:
                raise GateError('the deposited copy of %s does not read back with its hash; upload it by hand and run the same command' % rel)
            addresses[rel] = link
        else:
            addresses[rel] = held[sha]
    record = {'freeze_sha256': targets[FREEZE][1], 'bank_commitment_sha256': targets['bank_commitment.json'][1],
              'component': node, 'osf_file_urls': addresses, 'uploaded_utc': now()}
    save(root / DEPOSIT, record, exclusive=True)
    return record


def authorisation(root):
    """Before the first study call: the deposit record names this freeze."""
    root = Path(root)
    if not (root / DEPOSIT).exists():
        raise GateError('The runtime freeze deposit record is absent; deposit the freeze on OSF before the first study call')
    record = load(root / DEPOSIT)
    if record.get('freeze_sha256') != file_sha(root / FREEZE):
        raise GateError('The deposit record refers to another runtime freeze')
    if record.get('bank_commitment_sha256') != file_sha(root / 'bank_commitment.json'):
        raise GateError('The deposit record refers to another bank commitment')
    return record
