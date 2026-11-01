#!/usr/bin/env python3
"""The lab bench of an Eden-Lab unit: a generic engine made by the bench factory
(tools/research/eden_lab_bench_factory.py, version 1.2) from Unit 06's bench (version 1.1,
3 October 2026). Everything unit-specific is data: kit/STEPS.json (the steps, their exact commands and
their pass rules), kit/PROVIDERS.json (the pre-flight) and kit/CURRENT-HASHES.json (the registered hashes).

What it is: a plain script for the bench terminal (the terminal that is not logged into any AI).
It knows which step the run is at, prints the exact commands of kit/RUN-SHEET.md for that step,
checks each step's pass rule by reading the files the step leaves, the hashes and the lab log, and
keeps the lab log. It holds the run at its pre-flight: it refuses to mark the run started until every
provider's key, model string and balance has been read back and each result recorded in the lab log,
within the per-test cap, and until the folder as it stands carries the author's entry recording it as
reviewed. It records nothing (no step, no prepare, no pre-flight result) while the folder differs from
FOLDER-MANIFEST.sha256, or while kit/STEPS.json would let a call come before the pre-flight, wherever its
steps stand (it reads each step's commands again, so an edited flag cannot hide a call), and a read-back
counts only on the folder as sealed. It decides nothing about the study, and it never sees a review: the
seal is the author's entry.

What it never does: it never runs a line of the unit's code, never calls a network address, never
starts another program and never reads a key or a token. Its own code reads no environment variable;
Python's argparse, which it uses, reads only the locale names LANGUAGE, LC_ALL, LC_MESSAGES and LANG
(through gettext), never a key. Every command that runs code or calls out is printed for you to run
yourself, in your one shell, exactly as the run sheet gives it. The bench itself only copies files,
computes SHA-256 and reads the files the steps leave; it never reads a file named like a key, a token,
a credential or a sealed run (SEALED, KEY, TOKEN, a dotenv file, .netrc), nor the token file you name.
The run sheet stays the
authority: if this script and kit/RUN-SHEET.md ever differ, the run sheet wins, and the difference
goes in unit/run/DEFECT-LOG.md.

Standard library only. Python 3.9 or later.

  python3 bench/bench.py init                       set up lab/bench.json
  python3 bench/bench.py set KEY VALUE              record one of your blanks (see `blanks`)
  python3 bench/bench.py blanks                     list the blanks and which are still empty
  python3 bench/bench.py status                     where the run stands, and what is next
  python3 bench/bench.py show [STEP]                the step's instructions and exact commands
  python3 bench/bench.py prepare                    the steps that only copy and hash (allowed before registration)
  python3 bench/bench.py preflight                  every provider's read-backs, and what is recorded
  python3 bench/bench.py preflight NAME CHECK pass|fail [--served S] [--amount N --currency C] [--note T]
          [--source api | --source console --page PAGE --read-utc UTC [--screenshot-sha256 HEX]]
                                                    record one read-back (CHECK is key, model or balance; a key pass
                                                    carries its status line in --note; a balance pass says how it was read)
  python3 bench/bench.py done STEP [--exit N ...]   check the step's pass rule and record it
  python3 bench/bench.py note TEXT                  add one line of your own to the lab log
  python3 bench/bench.py selfcheck                  the run-sheet steps' commands stand in the run sheet; the files are the
                                                    registered bytes; the folder is its manifest; no TODO is left
  python3 bench/bench.py manifest                   compare FOLDER-MANIFEST.sha256 with the folder, and show the seal
  python3 bench/bench.py manifest --write --reviewed-by NAME --review WHERE
                                                    seal the folder after a review: your entry of who reviewed it and
                                                    where the review lives (the bench does not see the review)
"""
import argparse, datetime as dt, fnmatch, hashlib, json, math, os, re, shlex, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)                      # the lab folder holds bench/, kit/, deposit/, stage/, lab/
KIT = os.path.join(LAB, 'kit')
LABDIR = os.path.join(LAB, 'lab')
CFG = os.path.join(LABDIR, 'bench.json')
LOG = os.path.join(LABDIR, 'LAB-LOG.md')
STEPS_FILE = os.path.join(KIT, 'STEPS.json')
PROVIDERS_FILE = os.path.join(KIT, 'PROVIDERS.json')
HASHES_FILE = os.path.join(KIT, 'CURRENT-HASHES.json')
RUN_SHEET = os.path.join(KIT, 'RUN-SHEET.md')
FOLDER_MANIFEST = os.path.join(LAB, 'FOLDER-MANIFEST.sha256')
COVERED = ('START-HERE.md', 'bench', 'deposit', 'kit', 'stage')
ROOTS = ('unit', 'lab', 'deposit', 'stage', 'kit', 'folder')
CHECKS = ('key', 'model', 'balance')
UTC_FORM = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
PLACEHOLDER = re.compile(r'<([A-Z][A-Z0-9 _-]{2,})>')
# The name test, as in the factory (its self-test keeps the two copies equal).
SEALED_WORDS = {'SEALED', 'KEY', 'KEYS', 'TOKEN', 'TOKENS', 'SECRET', 'SECRETS', 'UNBLIND', 'CREDENTIAL', 'CREDENTIALS',
                'PASSWORD', 'PASSWORDS', 'PASSWD'}
JOINED_HEADS = {'API', 'AUTH', 'ACCESS', 'BEARER', 'REFRESH', 'SESSION', 'PRIVATE', 'SECRET', 'SEALED', 'CLIENT', 'MASTER',
                'SIGNING', 'OAUTH', 'SSH', 'ID'}
JOINED_CORES = ('KEYS', 'KEY', 'TOKENS', 'TOKEN', 'SECRETS', 'SECRET')
JOINED_TAILS = {'FILE', 'FILES', 'CHAIN', 'RING', 'STORE', 'PAIR'}
SECRET_NAME_RE = re.compile(r'^(\.env(\..+)?|.+\.env|[._]netrc|\.pgpass|\.git-credentials|\.npmrc|\.pypirc|\.htpasswd|'
                            r'id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|.+\.(pem|p12|pfx|jks|ppk|kdbx))$', re.I)
KNOWN_CHECKS = {'todo', 'registered', 'window_open', 'blank_set', 'step_done', 'preflight_complete', 'log_contains',
                'exit_codes', 'file_exists', 'dir_exists', 'file_absent', 'file_nonempty', 'file_contains',
                'last_line_equals', 'json_parses', 'json_field_equals', 'sha256_registered', 'record_sha256', 'manifest_ok'}
# What runs the unit's code or calls the network, as the factory reads it (the self-test keeps the two copies equal): the bench
# reads each step's commands again with these, so a flag edited by hand cannot hide a call before the run-start record.
RUNS_UNIT_CODE_RE = re.compile(r'(\$PY\b|\$\{PY\}|\.venv[^/\s]*/bin/python|\bpython3?\s+(?:-[A-Za-z]+\s+)*(?:code|attachments)/)')
NETWORK_RE = re.compile(r'\b(curl|wget|git\s+(?:clone|fetch|pull|push))\b|\bpip3?\s+install\b|-m\s+pip\s+install\b')
KEY_VAR_RE = re.compile(r'(^|_)(KEY|TOKEN)(_|$)')
# The per-test cap of decision E14 (SD-081), held here as well as in kit/PROVIDERS.json, so a kit edited above it is refused.
E14 = {'currency': 'GBP', 'anthropic': 200.0, 'others': 100.0}
POOL_WORDS = {'anthropic': 'the Anthropic API', 'others': 'the other providers together'}
# How long a read-back counts for at the run-start record: a balance is shared by every key of its account and can move.
PREFLIGHT_MAX_AGE_HOURS = 6
WIDTH = 100
_CACHE = {}
ACTOR = ['']


def now():
    return dt.datetime.now(dt.timezone.utc)


def stamp():
    return now().strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_utc(v):
    return dt.datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def name_words(name):
    """The words of a file name: it parts at every character that is not a letter (so letters joined to digits part too),
    between a small letter and a capital (SealedToken) and before the last capital of a run of capitals (APIKey)."""
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', os.path.basename(str(name)))
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return [w.upper() for w in re.findall(r'[A-Za-z]+', spaced)]


def sealed_like(name):
    """A name that may hold a key, a token, a credential or a sealed run: the bench never reads one. It is one by its
    whole name (a dotenv file, .netrc, an SSH key), by a word (KEY, TOKEN, SEALED, CREDENTIALS, PASSWORD...) or by a
    joined word (apikey, authtoken, keyfile)."""
    base = os.path.basename(str(name))
    if SECRET_NAME_RE.match(base):
        return True
    for w in name_words(base):
        if w in SEALED_WORDS:
            return True
        for core in JOINED_CORES:
            if (w.endswith(core) and w[:-len(core)] in JOINED_HEADS) or (w.startswith(core) and w[len(core):] in JOINED_TAILS):
                return True
    return False


def looks_like_a_secret(value):
    """A key or token by its shape alone: a known prefix, or one long run of letters and digits. No blank needs either."""
    for tok in str(value).split():
        tok = tok.strip('"\'(),.;:')
        if re.match(r'^(sk-|AIza|ghp_|gho_|github_pat_|xox[abp]-|ya29\.)', tok) and len(tok) >= 12:
            return True
        if len(tok) >= 32 and re.fullmatch(r'[A-Za-z0-9_\-]+', tok) and re.search(r'[A-Za-z]', tok) and re.search(r'\d', tok):
            return True
    return False


def placeholder_id(text):
    return re.sub(r'[^A-Z0-9]+', '_', text.upper()).strip('_')


def is_todo(v):
    return isinstance(v, str) and v.strip().upper().startswith('TODO')


def plural_items(n):
    return f'{n} item' + ('' if n == 1 else 's')


def wrap(text, indent='  ', width=WIDTH):
    words, lines, cur = str(text).split(), [], indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += w + ' '
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def read_json(path, what):
    if not os.path.exists(path):
        sys.exit(f'REFUSED: {what} is not there ({os.path.relpath(path, LAB)}). Nothing is checked without it.')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def steps_doc():
    if 'steps' not in _CACHE:
        d = read_json(STEPS_FILE, 'kit/STEPS.json, the table of steps')
        for s in d.get('steps', []):
            s['commands'] = [c if isinstance(c, dict) else {'run': str(c)} for c in s.get('commands', [])]
            rule = s.get('pass_rule', [])
            s['pass_rule'] = [rule] if isinstance(rule, dict) else list(rule)
            s.setdefault('text', [])
            s.setdefault('notes', [])
        _CACHE['steps'] = d
    return _CACHE['steps']


def providers_doc():
    if 'providers' not in _CACHE:
        _CACHE['providers'] = read_json(PROVIDERS_FILE, 'kit/PROVIDERS.json, the pre-flight table')
    return _CACHE['providers']


def providers():
    return [p for p in providers_doc().get('providers', []) if isinstance(p, dict)]


def current_hashes():
    if 'hashes' not in _CACHE:
        _CACHE['hashes'] = read_json(HASHES_FILE, 'kit/CURRENT-HASHES.json, the table of registered hashes every comparison needs')
    return _CACHE['hashes']


def steps():
    return steps_doc().get('steps', [])


def step_ids():
    return [s['id'] for s in steps()]


def step_by_id(sid):
    return next((s for s in steps() if s['id'] == sid), None)


def run_start_index():
    return next((i for i, s in enumerate(steps()) if s.get('run_start')), None)


def blank_table():
    return steps_doc().get('blanks', {})


def is_private(key):
    k = key.upper()
    return bool(blank_table().get(key, {}).get('private')) or 'KEYCHAIN' in k or 'TOKEN' in k


def window():
    w = steps_doc().get('window_close_utc')
    return parse_utc(w) if isinstance(w, str) and UTC_FORM.match(w) else None


def hours_left():
    w = window()
    return None if w is None else round((w - now()).total_seconds() / 3600, 1)


def load(check_folders=True):
    if not os.path.exists(CFG):
        sys.exit('no lab/bench.json yet: run  python3 bench/bench.py init')
    with open(CFG, encoding='utf-8') as f:
        cfg = json.load(f)
    if check_folders:
        for k, v in cfg.get('blanks', {}).items():
            why = bad_folder_blank(k, v) if v and blank_table().get(k, {}).get('path') else ''
            if why:
                sys.exit(f'REFUSED: the blank {k} in lab/bench.json points where the bench never goes: {why}. '
                         f'Set it again with  python3 bench/bench.py set {k} PATH')
    return cfg


def config_or_defaults():
    if os.path.exists(CFG):
        return load()
    return {'blanks': {k: m.get('default', '') for k, m in blank_table().items()}, 'done': {}}


def save(cfg):
    os.makedirs(LABDIR, exist_ok=True)
    tmp = CFG + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, CFG)


def log(line):
    os.makedirs(LABDIR, exist_ok=True)
    actor = ACTOR[0]
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f'- {stamp()} ' + (f'[{actor}] ' if actor else '') + f'{line}\n')


def read_log():
    if not os.path.exists(LOG):
        return ''
    with open(LOG, encoding='utf-8', errors='replace') as f:
        return f.read()


def resolve(p):
    """A path kept relative in lab/bench.json is relative to the lab folder, so the folder can be moved whole."""
    return p if os.path.isabs(p) else os.path.join(LAB, p)


def inside(base, p):
    base, p = os.path.abspath(base), os.path.abspath(p)
    return p == base or p.startswith(base.rstrip(os.sep) + os.sep)


def unit_like(folder):
    """A folder laid out as a lab unit: it holds bench/bench.py, or START-HERE.md beside kit/."""
    return os.path.isfile(os.path.join(folder, 'bench', 'bench.py')) or (
        os.path.isfile(os.path.join(folder, 'START-HERE.md')) and os.path.isdir(os.path.join(folder, 'kit')))


def bad_folder_blank(key, value):
    """Why a folder blank may not point where it points, or ''. The unit folder, which prepare writes
    into, never lies in another lab unit's folder, nor is it this lab folder, a folder holding it, or its
    kit, deposit, stage, bench or lab; the deposit and the stage, which are only read, never lie in
    another unit's unit/ or lab/ folder (its sealed run). The path is judged as written and again where
    its links lead, so a link cannot carry a blank into another unit's sealed run. Another unit's folder
    is found from the top down, so no file inside its sealed run is opened. Letter case is ignored, as
    the Mac's disk ignores it."""
    raw = os.path.abspath(resolve(str(value)))
    real = os.path.realpath(raw)
    why = folder_reason(key, raw, os.path.abspath(LAB))
    if not why and real != raw:
        why = folder_reason(key, real, os.path.realpath(LAB))
    return why


def folder_reason(key, p, lab):
    """bad_folder_blank's rule for one spelling of the path, p, against the same spelling of this lab folder, lab."""
    if key == 'UNIT':
        if inside(p.casefold(), lab.casefold()):
            return 'the unit folder cannot be this lab folder or a folder that holds it'
        for top in ('kit', 'deposit', 'stage', 'bench', 'lab'):
            if inside(os.path.join(lab, top).casefold(), p.casefold()):
                return f'the unit folder cannot lie in this lab folder\'s {top}/'
    parts = p.split(os.sep)
    cur = os.sep
    for i in range(1, len(parts)):
        cur = os.path.join(cur, parts[i])
        if cur.casefold() == lab.casefold():
            return ''
        if unit_like(cur):
            if key == 'UNIT':
                return f'{cur} is another lab unit\'s folder, and prepare never writes into it'
            nxt = parts[i + 1] if i + 1 < len(parts) else ''
            if nxt.casefold() in ('unit', 'lab'):
                return f'{os.path.join(cur, nxt)} is another lab unit\'s {nxt.lower()}/ folder, and the bench never reads it'
            return ''
    return ''


def root_dir(root, cfg):
    b = cfg.get('blanks', {})
    if root in ('unit', 'deposit', 'stage'):
        return resolve(b.get(root.upper()) or root)
    return {'lab': LABDIR, 'kit': KIT}.get(root, LAB)


def where(spec, cfg):
    """A check's path: unit:run/plan.json, deposit:NAME, stage:NAME, lab:NAME, kit:NAME or folder:NAME; no prefix means the unit folder."""
    root, sep, rest = str(spec).partition(':')
    if not sep or root not in ROOTS:
        root, rest = 'unit', str(spec)
    base = root_dir(root, cfg)
    p = os.path.normpath(os.path.join(base, rest)) if rest else base
    if not inside(base, p):
        raise ValueError(f'{spec} climbs out of its folder')
    return p


def same_file(a, b):
    """Whether two paths name one file, however spelt or linked (it looks only at the files' metadata)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a).casefold() == os.path.realpath(b).casefold()


def forbidden_read(p, cfg=None):
    """Why the bench may not read p, or '': a name like a key, a token or a sealed file (its own name or
    the name it links to), or the token file named in a blank, however the path reaches it."""
    if sealed_like(p) or sealed_like(os.path.realpath(p)):
        return f'{os.path.basename(p)} is named like a key, a token or a sealed file, and the bench never reads one'
    for k, v in (cfg or {}).get('blanks', {}).items():
        if v and 'TOKEN' in k.upper() and same_file(resolve(v), p):
            return 'that is the token file you named, and the bench never reads it'
    return ''


def guard_read(p, cfg):
    """The bench never reads a key, a token or a sealed file, not even to hash it."""
    why = forbidden_read(p, cfg)
    if why:
        raise ValueError(why)


def read_text(p, cfg):
    guard_read(p, cfg)
    with open(p, encoding='utf-8', errors='replace') as f:
        return f.read()


def registered(cfg):
    v = cfg.get('blanks', {}).get('approved_public_utc', '')
    return bool(UTC_FORM.match(v)) and parse_utc(v) <= now()


def next_step(cfg):
    return next((s['id'] for s in steps() if s['id'] not in cfg['done'] and not s.get('optional')), None)


def number(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def spend_pool(p):
    return 'anthropic' if p.get('family') == 'anthropic' else 'others'


def required_balance(p):
    """The balance a provider's account must hold: the larger of the planned spend with its margin and the
    registered cap, so the account cannot stop the run before the run's own cap does."""
    m = p.get('margin')
    if isinstance(m, bool) or not isinstance(m, (int, float)):
        return None, 'the margin of this provider is still a TODO in kit/PROVIDERS.json'
    sp = p.get('planned_spend') if isinstance(p.get('planned_spend'), dict) else {}
    a = sp.get('amount')
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0 or not math.isfinite(m) or m < 0:
        return None, ('the planned spend of this provider in kit/PROVIDERS.json must be a finite number above zero, and its margin '
                      'a finite number of at least zero')
    rc = p.get('registered_cap')
    if isinstance(rc, bool) or not isinstance(rc, (int, float)):
        return None, 'the registered cap of this provider is still a TODO in kit/PROVIDERS.json'
    if not math.isfinite(rc) or rc <= 0:
        return None, 'the registered cap of this provider in kit/PROVIDERS.json must be a finite number above zero'
    return round(max(float(a) * (1 + m), float(rc)), 2), ''


def cap_problems():
    """The per-test cap of decision E14, checked again from kit/PROVIDERS.json before the run-start record: the
    cap the kit carries is never above E14's (the bench holds its own copy), every planned spend is in GBP and
    above zero, and the planned spend and the written registered caps stay within the cap for each pool."""
    probs = []
    cap = providers_doc().get('spending_cap')
    if not isinstance(cap, dict):
        return ['kit/PROVIDERS.json carries no spending_cap (decision E14); the factory writes one, and the run does not start without it']
    if str(cap.get('currency', '')).upper() != E14['currency']:
        probs.append(f'the spending cap in kit/PROVIDERS.json is not in {E14["currency"]}, the currency of decision E14')
    lim = {}
    for pool in ('anthropic', 'others'):
        v = number(cap.get(pool))
        if v is None or v < 0:
            probs.append(f'the spending cap for {POOL_WORDS[pool]} in kit/PROVIDERS.json is not a number of GBP')
            continue
        if v > E14[pool]:
            probs.append(f'the spending cap for {POOL_WORDS[pool]} in kit/PROVIDERS.json (GBP {v:.2f}) is above decision E14 (GBP {E14[pool]:.2f})')
        lim[pool] = min(v, E14[pool])
    planned, caps = {'anthropic': 0.0, 'others': 0.0}, {'anthropic': 0.0, 'others': 0.0}
    for p in providers():
        sp = p.get('planned_spend') if isinstance(p.get('planned_spend'), dict) else {}
        a = number(sp.get('amount'))
        if a is None or a <= 0:
            probs.append(f'{p.get("name")}: the planned spend in kit/PROVIDERS.json must be a finite number above zero')
            continue
        if str(sp.get('currency', '')).upper() != E14['currency']:
            probs.append(f'{p.get("name")}: the planned spend is not in GBP, the currency of decision E14')
            continue
        planned[spend_pool(p)] += a
        rc = number(p.get('registered_cap'))
        if rc is not None:
            caps[spend_pool(p)] += rc
            if rc < a:
                probs.append(f'{p.get("name")}: the registered cap (GBP {rc:.2f}) is below the planned spend (GBP {a:.2f})')
    for pool, v in lim.items():
        if planned[pool] > v + 1e-9:
            probs.append(f'the planned spend on {POOL_WORDS[pool]} sums to GBP {planned[pool]:.2f}, above the cap of GBP {v:.2f} (decision E14)')
        if caps[pool] > v + 1e-9:
            probs.append(f'the registered caps on {POOL_WORDS[pool]} sum to GBP {caps[pool]:.2f}, above the cap of GBP {v:.2f} (decision E14)')
    return probs


def todo_markers(o, where_, out=None):
    """Every TODO marker in a part of the kit: a todo check, or a value that starts with TODO."""
    out = [] if out is None else out
    if isinstance(o, dict):
        if o.get('check') == 'todo':
            out.append(f'{where_}: {o.get("what", "TODO")}')
            return out
        for k, v in o.items():
            todo_markers(v, f'{where_}.{k}', out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            todo_markers(v, f'{where_}[{i}]', out)
    elif is_todo(o):
        out.append(f'{where_}: {o}')
    return out


def exports():
    """Every variable the run sheet's commands export, read again from kit/STEPS.json, in table order: [(variable, step id)]."""
    out = []
    for s in steps():
        for c in s.get('commands', []):
            m = re.match(r'^\s*export\s+([A-Z][A-Z0-9_]*)=', str(c.get('run', '')))
            if m:
                out.append((m.group(1), s['id']))
    return out


def exported_keys():
    """The key and token variables the run sheet's commands export, each once, with the first step that exports it."""
    out = []
    for var, sid in exports():
        if KEY_VAR_RE.search(var) and var not in [v for v, _ in out]:
            out.append((var, sid))
    return out


def late_exports(p):
    """Where a provider's key is exported only after the pre-flight, read again from kit/STEPS.json as it stands: the step
    ids of those export lines (empty when an export stands before the pre-flight, or the run sheet exports the key nowhere)."""
    ids = step_ids()
    pf = next((i for i, s in enumerate(steps()) if s.get('preflight')), None)
    at = [ids.index(sid) for var, sid in exports() if var == p.get('key_env')]
    if pf is None or not at or min(at) < pf:
        return []
    return [ids[i] for i in at]


def preflight_state(cfg):
    """(complete, what is missing): every provider's three read-backs recorded as a pass, after the
    registration was public and on the folder as sealed (the FOLDER-MANIFEST.sha256 it was recorded under is
    the one in the folder), each in lab/bench.json and each with its line in the lab log; no TODO left in the
    pre-flight table (an empty provider list stays a TODO until a person confirms the run calls none); the
    per-test cap kept; every key the run sheet exports named by a provider or ruled out with a reason; and
    no provider key exported only after the pre-flight unless a person has written why it is in the shell
    in time. The last two are read again from kit/STEPS.json, so an edit after the factory ran is noticed."""
    missing = [f'kit/PROVIDERS.json: {t}' for t in providers_doc().get('todo', [])]
    missing += [f'the per-test cap: {c}' for c in cap_problems()]
    named = {p.get('key_env') for p in providers()}
    ruled = providers_doc().get('keys_ruled_out')
    ruled = ruled if isinstance(ruled, dict) else {}
    for var, sid in exported_keys():
        why = ruled.get(var)
        if var not in named and not (isinstance(why, str) and why.strip() and not is_todo(why)):
            missing.append(f'the run sheet exports {var} at step {sid}, and no provider in kit/PROVIDERS.json names it: add the provider '
                           'it unlocks, or name it in keys_ruled_out with the reason it unlocks none')
    approved = cfg.get('blanks', {}).get('approved_public_utc', '')
    text = read_log()
    sealed = sha256(FOLDER_MANIFEST) if os.path.exists(FOLDER_MANIFEST) else None
    for p in providers():
        left = todo_markers(p, f'kit/PROVIDERS.json provider {p.get("name")}')
        if left:
            missing.append(f'{p.get("name")}: {len(left)} TODO marker(s) left in kit/PROVIDERS.json (selfcheck lists them)')
        late = late_exports(p)
        lines = [e for e in p.get('export_lines', []) if isinstance(e, dict)]
        settled = any(isinstance(e.get('resolved'), str) and e['resolved'].strip() and not is_todo(e['resolved']) for e in lines)
        if late and not settled and not any(is_todo(e.get('todo')) for e in lines):
            missing.append(f'the run sheet exports {p.get("key_env")} only at step {late[0]}, after the pre-flight: settle it in '
                           'kit/PROVIDERS.json (export_lines: in "resolved", why the key is in the shell when the pre-flight runs)')
        rec = cfg.get('preflight', {}).get(p.get('name'), {})
        for ch in CHECKS:
            r = rec.get(ch)
            if not r:
                missing.append(f'{p.get("name")} {ch}: not recorded')
            elif r.get('result') != 'pass':
                missing.append(f'{p.get("name")} {ch}: recorded as FAIL')
            elif not (UTC_FORM.match(approved) and r.get('utc', '') >= approved and str(r.get('read_utc') or r.get('utc', '')) >= approved):
                missing.append(f'{p.get("name")} {ch}: recorded before the registration was public')
            elif sealed is None or r.get('manifest_sha256') != sealed:
                missing.append(f'{p.get("name")} {ch}: recorded on the folder before its manifest last changed: read it back again on '
                               'the folder as sealed')
            elif not UTC_FORM.match(str(r.get('read_utc') or r.get('utc') or '')) or \
                    parse_utc(r.get('read_utc') or r['utc']) < now() - dt.timedelta(hours=PREFLIGHT_MAX_AGE_HOURS):
                missing.append(f'{p.get("name")} {ch}: read more than {PREFLIGHT_MAX_AGE_HOURS} hours ago '
                               f'({r.get("read_utc") or r.get("utc")}): read it back again, since a balance can move')
            elif f'PREFLIGHT {r.get("id")} ' not in text:
                missing.append(f'{p.get("name")} {ch}: its line is not in the lab log')
    return not missing, missing


def describe(c):
    k, p = c.get('check'), c.get('path', '')
    if k == 'todo':
        return str(c.get('what', 'TODO'))
    simple = {'registered': 'the registration is recorded as approved and public (approved_public_utc)',
              'window_open': 'the window has not closed',
              'preflight_complete': 'every provider\'s key, model string and balance read back, each recorded as a pass in the lab log after the registration was public, on the folder as sealed and within the last six hours, within the per-test cap'}
    if k in simple:
        return simple[k]
    if k == 'blank_set':
        return f'the blank {c.get("blank")} is recorded'
    if k == 'step_done':
        return f'step {c.get("step")} is recorded as done'
    if k == 'log_contains':
        return f'the lab log holds the words: {c.get("text")}'
    if k == 'exit_codes':
        return f'the exit codes given with --exit are {c.get("expect")}' if 'expect' in c else f'each exit code is one of {c.get("allowed")}'
    if k == 'file_exists':
        return f'{p} exists'
    if k == 'dir_exists':
        return f'the folder {p} exists'
    if k == 'file_absent':
        return f'{p} does not exist yet'
    if k == 'file_nonempty':
        return f'{p} exists and is not empty'
    if k == 'file_contains':
        return f'{p} holds ' + (f'the words "{c["text"]}"' if 'text' in c else f'the value of the blank {c.get("blank")}')
    if k == 'last_line_equals':
        return f'the last line of {p} reads "{c.get("text")}"'
    if k == 'json_parses':
        return f'{p} parses as JSON'
    if k == 'json_field_equals':
        return f'the field {c.get("field")} of {p} equals the expected value'
    if k == 'sha256_registered':
        return f'the SHA-256 of {p} equals the registered hash of {c.get("name")} ({c.get("table", "registered")})'
    if k == 'record_sha256':
        return f'the SHA-256 of {p} is recorded'
    if k == 'manifest_ok':
        return f'every line of {c.get("manifest")} checks OK, from {c.get("base", "unit:")}'
    return 'an unknown check: ' + json.dumps(c)


def manifest_rows(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f.read().splitlines():
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            m = re.match(r'^([0-9a-fA-F]{64}) [ *](.+)$', s)
            if m:
                rows.append((m.group(1).lower(), m.group(2)))
                continue
            m = re.match(r'^SHA256 \((.+)\) = ([0-9a-fA-F]{64})$', s)
            rows.append((m.group(2).lower(), m.group(1)) if m else (None, s))
    return rows


def dig(d, field):
    for part in str(field).split('.'):
        if not isinstance(d, dict) or part not in d:
            raise KeyError(field)
        d = d[part]
    return d


def run_check(c, cfg, exits):
    """One check of a pass rule: (passed, facts, the problem if it did not pass)."""
    k = c.get('check')
    facts = {}
    try:
        if k == 'todo':
            return False, facts, 'TODO in kit/STEPS.json (a person writes this rule before the bench can check it): ' + str(c.get('what', ''))
        if k == 'registered':
            ok = registered(cfg)
            return ok, facts, None if ok else 'approved_public_utc is not recorded'
        if k == 'window_open':
            raw = steps_doc().get('window_close_utc')
            if raw is None:
                return True, facts, None
            if is_todo(raw):
                return False, facts, ('the close of the window is still a TODO in kit/STEPS.json (window_close_utc): give it as UTC, '
                                      'or null if no window binds the run')
            w = window()
            if w is None:
                return False, facts, 'window_close_utc in kit/STEPS.json is not written as 2026-10-11T23:59:00Z'
            facts['hours_left'] = hours_left()
            return now() < w, facts, None if now() < w else 'the window has closed'
        if k == 'blank_set':
            ok = bool(cfg['blanks'].get(c['blank']))
            return ok, facts, None if ok else f'{c["blank"]} is not recorded'
        if k == 'step_done':
            ok = c['step'] in cfg['done']
            return ok, facts, None if ok else f'step {c["step"]} is not recorded as done'
        if k == 'preflight_complete':
            ok, missing = preflight_state(cfg)
            return ok, facts, None if ok else 'the pre-flight is not complete: ' + '; '.join(missing)
        if k == 'log_contains':
            ok = str(c['text']) in read_log()
            return ok, facts, None if ok else f'the lab log does not hold the words: {c["text"]}'
        if k == 'exit_codes':
            if exits is None:
                return False, facts, 'give the exit codes your terminal printed, in order:  --exit 0 ...'
            facts['exit_codes'] = list(exits)
            if 'expect' in c:
                ok = list(exits) == [int(x) for x in c['expect']]
            else:
                ok = (c.get('count') is None or len(exits) == int(c['count'])) and all(e in c['allowed'] for e in exits)
            return ok, facts, None if ok else f'the exit codes {list(exits)} are not what the rule passes ({describe(c)})'
        if k not in KNOWN_CHECKS:
            return False, facts, f'an unknown check: {k}'
        if k == 'manifest_ok':
            mp = where(c['manifest'], cfg)
            base = where(c.get('base', 'unit:'), cfg)
            if not os.path.isfile(mp):
                return False, facts, f'{c["manifest"]} is not there'
            guard_read(mp, cfg)
            rows, ok_n, bad = manifest_rows(mp), 0, []
            for h, name in rows:
                fp = os.path.normpath(os.path.join(base, name))
                if h is None or not inside(base, fp) or forbidden_read(fp, cfg):     # never hashed: a sealed name or the token file
                    bad.append(name[:80])
                elif os.path.isfile(fp) and sha256(fp) == h:
                    ok_n += 1
                else:
                    bad.append(name)
            facts[f'manifest {c["manifest"]}'] = f'{ok_n} of {len(rows)} OK'
            n_ok = c.get('count') is None or len(rows) == int(c['count'])
            ok = bool(rows) and not bad and n_ok
            why = (f'FAILED or missing: {", ".join(bad[:6])}' if bad else '') + ('' if n_ok else f' it has {len(rows)} lines, and the rule wants {c["count"]}')
            return ok, facts, None if ok else f'{c["manifest"]}: {why.strip() or "no line to check"}'
        p = where(c['path'], cfg)
        if k == 'file_exists':
            return os.path.isfile(p), facts, None if os.path.isfile(p) else f'{c["path"]} is not there'
        if k == 'dir_exists':
            return os.path.isdir(p), facts, None if os.path.isdir(p) else f'the folder {c["path"]} is not there'
        if k == 'file_absent':
            return not os.path.exists(p), facts, None if not os.path.exists(p) else f'{c["path"]} exists already'
        if k == 'file_nonempty':
            ok = os.path.isfile(p) and os.path.getsize(p) > 0
            return ok, facts, None if ok else f'{c["path"]} is missing or empty'
        if not os.path.isfile(p):
            return False, facts, f'{c["path"]} is not there'
        if k == 'file_contains':
            want = c['text'] if 'text' in c else cfg['blanks'].get(c['blank'], '')
            if not want:
                return False, facts, f'the blank {c.get("blank")} is empty, so there is nothing to look for'
            ok = str(want) in read_text(p, cfg)
            return ok, facts, None if ok else f'{c["path"]} does not hold ' + (f'the words "{want}"' if 'text' in c else f'the recorded {c["blank"]}')
        if k == 'last_line_equals':
            lines = [l for l in read_text(p, cfg).splitlines() if l.strip()]
            ok = bool(lines) and lines[-1].strip() == c['text']
            return ok, facts, None if ok else f'the last line of {c["path"]} is not "{c["text"]}"'
        if k in ('json_parses', 'json_field_equals'):
            try:
                d = json.loads(read_text(p, cfg))
            except ValueError as e:
                return False, facts, f'{c["path"]} does not parse: {e}'
            if k == 'json_parses':
                return True, facts, None
            if 'value' in c:
                want = c['value']
            elif 'blank' in c:
                want = cfg['blanks'].get(c['blank'])
            else:
                want = current_hashes().get(c.get('table', 'registered'), {}).get(c['registered'])
            try:
                got = dig(d, c['field'])
            except KeyError:
                return False, facts, f'{c["path"]} has no field {c["field"]}'
            return got == want, facts, None if got == want else f'the field {c["field"]} of {c["path"]} is not the expected value'
        guard_read(p, cfg)
        h = sha256(p)
        if k == 'record_sha256':
            facts[c.get('as') or c['path']] = h
            return True, facts, None
        if k == 'sha256_registered':
            want = current_hashes().get(c.get('table', 'registered'), {}).get(c['name'])
            facts[c['path']] = h
            if not want:
                return False, facts, f'kit/CURRENT-HASHES.json holds no registered hash for {c["name"]}'
            return h == want, facts, None if h == want else f'{c["path"]} DOES NOT EQUAL the registered hash of {c["name"]}'
    except (KeyError, ValueError, TypeError) as e:
        return False, facts, f'the check {k} cannot be read ({e}); fix it in kit/STEPS.json'
    return False, facts, f'the check {k} gave no answer'


def check_step(step, cfg, exits):
    facts, bad = {}, []
    rule = step.get('pass_rule') or [{'check': 'todo', 'what': 'no pass rule is written for this step'}]
    for c in rule:
        ok, f, problem = run_check(c, cfg, exits)
        facts.update(f)
        if not ok:
            bad.append(problem or ('not passed: ' + describe(c)))
    if exits is not None:
        facts.setdefault('exit_codes', list(exits))
    return not bad, facts, bad


def todo_list():
    out = []
    for s in steps():
        todo_markers(s, f'kit/STEPS.json step {s["id"]}', out)
    todo_markers(steps_doc().get('window_close_utc'), 'kit/STEPS.json window_close_utc', out)
    for p in providers():
        todo_markers(p, f'kit/PROVIDERS.json provider {p.get("name")}', out)
    for t in providers_doc().get('todo', []):
        out.append(f'kit/PROVIDERS.json: {t}')
    return out


def is_registration_step(s):
    return any(c.get('check') == 'registered' for c in s.get('pass_rule', []))


def ruled_out(s):
    """Whether a person has ruled a step's calls out, with the reason written in calls_out_ruled_out (sealed with the folder)."""
    why = s.get('calls_out_ruled_out')
    return isinstance(why, str) and bool(why.strip()) and not is_todo(why)


def runs_code(s):
    """Whether a step runs the unit's code: by its flag, or read again from its commands, so an edited flag cannot hide it."""
    if ruled_out(s):
        return False
    return bool(s.get('runs_unit_code')) or any(RUNS_UNIT_CODE_RE.search(str(c.get('run', ''))) for c in s.get('commands', []))


def calls_net(s):
    """Whether a step calls the network: by its flag, or read again from its commands, so an edited flag cannot hide it."""
    if ruled_out(s):
        return False
    return bool(s.get('calls_network')) or any(NETWORK_RE.search(str(c.get('run', ''))) for c in s.get('commands', []))


def gate_problems():
    """What would let the run start without its pre-flight, wherever the steps stand in the table: the table
    holds exactly one pre-flight step, carrying preflight_complete, before exactly one run-start record and
    apart from it; neither is done by prepare; no two steps share an id (a twin of the run-start record could
    stand in for it); and no step that runs the unit's code or calls the network, by its flags or by its
    commands read again here, stands before the run-start record (the registration step may call the network
    to read the registry, and never runs the unit's code), so no reordering of the table, and no flag edited by
    hand, brings a call ahead of the pre-flight."""
    probs = []
    ids = step_ids()
    for i in sorted(set(x for x in ids if ids.count(x) > 1)):
        probs.append(f'two steps share the id {i}')
    pf = [i for i, s in enumerate(steps()) if s.get('preflight')]
    rs = [i for i, s in enumerate(steps()) if s.get('run_start')]
    if len(pf) != 1:
        probs.append(f'there must be exactly one pre-flight step; there are {len(pf)}')
    if len(rs) != 1:
        probs.append(f'exactly one step must be the run-start record; there are {len(rs)}')
    if pf and rs and pf[0] >= rs[0]:
        probs.append('the pre-flight must come before the run-start record, as a step of its own')
    if pf and not any(c.get('check') == 'preflight_complete' for c in steps()[pf[0]]['pass_rule']):
        probs.append('the pre-flight step must carry the preflight_complete check')
    for i in pf + rs:
        if steps()[i].get('kind') == 'bench':
            probs.append(f'step {steps()[i]["id"]} is the pre-flight or the run-start record, and prepare never does either')
    if len(rs) == 1:
        for s in steps()[:rs[0]]:
            if s.get('preflight'):
                continue
            if runs_code(s) or (calls_net(s) and not is_registration_step(s)):
                probs.append(f'step {s["id"]} runs the unit\'s code or calls the network, and stands before the run-start record '
                             f'(step {steps()[rs[0]]["id"]}): no call may come before the pre-flight and the run-start record')
    return probs


SEAL_HINT = ('python3 bench/bench.py manifest --write --reviewed-by NAME --review WHERE   (your entry of who reviewed the folder '
             'and where the review lives; the bench does not see the review)')


def seal_problem(cfg):
    """Why the folder as it stands carries no entry of the author's recording it as reviewed, or ''."""
    seal = cfg.get('seal') or {}
    if not seal:
        return 'no entry of yours records this folder as reviewed'
    if not os.path.exists(FOLDER_MANIFEST) or seal.get('manifest_sha256') != sha256(FOLDER_MANIFEST):
        return 'your entry records another FOLDER-MANIFEST.sha256 than the one in the folder'
    if f'SEAL {seal.get("id")} ' not in read_log():
        return 'the lab log holds no line for your entry'
    return ''


def refuse_unless_sound(cfg, what):
    """The bench records nothing (no step, no prepare, no pre-flight result) while kit/STEPS.json cannot hold
    the run at its pre-flight, or while the folder differs from FOLDER-MANIFEST.sha256 (the folder as sealed):
    an edit to the table after the seal is caught at the next command, whatever step it moved."""
    probs = gate_problems()
    drift, _ = manifest_drift(cfg)
    if not probs and not drift:
        return
    log(f'{what}: refused, ' + '; '.join((['kit/STEPS.json cannot hold the run at its pre-flight'] if probs else []) +
                                         (['the folder differs from FOLDER-MANIFEST.sha256'] if drift else [])))
    lines = []
    if probs:
        lines += ['kit/STEPS.json cannot hold the run at its pre-flight, so nothing is recorded until it is mended and reviewed:']
        lines += ['  ' + p for p in probs]
    if drift:
        lines += ['the folder differs from FOLDER-MANIFEST.sha256, so it is not the folder that was sealed:']
        lines += ['  ' + d for d in drift[:12]] + ['  After a reviewed change:  ' + SEAL_HINT]
    sys.exit(f'REFUSED: {what}.\n' + '\n'.join(lines))


def lint_command(cmd, key_vars):
    """What makes a printed command unsafe: a value shaped like a key, set -x, a command that prints a
    key variable, or a key variable in an address (where logs keep it)."""
    probs = []
    if looks_like_a_secret(cmd):
        probs.append('it holds a value shaped like a key or token')
    if re.search(r'\bset\s+-[a-z]*x', cmd):
        probs.append('set -x prints every value')
    if re.search(r'(^|[;&|]\s*)(printenv|env)\s*($|[;&|])', cmd):
        probs.append('it prints the environment')
    for v in key_vars:
        ev = re.escape(v)
        if re.search(r'\b(echo|printf|printenv)\b[^|;&]*\$\{?' + ev + r'\b', cmd) and not re.search(
                r"printf\s+'[^']*%s[^']*'\s+\"\$\{?" + ev + r"\}?\"\s*\|\s*curl\b.*-H\s+@-", cmd):
            probs.append(f'it may print ${v}')
        if re.search(r'[?&][A-Za-z_]*key=\$\{?' + ev, cmd):
            probs.append(f'${v} in an address leaks into logs')
    return probs


def covered_files(cfg=None):
    out, skipped = [], []
    for top in COVERED:
        p = os.path.join(LAB, top)
        if os.path.isfile(p):
            out.append(top)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not d.startswith('.venv'))
                for f in sorted(files):
                    rel = os.path.relpath(os.path.join(root, f), LAB).replace(os.sep, '/')
                    if f == '.DS_Store':
                        continue
                    (skipped if forbidden_read(os.path.join(root, f), cfg) else out).append(rel)
    return sorted(out), skipped


def manifest_drift(cfg=None):
    """(problems, current text): how the folder differs from FOLDER-MANIFEST.sha256."""
    files, skipped = covered_files(cfg)
    current = ''.join(f'{sha256(os.path.join(LAB, *rel.split("/")))}  {rel}\n' for rel in files)
    probs = [f'{rel} is named like a key, a token or a sealed file, or is the token file you named; the bench does not read it, '
             'and it does not belong here' for rel in skipped]
    if not os.path.exists(FOLDER_MANIFEST):
        return probs + ['FOLDER-MANIFEST.sha256 is not there'], current
    want = {}
    with open(FOLDER_MANIFEST, encoding='utf-8') as f:
        for line in f.read().splitlines():
            m = re.match(r'^([0-9a-f]{64})  (.+)$', line)
            if m:
                want[m.group(2)] = m.group(1)
    have = {}
    for line in current.splitlines():
        h, rel = line.split('  ', 1)
        have[rel] = h
    for rel in sorted(set(want) | set(have)):
        if rel not in have:
            probs.append(f'{rel} is in the manifest and not in the folder')
        elif rel not in want:
            probs.append(f'{rel} is in the folder and not in the manifest')
        elif want[rel] != have[rel]:
            probs.append(f'{rel} has changed since the manifest was written')
    return probs, current


def shell_setup(cfg):
    lines = []
    for bid, meta in blank_table().items():
        v = cfg['blanks'].get(bid, '')
        if meta.get('variable') and v:
            lines.append(f'{bid}={shlex.quote(resolve(v) if meta.get("path") else v)}')
    lines.append('cd "$UNIT"')
    return lines + list(steps_doc().get('shell_vars', []))


def render_cmd(run, cfg):
    return PLACEHOLDER.sub(lambda m: cfg['blanks'].get(placeholder_id(m.group(1))) or m.group(0), run)


def show_preflight(cfg, intro=True):
    provs = providers()
    if not provs:
        print('  kit/PROVIDERS.json names no provider: the run calls none, and there is nothing to read back.')
        ok, missing = preflight_state(cfg)
        print('  Pre-flight: ' + ('COMPLETE.' if ok else 'NOT COMPLETE:'))
        for m in missing:
            print('    ' + m)
        return
    if intro:
        print('\n  Run each read-back in your one shell, where the key is already exported. The bench never reads the key:')
        print('  only the status line, the model string served and the balance figure go into the record.')
    for p in provs:
        need, why = required_balance(p)
        sp = p.get('planned_spend', {})
        cur = sp.get('currency', '')
        head = f'\n  PROVIDER {p.get("name")}: model {p.get("model")}, key in ${p.get("key_env")}, planned spend {float(sp.get("amount", 0)):.2f} {cur}'
        print(head + (f'; required balance {need:.2f} {cur} (the larger of the planned spend with its margin {p.get("margin")} and the '
                      f'registered cap {p.get("registered_cap")})' if need is not None else f'; {why}'))
        for e in p.get('export_lines', []):
            if is_todo(e.get('todo')):
                for line in wrap(e['todo'], '    '):
                    print(line)
                print(f'      {e.get("run")}')
            else:
                print(f'    The run sheet exports this key at step {e.get("step")}:  {e.get("run")}' +
                      (f'   (resolved: {e["resolved"]})' if e.get('resolved') else ''))
        for ch in CHECKS:
            rb = p.get('readback', {}).get(ch, {})
            cmds = rb.get('commands')
            print(f'    [{ch}]')
            if is_todo(cmds):
                print(f'      {cmds}')
            else:
                for c in cmds or []:
                    print('      ' + c)
            for line in wrap(rb.get('instructions', ''), '      '):
                print(line)
            for line in wrap('A pass: ' + rb['expect'], '      ') if rb.get('expect') else []:
                print(line)
            r = cfg.get('preflight', {}).get(p.get('name'), {}).get(ch)
            print('      Recorded: ' + (f'{r["result"].upper()} at {r["utc"]} (lab log id {r["id"]})' if r else 'nothing yet'))
        n = p.get('name')
        bal = p.get('readback', {}).get('balance', {})
        page = shlex.quote(str(bal.get('page') or 'PAGE'))
        print('    Record each result (a read-back that does not pass is recorded with fail, and the run does not start):')
        print(f'      python3 bench/bench.py preflight {n} key pass --note "HTTP 200"')
        print(f'      python3 bench/bench.py preflight {n} model pass --served {shlex.quote(str(p.get("model")))}')
        if isinstance(bal.get('commands'), list) and bal.get('commands'):
            print(f'      python3 bench/bench.py preflight {n} balance pass --amount N --currency {cur} --source api --note "the figure as printed; any rate and its source"')
        print(f'      python3 bench/bench.py preflight {n} balance pass --amount N --currency {cur} --source console --page {page} '
              '--read-utc UTC [--screenshot-sha256 HEX] --note "any rate and its source"')
    ok, missing = preflight_state(cfg)
    print('\n  Pre-flight: ' + ('COMPLETE.' if ok else 'NOT COMPLETE:'))
    for m in missing:
        print('    ' + m)


def run_started(cfg):
    """Whether the run has started: the run-start record is recorded in lab/bench.json, or its line stands in the lab log,
    which init --force and a lab/bench.json removed by hand both leave in place (T4's F16)."""
    rs = run_start_index()
    if rs is None:
        return False
    rid = steps()[rs]['id']
    if cfg and rid in cfg.get('done', {}):
        return True
    return bool(re.search(r'^- \S+ (?:\[[^\]\n]*\] )?' + re.escape(rid) + r' done: ', read_log(), re.M))


def cmd_init(args):
    cfg = None
    if os.path.exists(CFG):
        try:
            with open(CFG, encoding='utf-8') as f:
                cfg = json.load(f)
        except ValueError:
            cfg = None
    if run_started(cfg):
        sys.exit('REFUSED: the run has started (its run-start record stands in lab/bench.json or in lab/LAB-LOG.md), and the bench '
                 'state is never made again after it. Nothing changed.')
    if os.path.exists(CFG) and not args.force:
        sys.exit('lab/bench.json exists already; nothing changed')
    if os.path.exists(CFG):
        kept = f'{CFG}.replaced-{stamp().replace(":", "")}'
        os.replace(CFG, kept)
        log(f'bench initialised again with --force; the earlier state is kept as lab/{os.path.basename(kept)}')
    cfg = {'version': 1, 'engine': 'eden-lab-bench 1.2', 'created_utc': stamp(), 'done': {},
           'blanks': {k: m.get('default', '') for k, m in blank_table().items()}, 'preflight': {}, 'preflight_history': []}
    save(cfg)
    log('bench initialised')
    print('lab/bench.json written. Next:  python3 bench/bench.py selfcheck   and   python3 bench/bench.py blanks')


def cmd_set(args):
    cfg = load(check_folders=False)   # so that a folder blank pointing somewhere refused can be set again
    table = blank_table()
    if args.key not in table:
        sys.exit('unknown blank; see  bench.py blanks')
    v = args.value
    if table[args.key].get('path') and v:
        why = bad_folder_blank(args.key, v)
        if why:
            sys.exit(f'REFUSED: {why}. Nothing was recorded.')
    if looks_like_a_secret(v):
        log(f'blank refused: {args.key} was given a value shaped like a key or token; nothing was stored')
        sys.exit('REFUSED: that value is shaped like a key or a token. The bench never stores one. Nothing was recorded.')
    if 'KEYCHAIN' in args.key.upper() and "'" in v:
        sys.exit('REFUSED: a keychain label with a single quote would break the export line. Rename the keychain item or give the label without it.')
    if table[args.key].get('utc') or args.key.endswith('_utc'):
        if not UTC_FORM.match(v):
            sys.exit('write the time as 2026-10-03T17:20:00Z and nothing looser')
        if parse_utc(v) > now():
            sys.exit('REFUSED: that time is later than the clock. Record a time only after it has passed.')
    if 'TOKEN' in args.key.upper():
        unit_run = os.path.join(os.path.abspath(root_dir('unit', cfg)), 'run')
        if inside(unit_run, os.path.abspath(resolve(v))):
            sys.exit('the token file must stay outside the unit\'s run folder')
    cfg['blanks'][args.key] = v
    save(cfg)
    log(f'blank set: {args.key}' + ('' if is_private(args.key) else f' = {v}'))
    print('recorded')


def cmd_blanks(args):
    cfg = load()
    for k, meta in blank_table().items():
        v = cfg['blanks'].get(k, '')
        shown = '(set)' if v and is_private(k) else (v or '(empty)')
        print(f'  {k:26s} {shown:40s} {meta.get("what", "")}')


def cmd_status(args):
    cfg = load()
    d = steps_doc()
    print(f'{d.get("unit") or "This unit"}: the lab bench. The run sheet is the authority.')
    if registered(cfg):
        line = f'Registration {cfg["blanks"].get("registration_id") or "(identifier not recorded)"} approved and public at {cfg["blanks"]["approved_public_utc"]}.'
        if hours_left() is not None:
            line += f' Hours left in the window: {hours_left()}'
        print(line)
    else:
        print('The registration is not yet recorded as approved and public. Only the copy-and-hash steps (bench.py prepare) may be done.')
    for s in steps():
        dn = cfg['done'].get(s['id'])
        tags = {'bench': '  [prepare]', 'preflight': '  [pre-flight]'}.get(s.get('kind'), '')
        tags += '  [run start]' if s.get('run_start') else ''
        tags += '  (optional)' if s.get('optional') else ''
        print(f'  [{"x" if dn else " "}] {s["id"]:6s} {s["title"]}{tags}' + (f'   ({dn["utc"]})' if dn else ''))
    if providers() or providers_doc().get('todo'):
        ok, missing = preflight_state(cfg)
        print('Pre-flight: ' + ('complete' if ok else f'not complete, {plural_items(len(missing))} left (python3 bench/bench.py preflight lists them)'))
    probs = gate_problems()
    if probs:
        print('THE TABLE CANNOT HOLD THE RUN AT ITS PRE-FLIGHT: ' + '; '.join(probs) + '. Nothing is recorded until it is mended and reviewed.')
    drift, _ = manifest_drift(cfg)
    if drift:
        print(f'The folder differs from FOLDER-MANIFEST.sha256 ({plural_items(len(drift))}): nothing is recorded until it is sealed again '
              '(python3 bench/bench.py manifest).')
    todos = todo_list()
    if todos:
        print(f'TODO markers left for a person: {len(todos)} (selfcheck fails until each is filled; kit/TODO.md lists them)')
    empty = [k for k in blank_table() if not cfg['blanks'].get(k)]
    if empty:
        print('Blanks still empty: ' + ', '.join(empty))
    nxt = next_step(cfg)
    print('Next: ' + (f'step {nxt}.  python3 bench/bench.py show' if nxt else 'nothing; every step is recorded'))


def cmd_show(args):
    cfg = load()
    sid = args.step or next_step(cfg)
    if sid is None:
        print('every step is recorded as done')
        return
    step = step_by_id(sid)
    if step is None:
        sys.exit('no such step; steps are: ' + ' '.join(step_ids()))
    if step.get('waits') and not registered(cfg):
        print(f'Step {sid} WAITS FOR REGISTRATION. Record the approval time first:  bench.py set approved_public_utc 2026-..-..T..:..:..Z')
        return
    probs = gate_problems()
    if probs:
        sys.exit('THE TABLE CANNOT HOLD THE RUN AT ITS PRE-FLIGHT, so the bench shows no command until kit/STEPS.json is mended and '
                 'reviewed:\n  ' + '\n  '.join(probs))
    drift, _ = manifest_drift(cfg)
    if drift and (step.get('preflight') or runs_code(step) or calls_net(step)):
        sys.exit(f'The folder differs from FOLDER-MANIFEST.sha256, so the bench shows the commands of step {sid}, which call out, only '
                 'from the folder as sealed:\n  ' + '\n  '.join(drift[:12]) + '\n  After a reviewed change:  ' + SEAL_HINT)
    at = f'run sheet line {step["run_sheet_line"]}' if step.get('run_sheet_line') else 'a bench step, not in the run sheet'
    print(f'STEP {sid}: {step["title"]}   ({at})')
    if drift:
        print('  Note: the folder differs from FOLDER-MANIFEST.sha256; the bench records nothing until it is sealed again (bench.py manifest).')
    rs = run_start_index()
    if rs is not None and step_ids().index(sid) > rs and steps()[rs]['id'] not in cfg['done']:
        print(f'  THE RUN HAS NOT STARTED: run nothing below until step {steps()[rs]["id"]} (the run-start record) is recorded.')
    shown = 0
    for t in step.get('text', []):
        for line in wrap(t):
            if shown < 60:
                print(line)
            shown += 1
    if shown > 60:
        print(f'  (the rest of this step is in kit/RUN-SHEET.md, from line {step.get("run_sheet_line")})')
    for n in step.get('notes', []):
        print('  Note: ' + n)
    if step.get('preflight'):
        show_preflight(cfg, intro=False)
    cmds = step.get('commands', [])
    if cmds:
        if runs_code(step) or calls_net(step):
            print('\n  These commands run the unit\'s code or call the network: run them only in your one shell, never in an AI terminal.')
        print('\n  Shell variables, once per shell (your paths):')
        for line in shell_setup(cfg):
            print('    ' + line)
        print('\n  Commands, exactly as the run sheet gives them:')
        for c in cmds:
            print('    ' + render_cmd(c['run'], cfg))
        notes = [(c['run'], c['note']) for c in cmds if c.get('note')]
        if notes:
            print('\n  The run sheet\'s notes on those lines:')
            for run, note in notes:
                print(f'    {run.strip()[:48]}...  {note}' if len(run.strip()) > 48 else f'    {run.strip()}  {note}')
    print('\n  The bench checks:')
    for c in step.get('pass_rule', []):
        for i, line in enumerate(wrap(describe(c), '      ')):
            print(('    - ' + line.strip()) if i == 0 else line)
    if step.get('run_start'):
        ok, missing = preflight_state(cfg)
        if not ok:
            print('\n  THE RUN DOES NOT START YET: the pre-flight is not complete (python3 bench/bench.py preflight).')
    if step.get('kind') == 'bench':
        print('\n  bench.py prepare does this step: it only copies and hashes.')
    else:
        exits = any(c.get('check') == 'exit_codes' for c in step.get('pass_rule', []))
        print(f'\n  When it is done:  python3 bench/bench.py done {sid}' + ('  --exit ...   (the exit codes your terminal printed)' if exits else ''))


def run_op(op, cfg, out):
    unit = root_dir('unit', cfg)
    if op.get('op') == 'mkdir':
        p = where(op['path'], cfg)
        if not inside(unit, p):
            sys.exit(f'REFUSED: {op["path"]} is outside the unit folder; prepare writes only there')
        os.makedirs(p, exist_ok=True)
        out.append(f'mkdir {op["path"]}')
        return
    if op.get('op') != 'copy':
        sys.exit(f'REFUSED: an unknown plan step in kit/STEPS.json: {op}')
    root, _, pattern = op['from'].partition(':')
    if root not in ('deposit', 'stage'):
        sys.exit(f'REFUSED: prepare copies only from the deposit or the stage, not {op["from"]}')
    base = root_dir(root, cfg)
    d, b = os.path.split(pattern)
    folder = os.path.join(base, d)
    if any(ch in b for ch in '*?['):
        listing = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        names = [os.path.join(d, n).replace(os.sep, '/') if d else n for n in listing
                 if os.path.isfile(os.path.join(folder, n)) and fnmatch.fnmatchcase(n, b) and (b.startswith('.') or not n.startswith('.'))]
    else:
        names = [pattern] if os.path.isfile(os.path.join(base, pattern)) else []
    exp = op.get('expect')
    if exp is not None and sorted(names) != sorted(exp):
        sys.exit(f'REFUSED: {op["from"]} matches {names}, and the run sheet\'s copy expects {exp}. Nothing more is copied. Record it in unit/run/DEFECT-LOG.md.')
    if not names:
        sys.exit(f'REFUSED: {op["from"]} matches no file. Nothing more is copied.')
    dest = where(op['to'], cfg)
    if not inside(unit, dest):
        sys.exit(f'REFUSED: {op["to"]} is outside the unit folder; prepare writes only there')
    for n in names:
        src = os.path.join(base, n)
        why = forbidden_read(src, cfg)
        if why:
            sys.exit(f'REFUSED: {why}; prepare never copies it')
        if op.get('to_dir'):
            os.makedirs(dest, exist_ok=True)
            target = os.path.join(dest, os.path.basename(n))
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            target = dest
        shutil.copy2(src, target)
        out.append(f'copied {root}:{n} to unit:{os.path.relpath(target, unit)}')


def verify_table(base, table, out, label, cfg=None):
    bad = 0
    for name, h in sorted(table.items()):
        p = os.path.normpath(os.path.join(base, name))
        ok = inside(base, p) and os.path.isfile(p) and not forbidden_read(p, cfg) and sha256(p) == h
        out.append(f'{label}:{name}: ' + ('OK' if ok else 'FAILED'))
        bad += 0 if ok else 1
    return bad


def cmd_prepare(args):
    """The steps that only copy registered files into the unit folder and check every hash. No unit code runs."""
    cfg = load()
    cur = current_hashes()          # refuses before anything is copied if the table of registered hashes is missing
    probs = gate_problems()
    if probs:
        sys.exit('REFUSED: kit/STEPS.json cannot hold the run at its pre-flight: ' + '; '.join(probs) + '. Nothing was copied.')
    dep = root_dir('deposit', cfg)
    if not os.path.isdir(dep):
        sys.exit(f'DEPOSIT is not a folder: {dep}')
    out = ['# (1) the registered files, checked before anything is copied']
    bad = verify_table(dep, cur.get('registered', {}), out, 'deposit', cfg)
    if cur.get('stage'):
        bad += verify_table(root_dir('stage', cfg), cur['stage'], out, 'stage', cfg)
    os.makedirs(os.path.join(LABDIR, 'outputs'), exist_ok=True)
    record = os.path.join(LABDIR, 'outputs', 'prepare-checks.txt')
    n_reg = len(cur.get('registered', {}))
    if bad:
        with open(record, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        log(f'prepare FAILED at the bench: {bad} registered file(s) are not the registered bytes; see lab/outputs/prepare-checks.txt')
        print(f'FAILED: {bad} registered file(s) missing or not the registered bytes. Stop. Record it in unit/run/DEFECT-LOG.md. Nothing was copied.')
        sys.exit(1)
    refuse_unless_sound(cfg, 'prepare copied nothing')
    todo = [s for s in steps() if s.get('kind') == 'bench']
    if not todo:
        with open(record, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        log(f'prepare: the deposit checked, {n_reg} of {n_reg} files are the registered bytes; no copy-and-hash step to do')
        print(f'The deposit checks: {n_reg} of {n_reg} files are the registered bytes. No step of the run sheet is a copy-and-hash step the bench can do.')
        return
    out.append('# (2) the copies (a step already recorded is never copied again: its files are only checked)')
    for s in todo:
        if s['id'] in cfg['done']:
            out.append(f'step {s["id"]}: recorded already at {cfg["done"][s["id"]].get("utc")}; nothing copied')
            continue
        for op in s.get('plan', []):
            run_op(op, cfg, out)
    out.append('# (3) the pass rules')
    results, allok = [], True
    for s in todo:
        ok, facts, bad_ = check_step(s, cfg, None)
        allok = allok and ok
        results.append((s, ok, facts))
        out.append(f'step {s["id"]}: ' + ('PASS' if ok else 'NOT PASSED'))
        out += [f'  NOT PASSED: {b}' for b in bad_] + [f'  {k}: {v}' for k, v in facts.items()]
        print(f'step {s["id"]} ({s["title"]}): ' + ('PASS' if ok else 'NOT PASSED'))
        for b in bad_:
            print('  NOT PASSED: ' + b)
    with open(record, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    if not allok:
        log('prepare FAILED at the bench: see lab/outputs/prepare-checks.txt')
        print('FAILED. Stop. Record it in unit/run/DEFECT-LOG.md. No document reaches a model.')
        sys.exit(1)
    for s, ok, facts in results:
        if s['id'] not in cfg['done']:
            cfg['done'][s['id']] = dict(facts, utc=stamp())
    save(cfg)
    log('prepare done by the bench: steps ' + ' '.join(s['id'] for s in todo) + f'; deposit {n_reg} of {n_reg} OK')
    print('PASS. Steps ' + ' '.join(s['id'] for s in todo) + ' are recorded. The full output is in lab/outputs/prepare-checks.txt')


def cmd_preflight(args):
    cfg = load()
    provs = {p.get('name'): p for p in providers()}
    if not args.name:
        show_preflight(cfg)
        return
    if args.name not in provs:
        sys.exit('no such provider; the providers are: ' + ' '.join(sorted(provs)))
    if args.check not in CHECKS or args.result not in ('pass', 'fail'):
        sys.exit('give  preflight NAME CHECK RESULT  with CHECK one of key, model, balance and RESULT pass or fail')
    if not registered(cfg):
        sys.exit('the pre-flight WAITS FOR REGISTRATION: it is read after the registry shows the registration approved and public, and before the run-start record')
    refuse_unless_sound(cfg, f'the {args.check} read-back of {args.name} is not recorded')
    for v in (args.note, args.served, args.currency, args.page):
        if v and looks_like_a_secret(v):
            log(f'pre-flight record refused: a value for {args.name} {args.check} was shaped like a key or token; nothing was stored')
            sys.exit('REFUSED: a value is shaped like a key or a token. The bench never stores one. Nothing was recorded.')
    p = provs[args.name]
    rb = p.get('readback', {}).get(args.check, {})
    if args.result == 'pass' and is_todo(rb.get('commands')):
        sys.exit(f'REFUSED: the {args.check} read-back of {args.name} is still a TODO in kit/PROVIDERS.json. Write it, have it '
                 'reviewed, run it, and only then record a pass. Nothing was recorded.')
    if args.amount is not None and not math.isfinite(args.amount):
        sys.exit('REFUSED: give --amount as the finite figure the read-back showed. Nothing was recorded.')
    # The folder matched FOLDER-MANIFEST.sha256 a moment ago (refuse_unless_sound): the record names that manifest, and counts
    # only while it stands, so a kit changed and sealed again after this read-back needs the read-back again.
    rec = {'result': args.result, 'utc': stamp(), 'note': (args.note or '').strip(), 'manifest_sha256': sha256(FOLDER_MANIFEST)}
    if args.check == 'key' and args.result == 'pass':
        statuses = re.findall(r'\bHTTP\s*(\d{3})\b', rec['note'])
        if not rec['note']:
            sys.exit('REFUSED: give --note with the status line the read-back printed, such as "HTTP 200". Nothing was recorded.')
        if any(s != '200' for s in statuses) or (str(rb.get('source', '')).startswith('factory default') and '200' not in statuses):
            sys.exit('REFUSED: a key passes only on the status line HTTP 200; HTTP 401 or 403 means the key is not valid and HTTP 402 '
                     'that the account is not funded. Record it as fail. Nothing was recorded.')
    if args.check == 'model':
        if not args.served:
            sys.exit('give --served, the model string the read-back printed')
        rec['served'] = args.served.strip()
        if args.result == 'pass' and rec['served'] != p.get('model'):
            sys.exit(f'REFUSED: the served string {rec["served"]!r} is not the planned model string {p.get("model")!r}. Record it as fail.')
    if args.check == 'balance':
        if args.amount is None or not args.currency:
            sys.exit('give --amount and --currency: the balance the read-back showed, in the planned currency (state any rate in --note)')
        rec['amount'], rec['currency'] = float(args.amount), args.currency.strip().upper()
        if args.result == 'pass':
            if rec['amount'] <= 0:
                sys.exit('REFUSED: a balance of zero or less never passes: a prepaid account at zero answers the first call with HTTP 402, '
                         'as Unit 06\'s Run B found. Record it as fail, and fund the account before the run.')
            need, why = required_balance(p)
            if need is None:
                sys.exit('REFUSED: ' + why + '. A balance cannot pass until the margin is written.')
            want = str(p.get('planned_spend', {}).get('currency', '')).upper()
            if rec['currency'] != want:
                sys.exit(f'REFUSED: record the balance in the planned currency, {want}, stating the rate and its source in --note.')
            if rec['amount'] < need:
                sys.exit(f'REFUSED: {rec["amount"]:.2f} {want} is less than the required {need:.2f} {want}. Record it as fail, and fund the account before the run.')
            rec['required'] = need
            # How the figure was read: by the read-back's own API call, or by hand on the provider's page (T4's F13).
            if args.source not in ('api', 'console'):
                sys.exit('REFUSED: say how the balance was read: --source api (the read-back above printed it) or --source console '
                         '--page PAGE --read-utc UTC (read by hand on the provider\'s own page; --screenshot-sha256 HEX if you kept '
                         'a screenshot). Nothing was recorded.')
            rec['source'] = args.source
            if args.source == 'api' and not (isinstance(rb.get('commands'), list) and rb.get('commands')):
                sys.exit(f'REFUSED: no read-back of {args.name} reads its balance by API with the run\'s key; it is read by hand on the '
                         'provider\'s own page: --source console --page PAGE --read-utc UTC. Nothing was recorded.')
            if args.source == 'console':
                page, read = (args.page or '').strip(), (args.read_utc or '').strip()
                if not page or not read:
                    sys.exit('REFUSED: a balance read by hand names the page it was read on and the time: --page PAGE --read-utc UTC. '
                             'Nothing was recorded.')
                if not UTC_FORM.match(read) or parse_utc(read) > now():
                    sys.exit('REFUSED: give --read-utc as the time the page was read, written as 2026-10-03T17:20:00Z and not later '
                             'than the clock. Nothing was recorded.')
                if read < cfg.get('blanks', {}).get('approved_public_utc', ''):
                    sys.exit('REFUSED: the page was read before the registration was public; read it again. Nothing was recorded.')
                rec['page'], rec['read_utc'] = page, read
                if args.screenshot_sha256 is not None:
                    if not re.fullmatch(r'[0-9a-f]{64}', args.screenshot_sha256.strip().lower()):
                        sys.exit('REFUSED: --screenshot-sha256 is the SHA-256 of the saved screenshot, 64 hexadecimal characters. '
                                 'Nothing was recorded.')
                    rec['screenshot_sha256'] = args.screenshot_sha256.strip().lower()
    rec['id'] = hashlib.sha256(json.dumps([args.name, args.check, rec], sort_keys=True).encode('utf-8')).hexdigest()[:12]
    cfg.setdefault('preflight', {}).setdefault(args.name, {})[args.check] = rec
    cfg.setdefault('preflight_history', []).append(dict(rec, provider=args.name, check=args.check))
    save(cfg)
    how = (f' via {rec["source"]}' + (f', page {rec["page"]} read at {rec["read_utc"]}' if rec.get('page') else '') +
           (f', screenshot SHA-256 {rec["screenshot_sha256"]}' if rec.get('screenshot_sha256') else '')) if rec.get('source') else ''
    detail = {'model': f' served={rec.get("served")}', 'balance': f' {rec.get("amount")} {rec.get("currency")}{how}'}.get(args.check, '')
    log(f'PREFLIGHT {rec["id"]} {args.name} {args.check} {args.result.upper()}{detail}' + (f' ({rec["note"]})' if rec['note'] else ''))
    if args.result == 'fail':
        print(f'PRE-FLIGHT FAILED for {args.name} {args.check}: the run does not start until this read-back passes. Fix the account, read back again and record the new result.')
    else:
        print(f'recorded: {args.name} {args.check} PASS (lab log id {rec["id"]})')
    ok, missing = preflight_state(cfg)
    print('Pre-flight: ' + ('COMPLETE.' if ok else f'not complete, {plural_items(len(missing))} left.'))


def cmd_done(args):
    cfg = load()
    sid = args.step
    ids = step_ids()
    if sid not in ids:
        sys.exit('no such step; steps are: ' + ' '.join(ids))
    step = step_by_id(sid)
    refuse_unless_sound(cfg, f'step {sid} is not recorded')
    if step.get('kind') == 'bench':
        sys.exit('use  bench.py prepare  for this step')
    if step.get('waits') and not registered(cfg):
        sys.exit('this step WAITS FOR REGISTRATION: record approved_public_utc first')
    i = ids.index(sid)
    rs = run_start_index()
    if rs is not None and i > rs and ids[rs] not in cfg['done']:
        log(f'{sid} refused: the run-start record (step {ids[rs]}) is not recorded')
        sys.exit(f'REFUSED: the run has not started. Step {ids[rs]} (the run-start record) is not recorded, and no step after it is recorded before it, in order or out of it.')
    if step.get('run_start'):
        ok, missing = preflight_state(cfg)
        if not ok:
            log(f'{sid} refused: the pre-flight is not complete')
            sys.exit('REFUSED: the run does not start until the pre-flight is complete:\n  ' + '\n  '.join(missing) + '\n  python3 bench/bench.py preflight')
        why = seal_problem(cfg)
        if why:
            log(f'{sid} refused: {why}')
            sys.exit(f'REFUSED: the run starts only on a folder you have recorded as reviewed, and {why}.\n  ' + SEAL_HINT)
    earlier = [s['id'] for s in steps()[:i] if s['id'] not in cfg['done'] and not s.get('optional')]
    if earlier and not args.out_of_order:
        sys.exit('earlier steps are not recorded as done: ' + ' '.join(earlier))
    if earlier and args.out_of_order:
        log(f'{sid} checked OUT OF ORDER, with earlier steps not recorded: ' + ' '.join(earlier))
    ok, facts, bad = check_step(step, cfg, args.exit)
    for k, v in facts.items():
        print(f'  {k}: {v}')
    if not ok:
        for b in bad:
            print('  NOT PASSED:', b)
        log(f'{sid} checked and NOT passed: ' + '; '.join(bad))
        sys.exit(1)
    cfg['done'][sid] = dict(facts, utc=stamp())
    save(cfg)
    log(f'{sid} done: ' + json.dumps(facts, sort_keys=True))
    print(f'PASS. Step {sid} recorded at {cfg["done"][sid]["utc"]}.')
    nxt = next_step(cfg)
    if nxt:
        print(f'Next: step {nxt}.  python3 bench/bench.py show')


def cmd_note(args):
    text = ' '.join(args.text).strip()
    if not text:
        sys.exit('give the words of the note')
    if looks_like_a_secret(text):
        sys.exit('REFUSED: the note holds a value shaped like a key or a token. Nothing was recorded.')
    load()
    log('note: ' + text)
    print('noted in lab/LAB-LOG.md')


def cmd_selfcheck(args):
    """Every command of a run-sheet step that the bench prints stands in RUN-SHEET.md as written (the pre-flight
    read-backs and the shell set-up lines are the bench's own, and the read-backs are linted); the registered hashes are the
    deposit's; the steps carry no TODO; the folder is its manifest."""
    if not os.path.exists(RUN_SHEET):
        sys.exit('kit/RUN-SHEET.md is not there')
    d = steps_doc()
    problems = []
    with open(RUN_SHEET, 'rb') as f:
        raw = f.read()
    if d.get('run_sheet_sha256') and hashlib.sha256(raw).hexdigest() != d['run_sheet_sha256']:
        problems.append('kit/RUN-SHEET.md is not the run sheet the steps were read from (its SHA-256 has changed): the steps may be stale')
    rs_lines = [l.rstrip() for l in raw.decode('utf-8', errors='replace').split('\n')]

    def stands(cmd):
        c = cmd.rstrip()
        return any(l == c or (l.startswith(c) and l[len(c):].lstrip().startswith('#')) for l in rs_lines)
    missing, n = [], 0
    for s in steps():
        if s.get('source', 'run-sheet') != 'run-sheet':
            continue
        for c in s['commands']:
            n += 1
            if not stands(c['run']):
                missing.append((s['id'], c['run']))
    print(f'commands checked against RUN-SHEET.md: {n}; not found as written: {len(missing)}')
    for sid, c in missing:
        print(f'  [{sid}] {c}')
    problems += gate_problems()
    problems += [f'the per-test cap: {c}' for c in cap_problems()]
    for s in steps():
        for c in s['pass_rule']:
            if c.get('check') not in KNOWN_CHECKS:
                problems.append(f'step {s["id"]} carries an unknown check: {c.get("check")}')
    keys = {p.get('key_env') for p in providers() if p.get('key_env')}
    for p in providers():
        nm = p.get('name')
        if not re.match(r'^[A-Z][A-Z0-9_]{1,63}$', str(p.get('key_env', ''))):
            problems.append(f'provider {nm}: key_env is not an environment variable name')
        if not p.get('model'):
            problems.append(f'provider {nm}: no model string')
        for ch in CHECKS:
            cmds = p.get('readback', {}).get(ch, {}).get('commands')
            for c in (cmds if isinstance(cmds, list) else []):
                for pr in lint_command(c, keys):
                    problems.append(f'provider {nm} {ch} read-back: {pr}: {c[:60]}')
    warn = 0
    for s in steps():
        for c in s['commands']:
            for pr in lint_command(c['run'], keys):
                warn += 1
                print(f'  warning [{s["id"]}]: {pr} (the run sheet is the authority; any difference goes in the defect log)')
    todos = todo_list()
    print(f'TODO markers left for a person: {len(todos)}')
    for t in todos:
        print('  ' + t)
    cur = current_hashes()
    cfg = config_or_defaults()
    bad_hash = 0
    for label, root, table in (('deposit', 'deposit', cur.get('registered', {})), ('stage', 'stage', cur.get('stage') or {})):
        base = root_dir(root, cfg)
        nbad = 0
        for name, h in sorted(table.items()):
            p = os.path.normpath(os.path.join(base, name))
            if not inside(base, p) or not os.path.isfile(p) or forbidden_read(p, cfg) or sha256(p) != h:
                nbad += 1
                print(f'  {label} file missing or not the registered bytes: {name}')
        if table:
            print(f'{label} files equal to the registered archive: {len(table) - nbad} of {len(table)}')
        bad_hash += nbad
    fd = cur.get('frozen_design') or {}
    if fd.get('name'):
        p = os.path.join(KIT, 'frozen-design', fd['name'])
        if not os.path.isfile(p) or forbidden_read(p, cfg) or sha256(p) != fd.get('sha256'):
            bad_hash += 1
            print('  the frozen design is missing or not the bytes the factory recorded')
    drift, _ = manifest_drift(cfg)
    print(f'FOLDER-MANIFEST.sha256: {"the folder matches it" if not drift else str(len(drift)) + " difference(s)"}')
    for x in drift:
        print('  ' + x)
    for x in problems:
        print('  PROBLEM: ' + x)
    if missing or problems or todos or bad_hash or drift:
        print('SELFCHECK FAILED')
        sys.exit(1)
    print('SELFCHECK PASS')


def cmd_manifest(args):
    """Compare the folder with FOLDER-MANIFEST.sha256, or seal it after a review. The bench never sees a review:
    the seal records the author's entry of who reviewed the folder and where the review lives, and says so."""
    cfg = load() if os.path.exists(CFG) else None
    drift, current = manifest_drift(cfg)
    if not args.write:
        for x in drift:
            print('  ' + x)
        if drift:
            print('The folder differs from FOLDER-MANIFEST.sha256. After a reviewed change:  ' + SEAL_HINT)
            sys.exit(1)
        print('FOLDER-MANIFEST.sha256: the folder matches it')
        seal = (cfg or {}).get('seal') or {}
        if cfg is not None and not seal_problem(cfg):
            print(f'Sealed by your entry of {seal.get("utc")}: reviewed by {seal.get("reviewed_by")}; the review is at {seal.get("review")} '
                  '(the bench did not see it).')
        else:
            print('No entry of yours records this folder as reviewed yet, and the run-start record is refused until one does:  ' + SEAL_HINT)
        return
    if cfg is None:
        sys.exit('REFUSED: run  python3 bench/bench.py init  first: your entry of who reviewed the folder is kept in lab/bench.json')
    who, where = (args.reviewed_by or '').strip(), (args.review or '').strip()
    if not who or not where:
        sys.exit('REFUSED: the bench cannot see a review, so it records your entry of one: give --reviewed-by NAME (who reviewed the '
                 'folder) and --review WHERE (the path or address of the review). Nothing was written.')
    if any(looks_like_a_secret(v) for v in (who, where)):
        sys.exit('REFUSED: a value is shaped like a key or a token. The bench never stores one. Nothing was written.')
    if run_started(cfg):
        sys.exit('REFUSED: the run has started; the bench and its kit are frozen, and the manifest is not rewritten')
    probs = gate_problems()
    if probs:
        sys.exit('REFUSED: kit/STEPS.json cannot hold the run at its pre-flight, so the folder is not sealed:\n  ' + '\n  '.join(probs))
    old = sha256(FOLDER_MANIFEST) if os.path.exists(FOLDER_MANIFEST) else 'none'
    if drift:
        with open(FOLDER_MANIFEST, 'w', encoding='utf-8', newline='\n') as f:
            f.write(current)
    new = sha256(FOLDER_MANIFEST)
    # The review's bytes are hashed only where the bench may read: never inside another unit's sealed run (judged from the top
    # down, links followed, before the file is touched), never a file named like a key, and only if the file is there.
    rp = resolve(where)
    unread = (('it lies inside another lab unit\'s unit/ or lab/ folder' if bad_folder_blank('DEPOSIT', rp) else '') or
              forbidden_read(rp, cfg) or ('' if os.path.isfile(rp) else 'there is no such file here'))
    review_sha = None if unread else sha256(rp)
    seal = {'utc': stamp(), 'manifest_sha256': new, 'reviewed_by': who, 'review': where, 'review_sha256': review_sha}
    seal['id'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    cfg['seal'] = seal
    cfg.setdefault('seal_history', []).append(seal)
    save(cfg)
    log(f'SEAL {seal["id"]} folder manifest {old[:12]} to {new[:12]}' + (': ' + '; '.join(drift[:6]) if drift else ', unchanged') +
        f'. The author\'s entry: reviewed by {who}; the review is at {where}' +
        (f' (SHA-256 {review_sha})' if review_sha else f' (not read by the bench: {unread})') +
        '. The bench did not see the review and does not certify it.')
    print('FOLDER-MANIFEST.sha256 ' + ('rewritten' if drift else 'unchanged') + f' ({new}), sealed with your entry: reviewed by {who}; '
          f'the review is at {where}. The bench did not see the review and does not certify it.')


def main(argv=None):
    _CACHE.clear()
    fmt = lambda prog: argparse.HelpFormatter(prog, width=WIDTH)   # a fixed width: argparse reads no terminal size from the environment
    ap = argparse.ArgumentParser(description='The lab bench. The run sheet is the authority.', formatter_class=fmt)
    ap.add_argument('--actor', default='', help='a name to put on this command\'s lab log lines (optional)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init', formatter_class=fmt); p.add_argument('--force', action='store_true'); p.set_defaults(f=cmd_init)
    p = sub.add_parser('set', formatter_class=fmt); p.add_argument('key'); p.add_argument('value'); p.set_defaults(f=cmd_set)
    p = sub.add_parser('blanks', formatter_class=fmt); p.set_defaults(f=cmd_blanks)
    p = sub.add_parser('status', formatter_class=fmt); p.set_defaults(f=cmd_status)
    p = sub.add_parser('show', formatter_class=fmt); p.add_argument('step', nargs='?'); p.set_defaults(f=cmd_show)
    p = sub.add_parser('prepare', formatter_class=fmt); p.set_defaults(f=cmd_prepare)
    p = sub.add_parser('preflight', formatter_class=fmt)
    p.add_argument('name', nargs='?'); p.add_argument('check', nargs='?'); p.add_argument('result', nargs='?')
    p.add_argument('--served'); p.add_argument('--amount', type=float); p.add_argument('--currency'); p.add_argument('--note')
    p.add_argument('--source', choices=('api', 'console')); p.add_argument('--page'); p.add_argument('--read-utc')
    p.add_argument('--screenshot-sha256')
    p.set_defaults(f=cmd_preflight)
    p = sub.add_parser('done', formatter_class=fmt); p.add_argument('step'); p.add_argument('--exit', type=int, nargs='+')
    p.add_argument('--out-of-order', action='store_true'); p.set_defaults(f=cmd_done)
    p = sub.add_parser('note', formatter_class=fmt); p.add_argument('text', nargs='+'); p.set_defaults(f=cmd_note)
    p = sub.add_parser('selfcheck', formatter_class=fmt); p.set_defaults(f=cmd_selfcheck)
    p = sub.add_parser('manifest', formatter_class=fmt); p.add_argument('--write', action='store_true')
    p.add_argument('--reviewed-by'); p.add_argument('--review'); p.set_defaults(f=cmd_manifest)
    args = ap.parse_args(argv)
    ACTOR[0] = (args.actor or '').strip()
    args.f(args)


if __name__ == '__main__':
    main()
