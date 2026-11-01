"""Scoring, blind by construction (41-D-COUNTED, sections 4.1 and 6.4).

Package G of the build list (DESIGN-FREEZE.md, section 12). No person and no model scores anything. `table(root)`
gathers the distinct pairs of a case key and an entry line from every logged round (each round's draft, the corrector's
kept entry and the generating arm's written entry), keyed by the SHA-256 of the two, and writes them with nothing else:
no arm, block, round or engine, and the case key carries no label. `run(root)` reads only that table and the sealed
answers (case key to required line, written at `init`), compares each entry with its required entry, and writes score
records {'key', 'case_key', 'sound', 'fields_differing'}. One entry text is scored once however many arms produced it;
the map from keys to cases stays with the analysis. Repeatable; it reads no provider. `compare` is the one pure
function that every stage uses, the calibration stage's single-entry answers among them. Nothing here is executed before
both registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.
"""
import json
from pathlib import Path

from p21base.common import GateError, load, save

from . import collect, forms, seal

FIELDS = forms.Entry._fields


def compare(line, required_line):
    """Whether an entry line equals the required entry, as six parsed fields (code and form as text, the date as a
    date, the amount as an integer, the status as text, the flags as a set), and which fields differ."""
    entry, required = forms.parse(line), forms.parse(required_line)
    if required is None:
        raise GateError('a required line does not parse')
    if entry is None:
        return False, ['unparsed']
    differing = [name for name in FIELDS if getattr(entry, name) != getattr(required, name)]
    return not differing, differing


def table(root, config, tier, engines, n_pairs):
    """The scoring table for pairs 1 to n_pairs of each engine: [(key, case_key, entry_line)], distinct, sorted by key."""
    root = Path(root)
    seed = seal.bank_seed(root, config)
    rows = {}
    for engine in engines:
        for log in collect.pair_logs(root, engine, n_pairs):
            for block in log['blocks']:
                for rnd in block['rounds']:
                    case_key = seal.case_key(rnd['case_id'], seed)
                    for line in (rnd['draft'], rnd['kept'].get('C'), rnd['kept'].get('G')):
                        if line:
                            rows[seal.entry_key(case_key, line)] = (case_key, line)
    out = [(key, case_key, line) for key, (case_key, line) in sorted(rows.items())]
    save(root / 'scores' / 'table.json', [{'key': k, 'case_key': c, 'entry_line': l} for k, c, l in out])
    return out


def run(root):
    """Score the table against the sealed answers; writes and returns {key: score record}."""
    root = Path(root)
    rows = load(root / 'scores' / 'table.json')
    answers = {}
    for path in sorted((root / 'answers').glob('tier*.json')):
        answers.update(json.loads(path.read_text('utf-8')))
    records = {}
    for row in rows:
        required = answers.get(row['case_key'])
        if required is None:
            raise GateError('a case key has no sealed answer')
        if seal.entry_key(row['case_key'], row['entry_line']) != row['key']:
            raise GateError('a scoring row\'s key does not match its case key and line')
        sound, differing = compare(row['entry_line'], required)
        records[row['key']] = {'key': row['key'], 'case_key': row['case_key'], 'sound': sound, 'fields_differing': differing}
    save(root / 'scores' / 'score_records.json', records)
    return records
