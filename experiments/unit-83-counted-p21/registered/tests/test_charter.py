"""Authored case: the frozen wording equals Annex B of the design freeze byte for byte (41-D-COUNTED, Annex B.1, B.2 and
B.4). The freeze is read from the file P21LANE_FREEZE_FILE names (the deposit's copy of DESIGN-FREEZE.md), whose
SHA-256 must be the configuration's. Not run before both registrations read REGISTERED."""
import hashlib
import os
import re
from pathlib import Path

from p21lane import charter

from support import load_config


def _fences(text, heading):
    start = text.index(heading)
    return re.findall(r'```\n(.*?)\n```', text[start:], flags=re.S)


def test_the_strings_equal_annex_b():
    path = Path(os.environ['P21LANE_FREEZE_FILE'])
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == load_config()['design_freeze']['sha256']
    text = data.decode('utf-8')
    b1 = _fences(text, '### B.1 The system text')[0]
    b2 = _fences(text, '### B.2 The charter')[0]
    corrector, monitor, informed, generating = _fences(text, '### B.4 The task texts')[:4]
    assert charter.PARAGRAPH == b1 and charter.CHARTER == b2 and charter.SYSTEM == b1 + '\n\n' + b2
    fills = {'{kept}': '{kept entries, each with its case; or, at the first round, the line: No entries have been kept yet.}',
             '{case}': '{the arriving case}', '{draft}': '{the draft line}',
             '{entries}': '{the entries in view, each with its case, in order of code}',
             '{written}': '{the entries this arm has written, each with its case; or, at the first round, the line: No entries have been written yet.}'}

    def shown(template):
        for slot, words in fills.items():
            template = template.replace(slot, words)
        return template

    assert shown(charter.TASK_CORRECTOR) == corrector
    assert shown(charter.TASK_MONITOR) == monitor
    assert shown(charter.TASK_INFORMED) == informed
    assert shown(charter.TASK_GENERATE) == generating
