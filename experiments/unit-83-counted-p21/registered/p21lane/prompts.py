"""The prompt each role is sent (41-D-COUNTED, section 5 and Annex B.4).

Package E of the build list (DESIGN-FREEZE.md, section 12, revision 4.2.1). Every call carries the same system text, the paragraph and
the whole charter (charter.SYSTEM), and one user turn built here from the frozen task texts. An entry shown with its case
is the case paragraph and then `Entry: {line}`; the arriving draft in the corrector's turn is `Draft entry: {line}`;
lists of entries are separated by one blank line (Annex B.3). Nothing is marked as new in a monitor's view, and nothing
tells an engine what follows from its reply. Nothing here is executed before both registrations read REGISTERED
(SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

A view is a list of (case, entry) pairs. "In order of code" is the order of the codes as text (BUILD-DECISIONS.md): any
fixed order keeps the arriving entry's place uniform, because codes are drawn by seed, and one order serves the monitor,
the twin and the windowed monitor alike, so that the twin's prompt equals the monitor's byte for byte wherever the two
views hold the same entries.
"""
import re

from . import charter, forms

_SLOT = re.compile(r'\{(kept|case|draft|entries|written|code)\}')


def _fill(template, **values):
    """Fill each slot of a frozen task text once, in one pass, so that no filled text is read as a slot."""
    slots = _SLOT.findall(template)
    if sorted(slots) != sorted(values) or len(set(slots)) != len(slots):
        raise ValueError('the task text and the values do not name the same slots, once each')
    return _SLOT.sub(lambda m: values[m.group(1)], template)


def _listed(pairs):
    return '\n\n'.join(forms.shown(case, entry) for case, entry in pairs)


def code_order(view):
    """The view in order of code. Codes are unique within a pair, so the order is total."""
    codes = [entry.code for _, entry in view]
    if len(set(codes)) != len(codes):
        raise ValueError('two entries in one view share a code')
    return sorted(view, key=lambda pair: pair[1].code)


def corrector(kept, case, draft):
    """The in-loop arm (section 5.2): the kept entries in the order kept, then the arriving case and its draft."""
    user = _fill(charter.TASK_CORRECTOR, kept=_listed(kept) if kept else charter.NONE_KEPT,
                 case=forms.case_text(case), draft=forms.line(draft))
    return charter.SYSTEM, user


def monitor(view):
    """The outside arm (section 5.3) and the twin (section 5.4): every entry in view, in order of code."""
    if not view:
        raise ValueError('a monitor view holds at least the arriving draft')
    return charter.SYSTEM, _fill(charter.TASK_MONITOR, entries=_listed(code_order(view)))


def window(view):
    """The windowed monitor (section 5.5): the monitor's prompt on the first three entries, in order of code, of its
    own finished register."""
    return monitor(code_order(view)[:3])


def informed(view, code):
    """The informed monitor (section 5.7a): the monitor's prompt on its own finished register, in order of code with
    nothing marked in the list, with the one line of Annex B.4 before TASK naming the arriving case's code."""
    if not view:
        raise ValueError('an informed view holds at least the arriving draft')
    if code not in [entry.code for _, entry in view]:
        raise ValueError('the arriving code is not in the informed view')
    return charter.SYSTEM, _fill(charter.TASK_INFORMED, entries=_listed(code_order(view)), code=code)


def twin_view(view, planted_codes, required):
    """The twin's view (section 5.4): the outside arm's view with every older accepted entry that was planted replaced
    by its case's required entry; the arriving draft (the last pair of the view) unchanged. `planted_codes` names the
    older entries that were planted; `required(case)` gives a case's required entry."""
    if not view:
        raise ValueError('a twin view holds at least the arriving draft')
    older, arriving = view[:-1], view[-1]
    return [(case, required(case)) if entry.code in planted_codes else (case, entry) for case, entry in older] + [arriving]


def transport():
    """Admission's transport check: the base's transport text, unchanged (Annex B.4, "Admission")."""
    return charter.TRANSPORT_SYSTEM, charter.TRANSPORT_USER


def generating(kept, case):
    """The generating arm (section 5.6): the entries written so far in the order written, then the arriving case."""
    user = _fill(charter.TASK_GENERATE, written=_listed(kept) if kept else charter.NONE_WRITTEN,
                 case=forms.case_text(case))
    return charter.SYSTEM, user
