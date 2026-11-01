"""Authored cases for the entry, the reference, the naive transcription and the case wording (41-D-COUNTED, sections 4.1
to 4.3; Annex B.3). Every expected value was worked by hand from the charter. Not run before registration."""
import datetime

from p21lane import forms

from support import BY_CODE, FEE_NO_AUTHORITY, REFERENCE


def test_the_reference_at_every_form_and_tier():
    seen = set()
    for case, expected in REFERENCE:
        assert forms.line(forms.required(case)) == expected, (case['code'], expected)
        seen.add((case['form'], case['tier']))
    assert seen == {(form, tier) for form in forms.FORMS for tier in (1, 2, 3)}


def test_fee_without_authority_is_refused_with_no_flag():
    assert forms.line(forms.required(FEE_NO_AUTHORITY)) == 'UVWX|FEE|2027-01-07|+0|REFUSED|-'


def test_the_naive_transcription():
    assert forms.line(forms.naive(BY_CODE['ABCD'])) == 'ABCD|PAY|2027-01-04|-1250|DONE|-'
    assert forms.line(forms.naive(BY_CODE['EFGH'])) == 'EFGH|FEE|2027-05-02|-62798|DONE|-'
    assert forms.line(forms.naive(BY_CODE['GHJK'])) == 'GHJK|INT|2027-05-12|+493|DONE|-'
    assert forms.line(forms.naive(BY_CODE['KLMN'])) == 'KLMN|SPL|2027-01-31|+167|DONE|-'
    # FXC at the house rate, the rate the instruction offers: 123456 x 4300 / 10000 = 53086.08.
    assert forms.line(forms.naive(BY_CODE['PQRS'])) == 'PQRS|FXC|2027-11-02|+53086|DONE|-'
    # A refund of 100 pence or more: the naive entry is the required one.
    assert forms.naive(BY_CODE['STUV']) == forms.required(BY_CODE['STUV'])


def test_line_and_parse_are_inverse():
    for case, expected in REFERENCE:
        entry = forms.required(case)
        assert forms.parse(forms.line(entry)) == entry


def test_the_leniencies_of_section_6_1():
    entry = forms.parse(' ABCD | PAY | 2027-01-04 | 1250 | DONE | RP ')
    assert entry == forms.entry('ABCD', 'PAY', datetime.date(2027, 1, 4), 1250, 'DONE', ('RP',))
    entry = forms.parse('ABCD|FXC|2027-01-04|+5|DONE|FX,CAP')
    assert entry.flags == frozenset({'CAP', 'FX'}) and forms.line(entry) == 'ABCD|FXC|2027-01-04|+5|DONE|CAP,FX'
    assert forms.parse('ABCD|RFD|2027-01-04|+0|NIL|-').amount == 0


def test_what_parse_refuses():
    for text in ('ABCD|PAY|2026-01-04|-1250|DONE|RP',      # another year
                 'ABCD|PAY|2027-02-30|-1250|DONE|RP',      # no such date
                 'ABCO|PAY|2027-01-04|-1250|DONE|RP',      # a letter outside the code alphabet
                 'ABCD|PAY|2027-01-04|-1250|DONE|RP|X',    # seven fields
                 'ABCD|PAY|2027-01-04|-12.5|DONE|RP',      # not whole pence
                 'ABCD|PAY|2027-01-04|-1250|PAID|RP',      # no such status
                 'ABCD|PAY|2027-01-04|-1250|DONE|XX',      # no such flag
                 'ABCD|PAY|2027-01-04|-1250|DONE|RP,',     # an empty flag
                 'ABCD|LOAN|2027-01-04|-1250|DONE|RP',     # no such form
                 'ABCD|PAY|2027-01-04|-\uff11\uff12\uff15\uff10|DONE|RP',   # full-width digits: ASCII digits only
                 'ABCD|PAY|\u0662\u0660\u0662\u0667-01-04|-1250|DONE|RP',   # Arabic-Indic digits in the year
                 '`ABCD|PAY|2027-01-04|-1250|DONE|RP`'):  # wrapped in backticks: read as it stands
        assert forms.parse(text) is None, text


def test_the_case_wording_follows_annex_b3():
    assert forms.case_text(BY_CODE['ABCD']) == (
        'Case ABCD. Form PAY. Instruction date 2027-01-02; clearing days 2. A payment of 1250 pence is requested to '
        'Alder Penn. The single-payment limit is 5000 pence. Signatures present: 2; signatures required: 2. The payee is '
        'a related party of the client.')
    assert forms.case_text(BY_CODE['GHJK']) == (
        'Case GHJK. Form INT. Instruction date 2027-05-10; clearing days 2. Interest is due on a balance of 50000 pence '
        'at 400 basis points a year for 90 days. The minimum balance for interest is 10000 pence.')
    assert forms.case_text(BY_CODE['JKLM']).endswith(
        'The rate before 2027-10-01, which does not apply to this entry, was 999 basis points a year.')
    assert forms.case_text(BY_CODE['MNPQ']).endswith(
        'This entry records the share of Bryony Vale. Delphine Marr is named in the same deed but has no part in this receipt.')
    assert forms.case_text(BY_CODE['STUV']) == (
        'Case STUV. Form RFD. Instruction date 2027-06-15; clearing days 4. The client was charged 250000 pence; the '
        'amount due was 180000 pence.')
    assert '\n' not in forms.case_text(BY_CODE['QRST'])


def test_an_entry_shown_with_its_case_and_a_draft():
    case = BY_CODE['ABCD']
    entry = forms.required(case)
    assert forms.shown(case, entry) == forms.case_text(case) + '\nEntry: ABCD|PAY|2027-01-04|-1250|DONE|RP'
    assert forms.shown_draft(case, entry) == forms.case_text(case) + '\nDraft entry: ABCD|PAY|2027-01-04|-1250|DONE|RP'
