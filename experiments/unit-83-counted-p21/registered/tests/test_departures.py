"""Authored cases for the planted departures (41-D-COUNTED, section 4.4): every operator on a case worked by hand, the
admission rule and T-DIGIT's rule. Not run before both registrations read REGISTERED."""
from p21lane import departures, forms

from support import BY_CODE, FEE_FITS_FRACTION, FEE_NO_AUTHORITY, FXC_TIE


def produced(name, case):
    return forms.line(departures.FUNCTIONS[name](case, forms.required(case)))


def test_the_operators_of_kind_s():
    assert produced('S-DATE', BY_CODE['ABCD']) == 'ABCD|PAY|2027-01-02|-1250|DONE|RP'
    assert produced('S-RP', BY_CODE['ABCD']) == 'ABCD|PAY|2027-01-04|-1250|DONE|-'
    assert produced('S-LIMIT', BY_CODE['BCDE']) == 'BCDE|PAY|2027-04-02|-123456|DONE|-'
    assert produced('S-SIGS', BY_CODE['CDEF']) == 'CDEF|PAY|2027-07-04|-98765432|DONE|RP'
    assert produced('S-AUTH', FEE_NO_AUTHORITY) == 'UVWX|FEE|2027-01-07|-24000|DONE|-'
    assert produced('S-CAP', BY_CODE['EFGH']) == 'EFGH|FEE|2027-05-02|-62798|DONE|-'
    assert produced('S-ROUND', FEE_FITS_FRACTION) == 'VWXY|FEE|2027-01-07|-62799|DONE|-'
    assert produced('S-ROUND', BY_CODE['GHJK']) == 'GHJK|INT|2027-05-12|+493|DONE|-'
    assert produced('S-ROUND', BY_CODE['PQRS']) == 'PQRS|FXC|2027-11-02|+56382|DONE|FX'
    assert produced('S-MIN', BY_CODE['HJKL']) == 'HJKL|INT|2027-09-04|+5294|DONE|-'
    assert produced('S-REM', BY_CODE['KLMN']) == 'KLMN|SPL|2027-01-31|+166|DONE|-'
    assert produced('S-NEAR', BY_CODE['MNPQ']) == 'MNPQ|SPL|2027-09-30|+4830918|DONE|-'
    assert produced('S-HOUSE', BY_CODE['PQRS']) == 'PQRS|FXC|2027-11-02|+53087|DONE|FX'
    assert produced('S-FX', BY_CODE['NPQR']) == 'NPQR|FXC|2027-03-05|+60000|DONE|-'
    assert produced('S-KEEP', BY_CODE['RSTU']) == 'RSTU|RFD|2027-04-02|+0|NIL|-'
    assert produced('S-PART', BY_CODE['STUV']) == 'STUV|RFD|2027-06-19|+69900|DONE|-'


def test_where_an_operator_does_not_apply():
    req = forms.required
    assert departures.s_limit(BY_CODE['ABCD'], req(BY_CODE['ABCD'])) is None          # r within the limit
    assert departures.s_sigs(BY_CODE['BCDE'], req(BY_CODE['BCDE'])) is None           # signatures present
    assert departures.s_round(BY_CODE['EFGH'], req(BY_CODE['EFGH'])) is None          # the fee due exceeds the room
    assert departures.s_round(BY_CODE['NPQR'], req(BY_CODE['NPQR'])) is None          # no fraction of a penny
    assert departures.s_min(BY_CODE['GHJK'], req(BY_CODE['GHJK'])) is None            # the minimum does not bind
    assert departures.s_rem(BY_CODE['LMNP'], req(BY_CODE['LMNP'])) is None            # not the first-named
    assert departures.s_near(BY_CODE['LMNP'], req(BY_CODE['LMNP'])) is None           # the share is exact
    assert departures.s_keep(BY_CODE['STUV'], req(BY_CODE['STUV'])) is None           # a refund of 100 pence or more
    assert departures.s_part(BY_CODE['RSTU'], req(BY_CODE['RSTU'])) is None           # a refund under 100 pence
    assert departures.t_sign(BY_CODE['BCDE'], req(BY_CODE['BCDE'])) is None           # +0 has no sign to flip


def test_the_operators_of_kind_t():
    assert produced('T-SIGN', BY_CODE['ABCD']) == 'ABCD|PAY|2027-01-04|+1250|DONE|RP'
    assert produced('T-DAY', BY_CODE['ABCD']) == 'ABCD|PAY|2027-01-05|-1250|DONE|RP'
    assert produced('T-DIGIT', BY_CODE['ABCD']) == 'ABCD|PAY|2027-01-04|-2150|DONE|RP'


def test_t_digit_rule():
    assert departures.digit_exchange(1250) == 2150
    assert departures.digit_exchange(-1250) == -2150
    assert departures.digit_exchange(1001) == 1010      # 1 and 0 would put a zero first; 0 and 0 are equal
    assert departures.digit_exchange(105) == 150
    assert departures.digit_exchange(100) is None
    assert departures.digit_exchange(55) is None
    assert departures.digit_exchange(7) is None
    assert departures.digit_exchange(0) is None


def test_the_admitted_entries_of_a_case():
    names = [name for name, _, _ in departures.planted(BY_CODE['ABCD'])]
    assert names == ['S-DATE', 'S-RP', 'T-SIGN', 'T-DAY', 'T-DIGIT']
    names = [name for name, _, _ in departures.planted(BY_CODE['BCDE'])]
    assert names == ['S-DATE', 'S-LIMIT', 'T-DAY']
    for case in (BY_CODE['ABCD'], BY_CODE['BCDE'], BY_CODE['KLMN'], BY_CODE['QRST']):
        req = forms.required(case)
        entries = [e for _, _, e in departures.planted(case)]
        assert req not in entries and len(set(entries)) == len(entries)


def test_two_operators_giving_one_entry_are_both_left_out():
    # floor(12345 x 10007 / 10000) = 12353 = ceil(12345 x 10006 / 10000): S-ROUND and S-HOUSE give one entry.
    assert produced('S-ROUND', FXC_TIE) == produced('S-HOUSE', FXC_TIE) == 'WXYZ|FXC|2027-01-07|+12353|DONE|FX'
    names = [name for name, _, _ in departures.planted(FXC_TIE)]
    assert names == ['S-DATE', 'S-FX', 'T-SIGN', 'T-DAY', 'T-DIGIT']


def test_the_two_registered_classifications():
    assert sorted(n for n, c in departures.CLASS.items() if c == 'A') == sorted(
        ['S-DATE', 'S-ROUND', 'S-CAP', 'S-REM', 'S-NEAR', 'S-HOUSE', 'S-PART'])
    assert sorted(n for n, c in departures.CLASS.items() if c == 'B') == sorted(
        ['S-LIMIT', 'S-SIGS', 'S-RP', 'S-AUTH', 'S-MIN', 'S-FX', 'S-KEEP'])
    assert sorted(n for n, c in departures.CLASS.items() if c == 'T') == ['T-DAY', 'T-DIGIT', 'T-SIGN']
    assert sum(1 for c in departures.CATCH.values() if c == 'one_comparison') == 9
    assert sum(1 for c in departures.CATCH.values() if c == 'recomputation') == 8
    assert len(departures.OPERATORS) == 17 and set(departures.FUNCTIONS) == set(departures.KIND)
