import pytest

from app.parser import (
    AdductError,
    apply_adduct,
    parse_adduct,
    parse_formula,
)


def _signed(terms):
    return [(t.sign, t.coefficient, t.fragment, t.start) for t in terms]


def test_single_positive_fragment():
    assert _signed(parse_adduct("+H")) == [(1, 1, {"H": 1}, 0)]


def test_coefficient_multiplies_whole_fragment():
    terms = parse_adduct("+2H2O")
    assert _signed(terms) == [(1, 2, {"H": 2, "O": 1}, 0)]


def test_multi_fragment_adduct():
    terms = parse_adduct("+2H-H2O")
    assert _signed(terms) == [
        (1, 2, {"H": 1}, 0),
        (-1, 1, {"H": 2, "O": 1}, 3),
    ]


def test_sodium_adduct():
    terms = parse_adduct("+Na-H")
    assert _signed(terms) == [
        (1, 1, {"Na": 1}, 0),
        (-1, 1, {"H": 1}, 3),
    ]


def test_repeated_elements_in_fragment_accumulate():
    terms = parse_adduct("+HCl")
    assert terms[0].fragment == {"H": 1, "Cl": 1}


def test_must_start_with_sign():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("Na-H")
    assert excinfo.value.position == 0


def test_empty_adduct():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("")
    assert excinfo.value.position == 0


def test_whitespace_rejected():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+ H")
    assert excinfo.value.position == 1


def test_zero_coefficient_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+0H")
    assert excinfo.value.position == 1
    assert "positive integer" in excinfo.value.message


def test_zero_count_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+H0")
    assert excinfo.value.position == 2


def test_fragment_without_element_reports_sign():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+-H")
    assert excinfo.value.position == 0
    with pytest.raises(AdductError):
        parse_adduct("+2")


def test_dangling_sign_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+H-")
    assert excinfo.value.position == 2


def test_unknown_element_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+Xx")
    assert excinfo.value.position == 1
    assert "Xx" in excinfo.value.message


def test_stray_digit_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+2-H")
    assert excinfo.value.position == 1


def test_label_in_adduct_rejected():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+[13C]")
    assert excinfo.value.position == 1


def test_apply_adduct_adds_and_removes():
    parsed = parse_formula("C6H12O6")
    composition = apply_adduct(parsed, parse_adduct("+2H-H2O"))
    assert composition == {"C": 6, "H": 12, "O": 5}


def test_apply_adduct_protonation():
    parsed = parse_formula("C6H12O6")
    assert apply_adduct(parsed, parse_adduct("+H")) == {"C": 6, "H": 13, "O": 6}


def test_apply_adduct_new_element():
    parsed = parse_formula("CH3")
    assert apply_adduct(parsed, parse_adduct("+Na-H")) == {"C": 1, "H": 2, "Na": 1}


def test_apply_adduct_zero_count_element_dropped():
    parsed = parse_formula("H2O")
    assert apply_adduct(parsed, parse_adduct("-H2")) == {"O": 1}


def test_apply_adduct_negative_count_reports_fragment_start():
    parsed = parse_formula("C6H12O6")
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parsed, parse_adduct("+H-H20"))
    # the '-' fragment at index 2 drives H negative (13 - 20)
    assert excinfo.value.position == 2
    assert "H" in excinfo.value.message


def test_apply_adduct_coefficient_negative_reports_sign():
    parsed = parse_formula("H2O")
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parsed, parse_adduct("-2H2O"))
    assert excinfo.value.position == 0


def test_apply_adduct_first_negative_wins():
    # 'CH' with '+Na-C2-H': the '-C2' fragment (sign at index 3) drives C
    # negative (1 - 2) before the later '-H' term is considered
    parsed = parse_formula("CH")
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parsed, parse_adduct("+Na-C2-H"))
    assert excinfo.value.position == 3


def test_apply_adduct_leaves_labels_untouched():
    parsed = parse_formula("C6H6[13C]2")
    composition = apply_adduct(parsed, parse_adduct("+H"))
    assert composition == {"C": 6, "H": 7}
    assert parsed.labels == {("C", 13): 2}
