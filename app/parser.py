"""Molecular-formula parser with position-aware validation errors.

Grammar (case-sensitive)::

    formula := term+
    term    := element count? | label count?
    element := [A-Z] [a-z]?
    label   := "[" digit+ element "]"      e.g. [13C], [15N], [2H]
    count   := digit+                      positive integer, default 1

Examples: ``C6H12O6``, ``C6H6[13C]2``, ``C2H5[15N]O[18O]``.

A label ``[13C]2`` contributes two atoms of *pure* 13C in addition to any
natural-abundance ``C`` atoms listed elsewhere in the formula.

Adducts (used by the ion-envelope endpoint) are a separate grammar::

    adduct    := term+
    term      := sign coefficient? fragment
    sign      := "+" | "-"
    coefficient := digit+              positive integer, default 1
    fragment  := element count? +      plain elements only, no labels

Examples: ``+H``, ``+2H-H2O``, ``+Na-H``. The coefficient multiplies the
whole following fragment (``+2H2O`` adds 4 H and 2 O). The whole string must
be consumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .isotopes import ISOTOPES, isotope_by_mass_number

_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_LABEL_BODY_RE = re.compile(r"(\d+)([A-Z][a-z]?)")


class FormulaError(ValueError):
    """Validation error carrying the 0-based character position in the formula."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


class LabelError(ValueError):
    """Validation error for a bare isotope label, carrying the 0-based
    character position within the label string."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


class AdductError(ValueError):
    """Validation error for an adduct string, carrying the 0-based character
    position within the adduct (pointing at the first character of the
    offending fragment when applying it would drive an element count below
    zero)."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


@dataclass
class AdductTerm:
    """One signed adduct fragment: ``sign`` is +1 or -1, ``coefficient``
    multiplies every atom count in ``fragment``, and ``start`` is the 0-based
    position of the term's sign character."""

    sign: int
    coefficient: int
    fragment: dict[str, int]
    start: int



@dataclass
class ParsedFormula:
    # natural-abundance element symbol -> atom count
    composition: dict[str, int] = field(default_factory=dict)
    # (element symbol, mass number) -> atom count, for pure-isotope labels
    labels: dict[tuple[str, int], int] = field(default_factory=dict)


def _parse_count(text: str, i: int) -> tuple[int, int]:
    """Parse an optional count starting at ``i``; returns (count, next_index)."""
    j = i
    while j < len(text) and text[j].isdigit():
        j += 1
    if j == i:
        return 1, i
    count = int(text[i:j])
    if count < 1:
        raise FormulaError(
            f"Element count must be a positive integer, got {count}", i
        )
    return count, j


def parse_formula(text: str) -> ParsedFormula:
    """Parse and validate a molecular formula.

    Raises :class:`FormulaError` with a 0-based ``position`` pointing at the
    offending character/token for any malformed input.
    """
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("Formula must be a non-empty string", 0)
    if text != text.strip():
        pos = len(text) - len(text.lstrip()) if text != text.lstrip() else len(text.rstrip())
        raise FormulaError("Formula must not contain whitespace", pos)

    parsed = ParsedFormula()
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "[":
            i = _parse_label(text, i, parsed)
        elif ch.isupper():
            match = _ELEMENT_RE.match(text, i)
            symbol = match.group(0)
            if symbol not in ISOTOPES:
                raise FormulaError(f"Unknown element symbol '{symbol}'", i)
            count, i = _parse_count(text, match.end())
            parsed.composition[symbol] = parsed.composition.get(symbol, 0) + count
        elif ch.islower():
            raise FormulaError(
                f"Element symbols must start with an uppercase letter; "
                f"unexpected '{ch}'",
                i,
            )
        elif ch.isdigit():
            raise FormulaError(f"Unexpected count without a preceding element", i)
        else:
            raise FormulaError(f"Unexpected character '{ch}'", i)

    if not parsed.composition and not parsed.labels:
        raise FormulaError("Formula contains no elements", 0)
    return parsed


def _parse_label(text: str, i: int, parsed: ParsedFormula) -> int:
    """Parse ``[massNumberElement]count`` starting at ``i`` (the '[')."""
    close = text.find("]", i + 1)
    if close == -1:
        raise FormulaError("Unterminated isotope label: missing ']'", i)
    body = text[i + 1 : close]
    match = _LABEL_BODY_RE.fullmatch(body)
    if not match:
        raise FormulaError(
            f"Malformed isotope label '[{body}]'; expected e.g. '[13C]'",
            i,
        )
    mass_number = int(match.group(1))
    symbol = match.group(2)
    symbol_pos = i + 1 + len(match.group(1))
    if symbol not in ISOTOPES:
        raise FormulaError(f"Unknown element symbol '{symbol}' in label", symbol_pos)
    if isotope_by_mass_number(symbol, mass_number) is None:
        raise FormulaError(
            f"Element '{symbol}' has no known isotope with mass number "
            f"{mass_number}; check the label mass",
            i,
        )
    count, next_i = _parse_count(text, close + 1)
    key = (symbol, mass_number)
    parsed.labels[key] = parsed.labels.get(key, 0) + count
    return next_i


def parse_label(text: str) -> tuple[str, int]:
    """Parse and validate a bare isotope label such as ``'13C'``.

    Returns ``(element_symbol, mass_number)``. Raises :class:`LabelError`
    with a 0-based ``position`` pointing at the offending token.
    """
    if not isinstance(text, str) or not text.strip():
        raise LabelError("Label must be a non-empty string, e.g. '13C'", 0)
    match = _LABEL_BODY_RE.fullmatch(text)
    if not match:
        raise LabelError(
            f"Malformed label '{text}'; expected a mass number followed by "
            f"an element symbol, e.g. '13C'",
            0,
        )
    mass_number = int(match.group(1))
    symbol = match.group(2)
    symbol_pos = len(match.group(1))
    if symbol not in ISOTOPES:
        raise LabelError(f"Unknown element symbol '{symbol}' in label", symbol_pos)
    if isotope_by_mass_number(symbol, mass_number) is None:
        raise LabelError(
            f"Element '{symbol}' has no known isotope with mass number "
            f"{mass_number}; check the label mass",
            0,
        )
    return symbol, mass_number


def _parse_adduct_count(text: str, i: int) -> tuple[int, int]:
    """Parse a required positive count starting at ``i`` in an adduct
    fragment; returns (count, next_index). Raises :class:`AdductError` for a
    zero count."""
    j = i
    while j < len(text) and text[j].isdigit():
        j += 1
    count = int(text[i:j])
    if count < 1:
        raise AdductError(
            f"Element count must be a positive integer, got {count}", i
        )
    return count, j


def _parse_adduct_fragment(
    text: str, i: int, start: int
) -> tuple[dict[str, int], int]:
    """Parse the element fragment of an adduct term starting at ``i`` (just
    past the sign/coefficient) through the next sign or end of string."""
    n = len(text)
    fragment: dict[str, int] = {}
    saw_element = False
    while i < n and text[i] not in "+-":
        ch = text[i]
        if ch == "[":
            raise AdductError(
                "Isotope labels are not supported in adduct fragments; "
                "use plain element symbols",
                i,
            )
        if ch.isupper():
            match = _ELEMENT_RE.match(text, i)
            symbol = match.group(0)
            if symbol not in ISOTOPES:
                raise AdductError(f"Unknown element symbol '{symbol}'", i)
            i = match.end()
            if i < n and text[i].isdigit():
                count, i = _parse_adduct_count(text, i)
            else:
                count = 1
            fragment[symbol] = fragment.get(symbol, 0) + count
            saw_element = True
        elif ch.islower():
            raise AdductError(
                f"Element symbols must start with an uppercase letter; "
                f"unexpected '{ch}'",
                i,
            )
        elif ch.isdigit():
            raise AdductError(
                "Unexpected count without a preceding element", i
            )
        else:
            raise AdductError(f"Unexpected character '{ch}'", i)

    if not saw_element:
        raise AdductError(
            "Adduct fragment must contain at least one element", start
        )
    return fragment, i


def parse_adduct(text: str) -> list[AdductTerm]:
    """Parse an adduct specification such as ``'+2H-H2O'`` or ``'+Na-H'``.

    The string must consist of one or more signed element fragments; the
    optional integer coefficient right after a sign multiplies the whole
    following fragment. Raises :class:`AdductError` with a 0-based
    ``position`` pointing at the offending character.
    """
    if not isinstance(text, str) or not text:
        raise AdductError(
            "Adduct must be a non-empty string, e.g. '+H' or '+Na-H'", 0
        )
    if text != text.strip():
        pos = len(text) - len(text.lstrip()) if text != text.lstrip() else len(text.rstrip())
        raise AdductError("Adduct must not contain whitespace", pos)

    terms: list[AdductTerm] = []
    i, n = 0, len(text)
    while i < n:
        start = i
        ch = text[i]
        if ch not in "+-":
            raise AdductError(
                "Adduct must start each fragment with '+' or '-', e.g. "
                "'+2H-H2O'",
                i,
            )
        sign = 1 if ch == "+" else -1
        i += 1

        # Optional coefficient: digits immediately after the sign are a
        # coefficient when followed by an element symbol, otherwise they
        # belong to (or malform) the fragment itself.
        coefficient = 1
        j = i
        while j < n and text[j].isdigit():
            j += 1
        if j > i and j < n and text[j].isalpha() and text[j].isupper():
            coefficient = int(text[i:j])
            if coefficient < 1:
                raise AdductError(
                    f"Fragment coefficient must be a positive integer, got "
                    f"{coefficient}",
                    i,
                )
            i = j

        fragment, i = _parse_adduct_fragment(text, i, start)
        terms.append(
            AdductTerm(
                sign=sign, coefficient=coefficient, fragment=fragment, start=start
            )
        )
    return terms


def apply_adduct(
    parsed: ParsedFormula, terms: list[AdductTerm]
) -> dict[str, int]:
    """Return the natural-abundance composition after applying adduct terms.

    Terms are applied in order; if adding/removing a term would drive any
    element count negative, :class:`AdductError` is raised with ``position``
    at the first character (the sign) of the offending fragment. Zero-count
    elements are dropped from the result; pure-isotope labels are untouched.
    """
    composition = dict(parsed.composition)
    for term in terms:
        for symbol, count in term.fragment.items():
            delta = term.sign * term.coefficient * count
            new_count = composition.get(symbol, 0) + delta
            if new_count < 0:
                shown_delta = f"{'+' if term.sign > 0 else ''}{delta}"
                raise AdductError(
                    f"Adduct fragment would make element '{symbol}' "
                    f"count negative ({composition.get(symbol, 0)}{shown_delta})",
                    term.start,
                )
            composition[symbol] = new_count
    return {symbol: count for symbol, count in composition.items() if count > 0}

