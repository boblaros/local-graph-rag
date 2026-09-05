"""Deterministic answer normalization and lexical QA metrics.

The functions are independent of models and storage so metrics can be
recomputed from saved answers. They do not use the upstream evaluator's
permissive any-word-overlap rule.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Final


INSUFFICIENT_INFORMATION: Final[str] = "INSUFFICIENT_INFORMATION"
CANONICAL_UNANSWERABLE: Final[str] = INSUFFICIENT_INFORMATION

_ARTICLES = {"a", "an", "the"}
_WHITESPACE_RE = re.compile(r"\s+")
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?answer(?:\s+to\s+the\s+question)?\s+(?:is|was)\s*[:\-]?\s*",
    re.IGNORECASE,
)

_UNANSWERABLE_PHRASES = {
    "cannot answer",
    "cannot be answered",
    "cannot determine",
    "cannot be determined",
    "do not know",
    "i do not know",
    "i dont know",
    "insufficient context",
    "insufficient evidence",
    "insufficient info",
    "insufficient information",
    "not enough context",
    "not enough evidence",
    "not enough info",
    "not enough information",
    "unable to answer",
    "unable to determine",
    "unknown",
}

_MONTHS: dict[str, int] = {}
for _month_number in range(1, 13):
    _MONTHS[calendar.month_name[_month_number].casefold()] = _month_number
    _MONTHS[calendar.month_abbr[_month_number].casefold()] = _month_number
_MONTH_PATTERN = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))

_SCALE_FACTORS = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "mn": Decimal("1000000"),
    "million": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "t": Decimal("1000000000000"),
    "tn": Decimal("1000000000000"),
    "trillion": Decimal("1000000000000"),
}


def _coerce_text(value: object) -> str:
    return "" if value is None else str(value)


def _strip_answer_prefix(value: str) -> str:
    return _ANSWER_PREFIX_RE.sub("", value.strip(), count=1)


def _plain_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("_", " ").replace("’", "'")
    value = re.sub(r"[^\w\s']", " ", value, flags=re.UNICODE)
    value = value.replace("'", "")
    return _WHITESPACE_RE.sub(" ", value).strip()


def canonicalize_unanswerable(value: object) -> str | None:
    """Return the canonical abstention label when *value* is an abstention.

    Matching is intentionally anchored to the complete answer after removal of
    a harmless ``the answer is`` prefix.  A substantive answer that merely
    mentions insufficient evidence is therefore not silently converted into an
    abstention.
    """

    text = _strip_answer_prefix(_coerce_text(value)).strip(" `\"'\n\t")
    if not text:
        return None
    phrase = _plain_phrase(text)
    if phrase in _UNANSWERABLE_PHRASES:
        return INSUFFICIENT_INFORMATION

    suffix = (
        r"(?:\s+(?:in|from|based on)\s+(?:the\s+)?(?:provided\s+)?"
        r"(?:context|information|evidence))?"
    )
    purpose = (
        r"(?:\s+to\s+(?:answer|determine|resolve)(?:\s+the\s+(?:question|answer))?)?"
    )
    patterns = (
        rf"^insufficient\s+(?:information|info|context|evidence){purpose}$",
        rf"^not\s+enough\s+(?:information|info|context|evidence){purpose}$",
        rf"^(?:i\s+)?(?:can(?:not|t)|cannot|unable\s+to)\s+(?:answer|determine){suffix}$",
    )
    return (
        INSUFFICIENT_INFORMATION
        if any(re.fullmatch(pattern, phrase) for pattern in patterns)
        else None
    )


def normalize_text(value: object) -> str:
    """SQuAD-style lexical normalization with Unicode-aware punctuation."""

    unanswerable = canonicalize_unanswerable(value)
    if unanswerable is not None:
        return unanswerable

    text = unicodedata.normalize("NFKC", _coerce_text(value)).casefold()
    chars = []
    for char in text:
        category = unicodedata.category(char)
        chars.append(" " if category.startswith(("P", "S")) else char)
    tokens = [token for token in "".join(chars).split() if token not in _ARTICLES]
    return " ".join(tokens)


def normalize_yes_no(value: object) -> str | None:
    """Normalize a direct yes/no answer, allowing a short explanation."""

    if canonicalize_unanswerable(value) is not None:
        return None
    text = _plain_phrase(_strip_answer_prefix(_coerce_text(value)))
    if re.match(r"^(?:yes|yeah|yep|true|affirmative)(?:\s|$)", text):
        return "yes"
    if re.match(r"^(?:no|nope|false|negative)(?:\s|$)", text):
        return "no"
    return None


def yes_no_match(prediction: object, gold: object) -> bool:
    prediction_value = normalize_yes_no(prediction)
    gold_value = normalize_yes_no(gold)
    return prediction_value is not None and prediction_value == gold_value


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(value: object, *, day_first: bool = False) -> str | None:
    """Normalize common English and ISO date forms.

    Returned precision is preserved: a year stays ``YYYY`` and a month stays
    ``YYYY-MM``. Ambiguous all-numeric dates use month-first ordering unless
    ``day_first=True``; unambiguous dates are detected automatically.
    """

    if canonicalize_unanswerable(value) is not None:
        return None
    text = unicodedata.normalize("NFKC", _strip_answer_prefix(_coerce_text(value)))
    text = text.strip().casefold()
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text)
    text = _WHITESPACE_RE.sub(" ", text).strip(" .,")

    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[tT ].*)?", text)
    if match:
        return _valid_date(*(int(part) for part in match.groups()))

    match = re.fullmatch(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})", text)
    if match:
        return _valid_date(*(int(part) for part in match.groups()))

    match = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", text)
    if match:
        first, second, year = (int(part) for part in match.groups())
        if first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        elif day_first:
            day, month = first, second
        else:
            month, day = first, second
        return _valid_date(year, month, day)

    match = re.fullmatch(rf"({_MONTH_PATTERN})\s+(\d{{1,2}})(?:,)?\s+(\d{{4}})", text)
    if match:
        month_name, day_value, year_value = match.groups()
        return _valid_date(int(year_value), _MONTHS[month_name], int(day_value))

    match = re.fullmatch(rf"(\d{{1,2}})\s+({_MONTH_PATTERN})(?:,)?\s+(\d{{4}})", text)
    if match:
        day_value, month_name, year_value = match.groups()
        return _valid_date(int(year_value), _MONTHS[month_name], int(day_value))

    match = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if match:
        year, month = (int(part) for part in match.groups())
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None

    match = re.fullmatch(rf"({_MONTH_PATTERN})\s+(\d{{4}})", text)
    if match:
        month_name, year_value = match.groups()
        return f"{int(year_value):04d}-{_MONTHS[month_name]:02d}"

    match = re.fullmatch(r"\d{4}", text)
    return text if match else None


def date_match(prediction: object, gold: object, *, day_first: bool = False) -> bool:
    prediction_value = normalize_date(prediction, day_first=day_first)
    gold_value = normalize_date(gold, day_first=day_first)
    return prediction_value is not None and prediction_value == gold_value


def _decimal_to_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def normalize_numeric_answer(value: object) -> str | None:
    """Normalize a direct numeric answer, including separators and scale words."""

    if canonicalize_unanswerable(value) is not None:
        return None
    text = unicodedata.normalize("NFKC", _strip_answer_prefix(_coerce_text(value)))
    text = text.strip().casefold().replace("−", "-")
    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()

    pattern = re.compile(
        r"^[\s$€£¥]*([+\-]?)\s*"
        r"((?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.(\d+))?\s*"
        r"(thousand|million|billion|trillion|mn|bn|tn|k|m|b|t)?\s*"
        r"(%|percent|percentage\s+points?)?\s*"
        r"(?:usd|eur|gbp|dollars?|euros?|pounds?)?[\s.!?]*$",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(text)
    if not match:
        return None

    sign, integer, fraction, scale, percent = match.groups()
    number_text = integer.replace(",", "")
    if fraction is not None:
        number_text += f".{fraction}"
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return None
    if sign == "-" or negative_parentheses:
        number = -number
    if scale:
        number *= _SCALE_FACTORS[scale.casefold()]
    suffix = "%" if percent else ""
    return f"{_decimal_to_string(number)}{suffix}"


def normalize_number(value: object) -> str | None:
    """Alias retained for callers that use the shorter function name."""

    return normalize_numeric_answer(value)


def numeric_match(prediction: object, gold: object) -> bool:
    prediction_value = normalize_numeric_answer(prediction)
    gold_value = normalize_numeric_answer(gold)
    return prediction_value is not None and prediction_value == gold_value


def normalize_answer(value: object) -> str:
    """Normalize an answer for exact-match and token-overlap metrics."""

    unanswerable = canonicalize_unanswerable(value)
    if unanswerable is not None:
        return unanswerable
    yes_no = normalize_yes_no(value)
    if yes_no is not None:
        return yes_no
    date_value = normalize_date(value)
    if date_value is not None:
        return date_value
    numeric_value = normalize_numeric_answer(value)
    if numeric_value is not None:
        return numeric_value
    return normalize_text(value)


def normalized_exact_match(prediction: object, gold: object) -> bool:
    """Return true only when the complete normalized answers are identical."""

    return normalize_answer(prediction) == normalize_answer(gold)


def exact_match(prediction: object, gold: object) -> bool:
    return normalized_exact_match(prediction, gold)


@dataclass(frozen=True)
class TokenScores:
    precision: float
    recall: float
    f1: float


def token_scores(prediction: object, gold: object) -> TokenScores:
    """Compute bag-of-token precision, recall, and F1 after normalization."""

    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens and not gold_tokens:
        return TokenScores(1.0, 1.0, 1.0)
    if not prediction_tokens or not gold_tokens:
        return TokenScores(0.0, 0.0, 0.0)

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return TokenScores(precision, recall, f1)


def token_precision_recall_f1(
    prediction: object, gold: object
) -> tuple[float, float, float]:
    scores = token_scores(prediction, gold)
    return scores.precision, scores.recall, scores.f1


def token_f1(prediction: object, gold: object) -> float:
    return token_scores(prediction, gold).f1


__all__ = [
    "CANONICAL_UNANSWERABLE",
    "INSUFFICIENT_INFORMATION",
    "TokenScores",
    "canonicalize_unanswerable",
    "date_match",
    "exact_match",
    "normalize_answer",
    "normalize_date",
    "normalize_number",
    "normalize_numeric_answer",
    "normalize_text",
    "normalize_yes_no",
    "normalized_exact_match",
    "numeric_match",
    "token_f1",
    "token_precision_recall_f1",
    "token_scores",
    "yes_no_match",
]
