"""
Turn a sentence about the user's week into a weekly routine.

"I have university on Sunday and Monday and gym on Tuesday" or
"عندي جامعة من السبت للأربع وجيم التلات" becomes
{7: ["university"], 1: ["university"], 2: ["gym"]}, keyed by ISO weekday
(1 = Monday ... 7 = Sunday). The model reads free wording; a keyword parser
covers the plain cases when the model is not available.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.assistant.intent import OCCASION_WORDS, OCCASIONS
from models.assistant.openrouter_client import AssistantError

# Occasions a routine day can hold ("everyday" means a free day, "travel"
# is planned as a trip).
ROUTINE_OCCASIONS = tuple(o for o in OCCASIONS if o not in ("everyday", "travel"))
MAX_PER_DAY = 2

WEEKDAYS = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
}

# Egypt's work week runs Sunday to Thursday.
WORK_WEEK = (7, 1, 2, 3, 4)
WEEKEND = (5, 6)

_DAY_WORDS = {
    1: ("monday", "mon", "الاثنين", "الاتنين", "اثنين", "اتنين"),
    2: ("tuesday", "tue", "tues", "الثلاثاء", "الثلاثا", "التلات", "التلاتاء", "ثلاثاء", "تلات"),
    3: ("wednesday", "wed", "الاربعاء", "الاربع", "اربعاء", "اربع"),
    4: ("thursday", "thu", "thurs", "الخميس", "خميس"),
    5: ("friday", "fri", "الجمعه", "جمعه"),
    6: ("saturday", "sat", "السبت", "سبت"),
    7: ("sunday", "sun", "الاحد", "الحد", "احد"),
}
_GROUP_WORDS = {
    WORK_WEEK: ("weekdays", "weekday"),
    WEEKEND: ("weekend", "weekends", "الويكند", "ويكند", "الويك"),
}
_EVERY_DAY = ("every day", "everyday", "daily", "كل يوم", "يوميا")
_RANGE_WORDS = {"to", "through", "thru", "till", "until", "الي", "لحد", "لغايه", "حتي"}


@dataclass
class RoutineResult:
    days: Dict[int, List[str]] = field(default_factory=dict)
    reply: str = ""


def normalise_arabic(text: str) -> str:
    """Fold the spellings people type for the same Arabic word."""
    table = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ـ": None})
    return text.translate(table)


def _clean(text: str) -> str:
    text = normalise_arabic(text.lower())
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def _day_of(token: str) -> Tuple[Optional[int], bool]:
    """The weekday a token names, and whether it ends a range ("لل..." is
    "to the ..."). Arabic joins "and", "to" and "on" to the word."""
    candidates = [(token, False)]
    if token.startswith("لل"):
        candidates.append(("ال" + token[2:], True))
    if token[:1] in ("و", "ب", "ف", "ل") and len(token) > 3:
        candidates.append((token[1:], token[0] == "ل"))
    if token.startswith("وال") or token.startswith("بال"):
        candidates.append((token[1:], False))
    for word, is_range in candidates:
        for weekday, words in _DAY_WORDS.items():
            if word in words:
                return weekday, is_range
    return None, False


def _occasion_of(text: str) -> Optional[str]:
    for occasion, words in OCCASION_WORDS.items():
        if occasion not in ROUTINE_OCCASIONS:
            continue
        for word in words:
            word = normalise_arabic(word.lower())
            if word.isascii():
                if f" {word} " in f" {text} ":
                    return occasion
            elif word in text:
                return occasion
    return None


def _expand(start: int, end: int) -> List[int]:
    """Weekdays from start to end in the Egyptian week order (Saturday
    first), so "Saturday to Wednesday" wraps the right way."""
    order = [6, 7, 1, 2, 3, 4, 5]
    i, j = order.index(start), order.index(end)
    if i <= j:
        return order[i:j + 1]
    return order[i:] + order[:j + 1]


def _mentions(text: str) -> List[Tuple[str, object]]:
    """Days and occasions in the order they are said."""
    mentions: List[Tuple[str, object]] = []
    tokens = text.split()
    previous_day: Optional[int] = None
    pending_range = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        pair = f"{token} {tokens[i + 1]}" if i + 1 < len(tokens) else ""
        if pair in _EVERY_DAY or token in _EVERY_DAY:
            mentions.append(("days", [6, 7, 1, 2, 3, 4, 5]))
            i += 2 if pair in _EVERY_DAY else 1
            continue
        group = next((days for days, words in _GROUP_WORDS.items() if token in words), None)
        if group:
            mentions.append(("days", list(group)))
            i += 1
            continue
        if token in _RANGE_WORDS:
            pending_range = previous_day is not None
            i += 1
            continue
        weekday, ends_range = _day_of(token)
        if weekday is not None:
            if (pending_range or ends_range) and previous_day is not None and mentions \
                    and mentions[-1][0] == "days":
                mentions[-1] = ("days", _expand(previous_day, weekday))
            else:
                mentions.append(("days", [weekday]))
            previous_day = weekday
            pending_range = False
            i += 1
            continue
        # Two-word occasions such as "صلاة الجمعة". The second word is read
        # again, so a day in it still counts.
        occasion = _occasion_of(pair) if pair else None
        if occasion and _occasion_of(token) is None and _occasion_of(tokens[i + 1]) is None:
            mentions.append(("occasion", occasion))
            i += 1
            continue
        occasion = _occasion_of(token)
        if occasion:
            mentions.append(("occasion", occasion))
        i += 1
    return mentions


def keyword_routine(message: str) -> RoutineResult:
    """Pairs each run of occasions with the run of days next to it."""
    runs: List[Tuple[str, list]] = []
    for kind, value in _mentions(_clean(message)):
        values = value if isinstance(value, list) else [value]
        if runs and runs[-1][0] == kind:
            runs[-1][1].extend(values)
        else:
            runs.append((kind, list(values)))

    days: Dict[int, List[str]] = {}
    for i in range(0, len(runs) - 1, 2):
        first, second = runs[i], runs[i + 1]
        day_run, occasion_run = (first, second) if first[0] == "days" else (second, first)
        for weekday in day_run[1]:
            for occasion in occasion_run[1]:
                _add(days, weekday, occasion)
    return RoutineResult(days=days)


def _add(days: Dict[int, List[str]], weekday: int, occasion: str) -> None:
    current = days.setdefault(weekday, [])
    if occasion not in current and len(current) < MAX_PER_DAY:
        current.append(occasion)


def validate(raw: dict) -> RoutineResult:
    """Keep known weekdays and occasions only, at most two a day."""
    days: Dict[int, List[str]] = {}
    raw_days = raw.get("days")
    if isinstance(raw_days, dict):
        for name, occasions in raw_days.items():
            weekday = WEEKDAYS.get(str(name).strip().lower())
            if weekday is None or not isinstance(occasions, list):
                continue
            for occasion in occasions:
                key = str(occasion).strip().lower()
                if key in ROUTINE_OCCASIONS:
                    _add(days, weekday, key)
    reply = raw.get("reply")
    return RoutineResult(days=days, reply=reply.strip() if isinstance(reply, str) else "")


SYSTEM_PROMPT = """You turn what a user says about their usual week into a weekly routine for a wardrobe app in Egypt.
Answer with one JSON object and nothing else:
{
  "days": {"sunday": ["university"], "tuesday": ["university", "gym"]},
  "reply": "one short sentence confirming what you understood"
}
Rules:
- Keys are English weekday names in lower case. Leave out days with nothing planned.
- Each day has at most 2 occasions from %(occasions)s, in the order they happen that day.
- "family" covers Friday prayer, family visits and Eid.
- In Egypt the work week is Sunday to Thursday; the weekend is Friday and Saturday.
- Ranges such as "from Saturday to Wednesday" include both ends.
- Egyptian Arabic day names: السبت، الحد/الأحد، الاتنين، التلات، الأربع، الخميس، الجمعة.
- %(language)s"""


def build_messages(message: str, locale: Optional[str]) -> List[dict]:
    language = (
        "Write the reply in Egyptian Arabic."
        if (locale or "").lower().startswith("ar")
        else "Write the reply in the user's language."
    )
    system = SYSTEM_PROMPT % {
        "occasions": json.dumps(list(ROUTINE_OCCASIONS)),
        "language": language,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


def to_json(days: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """Weekday keys as strings, the way the app stores the routine."""
    return {str(weekday): occasions for weekday, occasions in sorted(days.items()) if occasions}


def parse(message: str, locale: Optional[str], complete) -> Tuple[RoutineResult, bool]:
    """The routine in [message], and whether the keyword parser was used.
    [complete] sends messages to the model and returns its JSON."""
    try:
        result = validate(complete(build_messages(message, locale)))
        if result.days:
            return result, False
    except AssistantError:
        pass
    return keyword_routine(message), True
