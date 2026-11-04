"""
Turn a chat message into choices the recommender understands.

The model only picks an occasion, style, weather and which parts of the
current outfit to change. The recommender still picks the real items, so a
reply can never contain clothes the user does not own.
"""
import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from api.schemas import WardrobeItemFull
from models.recommender.recommender import ZONES, _zone
from models.recommender.rule_engine import normalise_label

OCCASIONS = ("everyday", "university", "work", "date night", "gym", "party", "travel", "formal", "family")
STYLES = ("streetwear", "elegant", "minimal", "vintage", "sport")
WEATHERS = ("hot", "mild", "cold", "rain")
CHANGE_ZONES = tuple(ZONES) + ("all",)

MAX_PROMPT_ITEMS = 80


@dataclass
class Intent:
    wants_outfit: bool = True
    occasion: Optional[str] = None
    style: Optional[str] = None
    weather: Optional[str] = None
    change_zones: List[str] = field(default_factory=list)
    lock_ids: List[str] = field(default_factory=list)
    exclude_ids: List[str] = field(default_factory=list)
    reply: str = ""


def _pick(value, allowed: Sequence[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = normalise_label(value)
    for option in allowed:
        if normalise_label(option) == key:
            return option
    return None


def _string_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(entry) for entry in value if entry is not None and str(entry)]


def validate(raw: dict, wardrobe_ids: Iterable[str]) -> Intent:
    """Keep only values the recommender supports and ids the user owns."""
    owned = set(wardrobe_ids)
    zones = []
    for zone in _string_list(raw.get("change_zones")):
        picked = _pick(zone, CHANGE_ZONES)
        if picked and picked not in zones:
            zones.append(picked)
    reply = raw.get("reply")
    return Intent(
        wants_outfit=raw.get("wants_outfit") is not False,
        occasion=_pick(raw.get("occasion"), OCCASIONS),
        style=_pick(raw.get("style"), STYLES),
        weather=_pick(raw.get("weather"), WEATHERS),
        change_zones=zones,
        lock_ids=[item_id for item_id in _string_list(raw.get("lock_ids")) if item_id in owned],
        exclude_ids=[item_id for item_id in _string_list(raw.get("exclude_ids")) if item_id in owned],
        reply=reply.strip() if isinstance(reply, str) else "",
    )


OCCASION_WORDS = {
    "gym": ("gym", "workout", "training", "exercise", "jog", "جيم", "تمرين", "رياضة"),
    "work": ("work", "office", "meeting", "interview", "job", "عمل", "شغل", "مكتب", "مقابلة", "اجتماع"),
    "date night": ("date", "dinner", "romantic", "موعد", "عشاء"),
    "party": ("party", "wedding", "celebration", "حفلة", "فرح", "عرس"),
    "university": ("university", "college", "class", "campus", "lecture", "جامعة", "كلية", "محاضرة"),
    "travel": ("travel", "trip", "flight", "airport", "سفر", "رحلة", "مطار"),
    "formal": ("formal", "رسمي"),
    "family": ("family", "friday prayer", "eid", "عيلة", "عائلة", "صلاة الجمعة", "العيد", "زيارة"),
    "everyday": ("casual", "everyday", "relaxed", "كاجوال", "عادي"),
}
STYLE_WORDS = {
    "elegant": ("elegant", "classy", "chic", "fancy", "أنيق", "شيك"),
    "streetwear": ("streetwear", "street", "ستريت"),
    "minimal": ("minimal", "simple", "clean", "بسيط"),
    "vintage": ("vintage", "retro", "كلاسيكي قديم", "فينتج"),
    "sport": ("sporty", "athletic", "رياضي"),
}
WEATHER_WORDS = {
    "cold": ("warm", "warmer", "cold", "chilly", "freezing", "دافئ", "أدفأ", "أدفى", "دفا", "برد", "بارد"),
    "hot": ("hot", "cooler", "lighter", "summer", "الحر", "حار", "أخف"),
    "rain": ("rain", "rainy", "مطر", "ممطر"),
}
ZONE_WORDS = {
    "shoes": ("shoe", "shoes", "sneaker", "boots", "حذاء", "جزمة", "كوتشي"),
    "top": ("top", "shirt", "t-shirt", "tee", "blouse", "sweater", "قميص", "تيشيرت", "بلوزة"),
    "bottom": ("pants", "jeans", "trousers", "bottom", "skirt", "shorts", "بنطلون", "جينز", "تنورة"),
    "outerwear": ("jacket", "coat", "blazer", "جاكيت", "معطف", "بليزر"),
    "bag": ("bag", "backpack", "purse", "شنطة", "حقيبة"),
    "accessory": ("accessory", "accessories", "watch", "belt", "hat", "cap", "ساعة", "حزام", "اكسسوار"),
    "all": ("different", "another", "something else", "new look", "color", "colour", "مختلف", "لون", "تاني"),
}


def _matches(text: str, words: Sequence[str]) -> bool:
    for word in words:
        if word.isascii():
            if f" {word} " in f" {text} ":
                return True
        elif word in text:
            return True
    return False


def _first(text: str, table: dict) -> Optional[str]:
    return next((key for key, words in table.items() if _matches(text, words)), None)


def keyword_intent(message: str) -> Intent:
    text = " ".join(
        "".join(ch if ch.isalnum() or ch in "-'" else " " for ch in message.lower()).split()
    )
    zones = [zone for zone, words in ZONE_WORDS.items() if _matches(text, words)]
    if len(zones) > 1 and "all" in zones:
        zones.remove("all")
    return Intent(
        occasion=_first(text, OCCASION_WORDS),
        style=_first(text, STYLE_WORDS),
        weather=_first(text, WEATHER_WORDS),
        change_zones=zones,
    )


SYSTEM_PROMPT = """You are a friendly personal stylist inside a wardrobe app.
The app picks real clothes from the user's wardrobe. You only decide what to ask it for.
Answer with one JSON object and nothing else:
{
  "wants_outfit": true or false,
  "occasion": one of %(occasions)s or null,
  "style": one of %(styles)s or null,
  "weather": one of %(weathers)s or null,
  "change_zones": list from %(zones)s,
  "lock_ids": ids from the wardrobe to keep,
  "exclude_ids": ids from the wardrobe to avoid,
  "reply": "one or two short sentences for the user"
}
Rules:
- wants_outfit is false only for questions that need no outfit, such as styling advice.
- When the user wants to change part of the current outfit, list those parts in change_zones. Use "all" for a whole new look.
- "warmer" means weather "cold". "lighter" or "cooler" means weather "hot".
- Only use ids that appear in the wardrobe list.
- Never name clothes the user does not own. Do not describe the exact items; the app shows them.
- %(language)s"""


def _summary_line(item: WardrobeItemFull) -> str:
    parts = [str(item.id), _zone(item) or "other", item.type, item.color]
    if item.style:
        parts.append(item.style)
    return " | ".join(parts)


def build_messages(
    message: str,
    history: Sequence[dict],
    wardrobe: Sequence[WardrobeItemFull],
    current_outfit_ids: Sequence[str],
    occasion: Optional[str],
    weather: Optional[str],
    season: Optional[str],
    locale: Optional[str],
) -> List[dict]:
    language = (
        "Write the reply in Arabic."
        if (locale or "").lower().startswith("ar")
        else "Write the reply in the user's language."
    )
    system = SYSTEM_PROMPT % {
        "occasions": json.dumps(list(OCCASIONS)),
        "styles": json.dumps(list(STYLES)),
        "weathers": json.dumps(list(WEATHERS)),
        "zones": json.dumps(list(CHANGE_ZONES)),
        "language": language,
    }
    current = set(current_outfit_ids)
    lines = [_summary_line(item) for item in wardrobe[:MAX_PROMPT_ITEMS]]
    outfit = [_summary_line(item) for item in wardrobe if str(item.id) in current]
    context = "\n".join([
        f"Weather now: {weather or 'unknown'}. Season: {season or 'unknown'}.",
        f"Current occasion: {occasion or 'none'}.",
        "Wardrobe (id | part | type | color | style):",
        *lines,
        "Current outfit:" if outfit else "No current outfit.",
        *outfit,
    ])

    messages = [{"role": "system", "content": system}, {"role": "system", "content": context}]
    for turn in list(history)[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})
    return messages
