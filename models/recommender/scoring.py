"""Scores how good an outfit is, beyond the hard rules.

The rule engine only says yes or no (a type banned for the occasion, the
season or the weather). Among the outfits it allows, this module ranks them
on six things, each between 0 and 1:

- color:     do the colors go together (hue, tone, neutrals)?
- formality: are the pieces about as dressy as each other and the occasion?
- warmth:    is it warm enough for the day, and with its jacket for the night?
- style:     do the pieces' styles (streetwear, elegant...) agree?
- pattern:   at most one bold pattern.
- diversity: favor pieces not worn lately, and differ from the last outfit.

Each part also says why, as reason codes the app translates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from api.schemas import ReasonCode, WardrobeItemFull
from models.colors import REFERENCE_SHADES, family as color_family, hex_to_rgb, rgb_to_lab
from models.recommender.rule_engine import normalise_label, normalise_type

WEIGHTS = {
    "color": 0.28,
    "formality": 0.24,
    "warmth": 0.20,
    "style": 0.09,
    "pattern": 0.09,
    "diversity": 0.10,
}

# How dressy each type is, from 1 (gym, beach) to 5 (formal).
FORMALITY = {
    "tank_top": 1.0, "t_shirt": 2.0, "polo": 2.5, "shirt": 3.5, "blouse": 3.5,
    "sweater": 3.0, "hoodie": 1.5, "sweatshirt": 1.5, "cardigan": 3.0,
    "jeans": 2.5, "trousers": 4.0, "chinos": 3.0, "joggers": 1.5, "shorts": 1.5,
    "skirt": 3.0, "leggings": 1.5,
    "jacket": 2.5, "denim_jacket": 2.0, "leather_jacket": 3.0, "heavy_coat": 3.0,
    "coat": 4.0, "blazer": 4.5, "vest": 4.0,
    "dress": 3.5, "jumpsuit": 3.0,
    "sneakers": 2.0, "boots": 3.0, "heels": 4.5, "flats": 3.0, "sandals": 1.5,
    "flip_flops": 1.0, "loafers": 3.5, "oxford_shoes": 5.0,
}
DEFAULT_FORMALITY = 2.5
STYLE_FORMALITY = {"elegant": 0.5, "sport": -0.75, "streetwear": -0.25}

# The dressiness each occasion wants, as (lowest, highest).
OCCASION_FORMALITY = {
    "everyday": (1.5, 3.0), "casual": (1.5, 3.0), "university": (2.0, 3.25),
    "work": (3.0, 4.5), "business": (3.5, 5.0), "date_night": (3.0, 4.25),
    "party": (2.5, 4.25), "formal": (4.0, 5.0), "wedding": (4.0, 5.0),
    "family": (2.5, 3.75), "gym": (1.0, 2.0), "sport": (1.0, 2.0),
    "beach": (1.0, 2.25), "sightseeing": (1.5, 3.0), "travel": (1.5, 3.0),
}

# How warm each type is, roughly in layers of a t-shirt.
WARMTH = {
    "tank_top": 0.5, "t_shirt": 1.0, "polo": 1.0, "shirt": 1.5, "blouse": 1.25,
    "sweater": 3.0, "hoodie": 2.5, "sweatshirt": 2.5, "cardigan": 2.25,
    "jeans": 2.0, "trousers": 1.5, "chinos": 1.5, "joggers": 2.0, "shorts": 0.5,
    "skirt": 1.0, "leggings": 1.5,
    "jacket": 2.5, "denim_jacket": 2.0, "leather_jacket": 3.0, "heavy_coat": 5.0,
    "coat": 4.0, "blazer": 2.0, "vest": 1.0,
    "dress": 1.5, "jumpsuit": 2.0,
    "sneakers": 0.5, "boots": 1.0, "heels": 0.25, "flats": 0.25, "sandals": 0.0,
    "flip_flops": 0.0, "loafers": 0.5, "oxford_shoes": 0.5,
}
DEFAULT_WARMTH = 1.0
SEASON_WARMTH = {"winter": 0.5, "autumn": 0.25, "summer": -0.25}

# When only the weather word is known, a typical day for it (°C).
WEATHER_HIGH = {"hot": 33.0, "mild": 23.0, "cold": 13.0, "rain": 17.0}

# How well two styles sit together; missing pairs count as 0.7.
STYLE_PAIRS = {
    ("elegant", "minimal"): 0.95, ("minimal", "streetwear"): 0.9,
    ("minimal", "vintage"): 0.9, ("minimal", "sport"): 0.75,
    ("elegant", "vintage"): 0.8, ("streetwear", "sport"): 0.75,
    ("streetwear", "vintage"): 0.6, ("elegant", "streetwear"): 0.4,
    ("sport", "vintage"): 0.35, ("elegant", "sport"): 0.1,
}

PLAIN_PATTERNS = {"", "plain", "solid", "none", "unknown"}

# Fashion neutrals: they go with nearly everything.
NEUTRAL_FAMILIES = {
    "white", "cream", "beige", "khaki", "tan", "brown", "grey", "light grey",
    "charcoal", "black", "navy",
}
NEUTRAL_CHROMA = 12.0
BOLD_CHROMA = 35.0
# A night at least this warm (°C) needs no jacket.
WARM_NIGHT = 22.0
# Hues this close count as one color; lighter tints drift by about 30°.
SAME_HUE = 35.0

CORE_ZONES = ("top", "bottom", "one_piece", "outerwear", "shoes")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class _Color:
    name: str
    lightness: float
    chroma: float
    hue: float
    weight: float

    @property
    def neutral(self) -> bool:
        return self.name in NEUTRAL_FAMILIES or self.chroma < NEUTRAL_CHROMA


@dataclass
class _Features:
    key: str
    item_id: str
    zone: Optional[str]
    type: str
    formality: float
    warmth: float
    style: str
    pattern: str
    colors: List[_Color] = field(default_factory=list)
    days_since_worn: Optional[int] = None


def _lab(hex_code: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not hex_code:
        return None
    try:
        lightness, a, b = (float(v) for v in rgb_to_lab(hex_to_rgb(hex_code)))
    except Exception:
        return None
    return lightness, a, b


def _color(name: Optional[str], hex_value: Optional[str], weight: float) -> Optional[_Color]:
    label = (name or "").strip().lower()
    shades = REFERENCE_SHADES.get(label)
    lab = _lab(hex_value)
    reference = _lab(shades[0]) if shades else None
    if lab is None:
        lab = reference
    elif reference is not None and label not in NEUTRAL_FAMILIES:
        # A colorful name with a grey measured shade: the hex was not
        # recorded, so trust the name.
        if math.hypot(lab[1], lab[2]) < NEUTRAL_CHROMA <= math.hypot(reference[1], reference[2]):
            lab = reference
    if lab is None:
        return None
    lightness, a, b = lab
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360
    return _Color(label, lightness, chroma, hue, weight)


def _hue_gap(first: float, second: float) -> float:
    gap = abs(first - second) % 360
    return min(gap, 360 - gap)


def _fit(value: float, target: float, tolerance: float = 1.5, spread: float = 4.0) -> float:
    """1 within [tolerance] of [target], falling to 0 [spread] further out."""
    return _clamp(1 - max(0.0, abs(value - target) - tolerance) / spread)


def warmth_target(temperature: float) -> float:
    """The total warmth an outfit should have at [temperature] °C."""
    return 3.0 + max(0.0, 28.0 - temperature) * 0.35


@dataclass
class Score:
    total: float
    parts: Dict[str, float]


class OutfitScorer:
    """Scores candidate outfits for one request."""

    def __init__(
        self,
        zone_of,
        color_rules: Dict[str, List[str]],
        occasion: Optional[str] = None,
        weather: Optional[str] = None,
        season: Optional[str] = None,
        temp_high: Optional[float] = None,
        temp_low: Optional[float] = None,
        layer: bool = False,
        last_worn: Optional[Dict[str, str]] = None,
        previous_ids: Optional[Sequence[str]] = None,
        today: Optional[date] = None,
    ):
        self._zone_of = zone_of
        self._color_rules = color_rules
        self.occasion = normalise_label(occasion)
        self.season = normalise_label(season)
        self.temp_high, self.temp_low = self._temperatures(
            normalise_label(weather), temp_high, temp_low, layer,
        )
        self._today = today or date.today()
        self._last_worn = {str(k): v for k, v in (last_worn or {}).items()}
        self.previous = {str(item_id) for item_id in (previous_ids or []) if str(item_id)}
        self._features: Dict[str, _Features] = {}
        self._pairs: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _temperatures(weather, high, low, layer) -> Tuple[Optional[float], Optional[float]]:
        if high is None and weather in WEATHER_HIGH:
            high = WEATHER_HIGH[weather]
        if high is not None and low is None:
            # A cool night was flagged, or assume the usual day-night swing.
            low = min(high - 8.0, 16.0) if layer else high - 8.0
        return high, low

    # ---- per item ----

    def features(self, item: WardrobeItemFull) -> _Features:
        # The piece's id plus what it is, so two different pieces never share
        # cached features.
        key = f"{item.id}|{item.type}|{item.color}|{item.hex}|{item.pattern}|{item.style}"
        cached = self._features.get(key)
        if cached is not None:
            return cached
        type_key = normalise_type(item.type)
        style = normalise_label(item.style)
        formality = FORMALITY.get(type_key, DEFAULT_FORMALITY) + STYLE_FORMALITY.get(style, 0.0)
        warmth = WARMTH.get(type_key, DEFAULT_WARMTH) + SEASON_WARMTH.get(normalise_label(item.season), 0.0)
        colors = [c for c in (
            _color(color_family(item.color), item.hex, 1.0),
            _color(color_family(item.secondary_color), item.secondary_hex, 0.5)
            if item.secondary_color else None,
        ) if c is not None]
        days = None
        worn = self._last_worn.get(str(item.id)) if item.id is not None else None
        if worn:
            try:
                days = (self._today - date.fromisoformat(worn[:10])).days
            except ValueError:
                days = None
        features = _Features(
            key=key,
            item_id="" if item.id is None else str(item.id),
            zone=self._zone_of(item),
            type=type_key,
            formality=_clamp(formality, 1.0, 5.0),
            warmth=max(0.0, warmth),
            style=style,
            pattern=normalise_label(item.pattern),
            colors=colors,
            days_since_worn=days,
        )
        self._features[key] = features
        return features

    def too_warm(self, item: WardrobeItemFull) -> bool:
        """A piece far too warm to suggest at all: a heavy piece on a hot day,
        or a jacket when even the night is warm (a blazer still suits a
        dressy occasion)."""
        if self.temp_high is None:
            return False
        features = self.features(item)
        if features.zone == "outerwear":
            if self.temp_low is None or self.temp_low < WARM_NIGHT:
                return False
            low, _ = OCCASION_FORMALITY.get(self.occasion, (1.0, 5.0))
            return not (features.type in {"blazer", "vest"} and low >= 3.0)
        return self.temp_high >= 30 and features.warmth >= 3.0

    def item_fit(self, item: WardrobeItemFull) -> float:
        """How well one piece suits the occasion on its own, for pruning."""
        features = self.features(item)
        low, high = OCCASION_FORMALITY.get(self.occasion, (1.0, 5.0))
        gap = max(0.0, low - features.formality, features.formality - high)
        return _clamp(1 - gap / 2.0)

    # ---- outfit parts ----

    def _pair_score(self, first: _Color, second: _Color) -> float:
        key = (f"{first.name}:{first.hue:.0f}:{first.lightness:.0f}",
               f"{second.name}:{second.hue:.0f}:{second.lightness:.0f}")
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        allowed_by_rules = any(
            "matches_all" in self._color_rules.get(a, []) or b in self._color_rules.get(a, [])
            for a, b in ((first.name, second.name), (second.name, first.name))
        )
        if first.neutral and second.neutral:
            score = 0.9
        elif first.neutral or second.neutral:
            score = 0.95
        else:
            gap = _hue_gap(first.hue, second.hue)
            if gap < SAME_HUE:
                score = 1.0 if abs(first.lightness - second.lightness) >= 12 else 0.85
            elif gap < 60:
                score = 0.85
            elif gap >= 150:
                muted = min(first.chroma, second.chroma) < BOLD_CHROMA
                score = 0.75 if muted else 0.6
            else:
                score = 0.35
        if allowed_by_rules:
            score = max(score, 1.0)
        self._pairs[key] = score
        return score

    def _color_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        colors = [(f, c) for f in items for c in f.colors]
        if len(colors) < 2:
            return 0.8, []
        total = weight = 0.0
        best: Tuple[float, str, str] = (0.0, "", "")
        worst: Tuple[float, str, str] = (1.0, "", "")
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                (fa, a), (fb, b) = colors[i], colors[j]
                if fa is fb:
                    continue  # an item's own two colors already go together
                pair = self._pair_score(a, b)
                w = a.weight * b.weight
                total += pair * w
                weight += w
                if a.name != b.name and pair > best[0]:
                    best = (pair, a.name, b.name)
                if pair < worst[0]:
                    worst = (pair, a.name, b.name)
        score = total / weight if weight else 0.8
        chromatic = [c for _, c in colors if not c.neutral]
        bold_hues: List[float] = []
        for c in chromatic:
            if c.chroma >= BOLD_CHROMA and all(_hue_gap(c.hue, h) >= SAME_HUE for h in bold_hues):
                bold_hues.append(c.hue)
        if len(bold_hues) > 2:
            score -= 0.15 * (len(bold_hues) - 2)
        codes: List[ReasonCode] = []
        hues = [c.hue for c in chromatic]
        if len(hues) >= 2 and len(colors) >= 3 and all(_hue_gap(hues[0], h) < SAME_HUE for h in hues):
            codes.append(ReasonCode(code="tonal_outfit", colors=[chromatic[0].name]))
        elif not chromatic:
            codes.append(ReasonCode(code="neutral_palette"))
        if best[0] >= 0.85 and best[1]:
            codes.append(ReasonCode(code="color_pair", colors=[best[1], best[2]]))
        if worst[0] < 0.5 and worst[1]:
            codes.append(ReasonCode(code="color_clash", colors=[worst[1], worst[2]]))
        if len(bold_hues) > 2:
            codes.append(ReasonCode(code="bold_colors"))
        return _clamp(score), codes

    def _formality_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        levels = [f.formality for f in items]
        if not levels:
            return 0.8, []
        coherence = _clamp(1 - max(0.0, max(levels) - min(levels) - 1.5) / 2.5)
        low, high = OCCASION_FORMALITY.get(self.occasion, (1.0, 5.0))
        mean = sum(levels) / len(levels)
        fit = _clamp(1 - max(0.0, low - mean, mean - high) / 1.5)
        if min(levels) < low - 1.0:
            fit = _clamp(fit - 0.3)
        score = 0.5 * coherence + 0.5 * fit
        codes = []
        if self.occasion in OCCASION_FORMALITY and score >= 0.85:
            codes.append(ReasonCode(code="formality_match", detail=self.occasion))
        return score, codes

    def _warmth_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        if self.temp_high is None:
            return 0.8, []
        outer = [f for f in items if f.zone == "outerwear"]
        base = sum(f.warmth for f in items if f.zone != "outerwear")
        full = base + sum(f.warmth for f in outer)
        day = _fit(base, warmth_target(self.temp_high))
        night = _fit(full, warmth_target(self.temp_low)) if self.temp_low is not None else day
        score = 0.6 * day + 0.4 * night
        codes = []
        if outer and self.temp_low is not None and self.temp_low <= 18 and night >= 0.6:
            codes.append(ReasonCode(code="warm_enough_for_night", detail=f"{self.temp_low:.0f}"))
        if self.temp_high >= 30 and day >= 0.8:
            codes.append(ReasonCode(code="light_for_heat", detail=f"{self.temp_high:.0f}"))
        elif self.temp_high <= 16 and day >= 0.8:
            codes.append(ReasonCode(code="warm_for_cold", detail=f"{self.temp_high:.0f}"))
        return score, codes

    def _style_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        styles = [f.style for f in items if f.style]
        if len(styles) < 2:
            return 0.85, []
        total = count = 0
        for i in range(len(styles)):
            for j in range(i + 1, len(styles)):
                a, b = sorted((styles[i], styles[j]))
                total += 1.0 if a == b else STYLE_PAIRS.get((a, b), 0.7)
                count += 1
        score = total / count
        codes = []
        if len(set(styles)) == 1:
            codes.append(ReasonCode(code="style_harmony", detail=styles[0]))
        return score, codes

    def _pattern_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        bold = [f for f in items if f.pattern not in PLAIN_PATTERNS]
        if not bold:
            return 1.0, []
        if len(bold) == 1:
            return 1.0, [ReasonCode(code="one_statement_pattern", detail=bold[0].pattern)]
        if len(bold) == 2:
            return 0.45, [ReasonCode(code="pattern_clash")]
        return 0.15, [ReasonCode(code="pattern_clash")]

    def _diversity_part(self, items: List[_Features]) -> Tuple[float, List[ReasonCode]]:
        def fresh(f: _Features) -> float:
            days = f.days_since_worn
            if days is None or days >= 30:
                return 1.0
            if days <= 3:
                return 0.2
            if days <= 7:
                return 0.5
            if days <= 14:
                return 0.75
            return 0.9

        score = sum(fresh(f) for f in items) / len(items) if items else 1.0
        if self.previous:
            overlap = sum(1 for f in items if f.item_id in self.previous) / max(1, len(items))
            score *= 1 - 0.5 * overlap
        codes = []
        if self._last_worn:
            forgotten = [f for f in items if f.days_since_worn is None or f.days_since_worn >= 30]
            if forgotten:
                codes.append(ReasonCode(code="forgotten_piece", detail=forgotten[0].type))
        return score, codes

    # ---- whole outfit ----

    def _core(self, outfit: Iterable[WardrobeItemFull]) -> List[_Features]:
        return [f for f in (self.features(item) for item in outfit) if f.zone in CORE_ZONES]

    def score(self, outfit: Sequence[WardrobeItemFull]) -> Score:
        items = self._core(outfit)
        parts = {
            "color": self._color_part(items)[0],
            "formality": self._formality_part(items)[0],
            "warmth": self._warmth_part(items)[0],
            "style": self._style_part(items)[0],
            "pattern": self._pattern_part(items)[0],
            "diversity": self._diversity_part(items)[0],
        }
        total = sum(WEIGHTS[name] * value for name, value in parts.items())
        return Score(total, parts)

    def reasons(self, outfit: Sequence[WardrobeItemFull]) -> Tuple[float, List[ReasonCode]]:
        """The color score and the reasons for [outfit], most telling first."""
        items = self._core(outfit)
        color_score, color_codes = self._color_part(items)
        codes: List[ReasonCode] = []
        for part in (
            self._warmth_part(items)[1],
            self._formality_part(items)[1],
            color_codes,
            self._pattern_part(items)[1],
            self._style_part(items)[1],
            self._diversity_part(items)[1],
        ):
            codes.extend(part)
        return color_score, codes
