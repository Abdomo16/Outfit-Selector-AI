from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import random

from api.schemas import OutfitItem, ReasonCode, WardrobeItemFull
from models.taxonomy import category_for_type
from models.recommender.rule_engine import RuleEngine, is_sport_item, normalise_label, normalise_type
from models.recommender.scoring import OutfitScorer

TOPS = {
    "t_shirt", "hoodie", "dress_shirt", "blouse", "wool_sweater", "polo",
    "tank_top", "shirt", "top", "sweater", "jersey", "sweatshirt", "cardigan",
}
BOTTOMS = {"jeans", "trousers", "chinos", "shorts", "skirt", "leggings", "joggers"}
SHOES = {
    "sneakers", "oxford_shoes", "flip_flops", "loafers", "shoes", "heels",
    "boots", "sandals", "shoe", "boot", "sneaker", "sandal", "flats",
}
OUTERWEAR = {
    "blazer", "suit", "heavy_coat", "outerwear", "outwear", "jacket", "coat",
    "denim_jacket", "leather_jacket", "vest",
}
ONE_PIECE = {"dress", "tracksuit", "tuxedo", "pyjama", "kurta", "jumpsuit"}
BAGS = {"handbag", "backpack", "crossbody_bag", "tote_bag", "wallet", "bag"}
ACCESSORIES = {
    "cap", "hat", "sunglasses", "watch", "belt", "scarf", "tie", "socks",
    "jewelry", "accessory",
}
ZONES = {
    "top": TOPS,
    "bottom": BOTTOMS,
    "shoes": SHOES,
    "outerwear": OUTERWEAR,
    "one_piece": ONE_PIECE,
    "bag": BAGS,
    "accessory": ACCESSORIES,
}
CATEGORY_ZONES = {
    "tops": "top",
    "bottoms": "bottom",
    "shoes": "shoes",
    "outerwear": "outerwear",
    "dresses": "one_piece",
    "bags": "bag",
    "accessories": "accessory",
}
CORE_ZONES = {"top", "bottom", "shoes", "outerwear", "one_piece"}
EXTRA_ZONES = ("bag", "accessory")

ItemId = Union[int, str]


class RecommendResult(NamedTuple):
    outfit: List[OutfitItem]
    score: float
    color_score: float
    reasons: List[str]
    reason_codes: List[ReasonCode]
    # Zones with nothing fit for the occasion ("top", "bottom", "shoes").
    missing: List[str] = []
    # The occasion's rules were set aside to make any outfit at all.
    relaxed: bool = False


# Candidate outfits beyond this are pruned to the best-fitting pieces.
MAX_CANDIDATES = 20000
# Outfits this close to the best score are all fair picks, so Regenerate
# shows something new.
NEAR_BEST = 0.02

# English sentences for the reason codes; the app shows its own translations.
REASON_TEXT = {
    "weather": "Works in {weather} weather.",
    "season": "Works for {season}.",
    "style_relaxed": "Style filter relaxed so an outfit could be made.",
    "style_match": "Matches {style}.",
    "repeated_piece": "Repeated a piece because the wardrobe is small.",
    "color_pair": "{0} pairs with {1}.",
    "color_clash": "{0} clashes with {1}.",
    "color_neutral": "Some colors have no rule, so they stay neutral.",
    "tonal_outfit": "Shades of one color look put together.",
    "neutral_palette": "Neutral colors that always work together.",
    "bold_colors": "Several bold colors at once.",
    "formality_match": "Dressed right for the occasion.",
    "warm_enough_for_night": "The jacket covers the cool night ({detail}°).",
    "light_for_heat": "Light enough for {detail}°.",
    "warm_for_cold": "Warm enough for {detail}°.",
    "style_harmony": "Every piece shares one style.",
    "one_statement_pattern": "One patterned piece stands out.",
    "pattern_clash": "Two patterns compete.",
    "forgotten_piece": "Brings back a piece you haven't worn lately.",
    "relaxed_occasion": "Nothing fits the occasion fully, so this is the closest look.",
}


def _item_key(item_id: Optional[ItemId]) -> str:
    return "" if item_id is None else str(item_id)


def zone_of(item: WardrobeItemFull) -> Optional[str]:
    label = normalise_type(item.type)
    for name, types in ZONES.items():
        if label in types:
            return name
    category = item.category or category_for_type(item.type)
    return CATEGORY_ZONES.get(normalise_label(category))


def _reason_text(code: ReasonCode, weather: Optional[str], season: Optional[str], style: Optional[str]) -> str:
    template = REASON_TEXT.get(code.code, code.code)
    colors = list(code.colors) + ["", ""]
    return template.format(
        colors[0], colors[1],
        weather=weather or "", season=season or "", style=style or "",
        detail=code.detail or "",
    )


class Recommender:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()

    @staticmethod
    def _by_zone(wardrobe: List[WardrobeItemFull]) -> dict:
        zones = {name: [] for name in ZONES}
        for item in wardrobe:
            zone = zone_of(item)
            if zone is not None:
                zones[zone].append(item)
        return zones

    @staticmethod
    def _count(zones: dict, with_shoes: bool) -> int:
        bases = len(zones["top"]) * len(zones["bottom"]) + len(zones["one_piece"])
        shoes = len(zones["shoes"]) if with_shoes else 1
        return bases * (len(zones["outerwear"]) + 1) * max(1, shoes)

    def _prune(
        self,
        zones: dict,
        scorer: OutfitScorer,
        keep_keys: set,
        with_shoes: bool,
    ) -> dict:
        """Keep the pieces that best suit the occasion until the number of
        combinations is workable. Locked pieces always stay.

        Each iteration that is still above MAX_CANDIDATES tightens every
        limit by 25 % (the 0.75 multiplier) so the loop always terminates
        rather than getting stuck when no single zone is over its limit.
        A floor of 3 items per zone preserves variety even in tiny wardrobes.
        """
        limits = {"top": 14, "bottom": 12, "one_piece": 10, "outerwear": 6, "shoes": 8}
        while self._count(zones, with_shoes) > MAX_CANDIDATES and any(
            len(zones[z]) > max(3, limits[z]) for z in limits
        ):
            pruned = dict(zones)
            for zone, limit in limits.items():
                items = zones[zone]
                if len(items) <= limit:
                    continue
                ranked = sorted(
                    items,
                    key=lambda item: (_item_key(item.id) in keep_keys, scorer.item_fit(item), random.random()),
                    reverse=True,
                )
                pruned[zone] = ranked[:limit]
            zones = pruned
            limits = {z: max(3, int(n * 0.75)) for z, n in limits.items()}
        return zones

    def _generate_candidates(
        self,
        wardrobe: List[WardrobeItemFull],
        with_shoes: bool,
        scorer: OutfitScorer,
        keep_keys: set,
    ) -> List[List[WardrobeItemFull]]:
        zones = self._by_zone(wardrobe)
        if self._count(zones, with_shoes) > MAX_CANDIDATES:
            zones = self._prune(zones, scorer, keep_keys, with_shoes)
        tops, bottoms = zones["top"], zones["bottom"]
        outerwear, one_piece = zones["outerwear"], zones["one_piece"]
        shoe_options = [[s] for s in zones["shoes"]] if with_shoes else [[]]
        layer_options = [[]] + [[o] for o in outerwear]

        bases = [[t, b] for t in tops for b in bottoms] + [[o] for o in one_piece]
        return [
            base + layer + shoe
            for base in bases
            for layer in layer_options
            for shoe in shoe_options
        ]

    def _style_filtered(self, wardrobe: List[WardrobeItemFull], style: Optional[str]) -> List[WardrobeItemFull]:
        if not style:
            return wardrobe
        wanted = normalise_label(style)
        return [
            item for item in wardrobe
            if not item.style or normalise_label(item.style) == wanted
        ]

    def _passes_lock_and_swap(
        self,
        candidate: List[WardrobeItemFull],
        lock_ids: Sequence[str],
        swap_id: Optional[str],
        swapped_item: Optional[WardrobeItemFull],
    ) -> bool:
        keys = {_item_key(item.id) for item in candidate}
        if lock_ids and not set(lock_ids).issubset(keys):
            return False
        if swap_id and swap_id in keys:
            return False
        if swapped_item is not None:
            zone = zone_of(swapped_item)
            if zone is not None:
                filled = any(
                    zone_of(item) == zone and _item_key(item.id) != swap_id
                    for item in candidate
                )
                if not filled:
                    return False
        return True

    def _collect(
        self,
        wardrobe: List[WardrobeItemFull],
        occasion: Optional[str],
        season: Optional[str],
        weather: Optional[str],
        lock_ids: Sequence[str],
        swap_id: Optional[str],
        swapped_item: Optional[WardrobeItemFull],
        scorer: OutfitScorer,
    ) -> Tuple[List[Tuple[float, List[WardrobeItemFull]]], bool]:
        """Scored outfits that pass the rules, and whether the occasion's
        rules had to be set aside to find any."""
        def keep(candidate: List[WardrobeItemFull], occasion_key: Optional[str]) -> bool:
            if not self._passes_lock_and_swap(candidate, lock_ids, swap_id, swapped_item):
                return False
            if any(scorer.too_warm(item) and _item_key(item.id) not in lock_ids for item in candidate):
                return False
            return self.rule_engine.validate_outfit(candidate, occasion_key, season, weather)

        def score(
            candidates: List[List[WardrobeItemFull]],
            strict: bool,
        ) -> List[Tuple[float, List[WardrobeItemFull]]]:
            scored = []
            for candidate in candidates:
                if strict:
                    if not keep(candidate, occasion):
                        continue
                else:
                    if normalise_label(occasion) != "gym" and any(is_sport_item(item) for item in candidate):
                        continue
                    if not keep(candidate, None):
                        continue
                scored.append((scorer.score(candidate).total, candidate))
            return scored

        # Shoes complete a look, so shoeless outfits are only a fallback.
        has_shoes = any(zone_of(item) == "shoes" for item in wardrobe)
        shoe_modes = (True, False) if has_shoes else (False,)
        keep_keys = set(lock_ids)
        candidate_sets = {
            with_shoes: self._generate_candidates(wardrobe, with_shoes, scorer, keep_keys)
            for with_shoes in shoe_modes
        }

        # Two-pass: first try with the occasion's rules enforced (strict=True);
        # if nothing survives, relax the occasion filter and try again so the
        # user always gets an outfit even with a sparse wardrobe.
        for strict in ((True, False) if occasion else (True,)):
            for with_shoes in shoe_modes:
                scored = score(candidate_sets[with_shoes], strict)
                if scored:
                    return scored, not strict
        return [], False

    def _missing(
        self,
        wardrobe: List[WardrobeItemFull],
        occasion: Optional[str],
        season: Optional[str],
        weather: Optional[str],
        scorer: OutfitScorer,
    ) -> List[str]:
        """The zones with no piece fit for the occasion, season and weather."""
        def fits(item: WardrobeItemFull) -> bool:
            return not scorer.too_warm(item) and self.rule_engine.validate_outfit(
                [item], occasion, season, weather,
            )

        zones = self._by_zone([item for item in wardrobe if fits(item)])
        missing = []
        if not zones["one_piece"]:
            if not zones["top"]:
                missing.append("top")
            if not zones["bottom"]:
                missing.append("bottom")
        if not zones["shoes"]:
            missing.append("shoes")
        return missing

    @staticmethod
    def _pick(
        scored: List[Tuple[float, List[WardrobeItemFull]]],
        previous: set,
    ) -> List[WardrobeItemFull]:
        """A random pick among the outfits near the best score, other than
        the one the user just saw when there is a choice.

        NEAR_BEST keeps outfits that are only marginally worse in the pool so
        "Regenerate" actually shows something different rather than bouncing
        between the same two combos. When every near-best outfit is the
        previous one (tiny wardrobe), the best outfit from the full ranked
        list is used as a fallback.
        """
        best = max(score for score, _ in scored)
        near = [outfit for score, outfit in scored if score >= best - NEAR_BEST]
        if previous:
            fresh = [
                outfit for outfit in near
                if {_item_key(item.id) for item in outfit} != previous
            ]
            if not fresh:
                fresh = [
                    outfit for _, outfit in sorted(scored, key=lambda row: row[0], reverse=True)
                    if {_item_key(item.id) for item in outfit} != previous
                ][:1]
            near = fresh or near
        return random.choice(near)

    def _attach_extras(
        self,
        outfit: List[WardrobeItemFull],
        pool: List[WardrobeItemFull],
        occasion: Optional[str],
        season: Optional[str],
        weather: Optional[str],
        lock_ids: Sequence[str],
        swap_id: Optional[str],
    ) -> List[WardrobeItemFull]:
        """Add at most one bag and one accessory that fit the core outfit."""
        result = list(outfit)
        zones = self._by_zone(pool)
        for zone in EXTRA_ZONES:
            options = [item for item in zones[zone] if _item_key(item.id) != swap_id]
            locked = [item for item in options if _item_key(item.id) in lock_ids]
            if locked:
                result.append(locked[0])
                continue

            best: List[Tuple[float, WardrobeItemFull]] = []
            for item in options:
                if not self.rule_engine.validate_outfit([item], occasion, season, weather):
                    continue
                if any(
                    code == "color_clash"
                    for other in result
                    for code, _ in self.rule_engine.score_colors([other, item])[2]
                ):
                    continue
                best.append((self.rule_engine.score_colors(result + [item])[0], item))
            if not best:
                continue
            top_score = max(score for score, _ in best)
            result.append(random.choice([item for score, item in best if score == top_score]))
        return result

    def _reasons(
        self,
        outfit: List[WardrobeItemFull],
        scorer: OutfitScorer,
        season: Optional[str],
        weather: Optional[str],
        style: Optional[str],
        style_relaxed: bool,
        repeated: bool,
        relaxed: bool,
    ) -> Tuple[float, List[str], List[ReasonCode]]:
        color_score, scored_codes = scorer.reasons(outfit)
        codes: List[ReasonCode] = []
        if relaxed:
            codes.append(ReasonCode(code="relaxed_occasion"))
        warmth_codes = {"warm_enough_for_night", "light_for_heat", "warm_for_cold"}
        if weather and not any(code.code in warmth_codes for code in scored_codes):
            codes.append(ReasonCode(code="weather"))
        if season:
            codes.append(ReasonCode(code="season"))
        if style_relaxed:
            codes.append(ReasonCode(code="style_relaxed"))
        elif style:
            codes.append(ReasonCode(code="style_match"))
        if repeated:
            codes.append(ReasonCode(code="repeated_piece"))
        # The specific reasons first; the general weather/season ones last.
        general = [code for code in codes if code.code in {"weather", "season"}]
        codes = [code for code in codes if code not in general] + scored_codes + general
        reasons = [_reason_text(code, weather, season, style) for code in codes]
        return color_score, reasons, codes

    def recommend(
        self,
        wardrobe: List[WardrobeItemFull],
        occasion: str = None,
        season: str = None,
        weather: str = None,
        style: str = None,
        lock_ids: Optional[Sequence[ItemId]] = None,
        swap_id: Optional[ItemId] = None,
        exclude_ids: Optional[Sequence[ItemId]] = None,
        layer: bool = False,
        temp_high: Optional[float] = None,
        temp_low: Optional[float] = None,
        last_worn: Optional[Dict[str, str]] = None,
        previous_ids: Optional[Sequence[ItemId]] = None,
    ) -> RecommendResult:
        locks = [_item_key(item_id) for item_id in (lock_ids or []) if _item_key(item_id)]
        swap_key = _item_key(swap_id) or None
        if swap_key:
            locks = [item_id for item_id in locks if item_id != swap_key]
        swapped_item = next((item for item in wardrobe if _item_key(item.id) == swap_key), None) if swap_key else None
        excluded = {
            _item_key(item_id)
            for item_id in (exclude_ids or [])
            if _item_key(item_id) and _item_key(item_id) not in locks
        }
        previous = {_item_key(item_id) for item_id in (previous_ids or []) if _item_key(item_id)}

        scorer = OutfitScorer(
            zone_of,
            self.rule_engine.rules.get("color_rules", {}),
            occasion=occasion,
            weather=weather,
            season=season,
            temp_high=temp_high,
            temp_low=temp_low,
            layer=layer,
            last_worn=last_worn,
            previous_ids=previous,
        )
        by_key = {_item_key(item.id): item for item in wardrobe}
        core_locks = [key for key in locks if key in by_key and zone_of(by_key[key]) in CORE_ZONES]
        core_swapped = swapped_item if swapped_item is not None and zone_of(swapped_item) in CORE_ZONES else None

        def score_wardrobe(source: List[WardrobeItemFull]):
            styled = self._style_filtered(source, style)
            found, relaxed = self._collect(styled, occasion, season, weather, core_locks, swap_key, core_swapped, scorer)
            if found or not style:
                return found, relaxed, False, styled
            found, relaxed = self._collect(source, occasion, season, weather, core_locks, swap_key, core_swapped, scorer)
            return found, relaxed, bool(found), source

        active = [item for item in wardrobe if _item_key(item.id) not in excluded]
        scored, relaxed, style_relaxed, extras_pool = score_wardrobe(active)
        repeated = False
        if not scored and excluded:
            scored, relaxed, style_relaxed, extras_pool = score_wardrobe(wardrobe)
            repeated = bool(scored)

        missing = self._missing(wardrobe, occasion, season, weather, scorer)
        if not scored:
            return RecommendResult([], 0.0, 0.0, [], [], missing, False)

        core_outfit = self._pick(scored, previous)
        best_outfit = self._attach_extras(
            core_outfit, extras_pool, occasion, season, weather, locks, swap_key,
        )
        total = scorer.score(best_outfit).total
        color_score, reasons, reason_codes = self._reasons(
            best_outfit, scorer, season, weather, style, style_relaxed, repeated, relaxed,
        )
        if not relaxed:
            # Only what the outfit itself lacks, such as shoes.
            worn_zones = {zone_of(item) for item in best_outfit}
            missing = [zone for zone in missing if zone not in worn_zones]

        outfit_items = [
            OutfitItem(
                id=item.id,
                category=item.type,
                wardrobeCategory=item.category or category_for_type(item.type),
                color=item.color,
                confidence=item.confidence,
                style=item.style,
                pattern=item.pattern,
                season=item.season,
                imageUrl=item.image_path,
            )
            for item in best_outfit
        ]
        return RecommendResult(
            outfit_items, round(total, 3), round(color_score, 3), reasons, reason_codes,
            missing, relaxed,
        )
