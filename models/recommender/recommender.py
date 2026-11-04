from typing import List, NamedTuple, Optional, Sequence, Tuple, Union
import random

from api.schemas import OutfitItem, ReasonCode, WardrobeItemFull
from models.taxonomy import category_for_type
from models.recommender.rule_engine import RuleEngine, is_sport_item, normalise_label, normalise_type

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


def _item_key(item_id: Optional[ItemId]) -> str:
    return "" if item_id is None else str(item_id)


def _zone(item: WardrobeItemFull) -> Optional[str]:
    label = normalise_type(item.type)
    for name, types in ZONES.items():
        if label in types:
            return name
    category = item.category or category_for_type(item.type)
    return CATEGORY_ZONES.get(normalise_label(category))


class Recommender:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()
        self._last_outfit_ids = None

    @staticmethod
    def _outfit_signature(outfit: List[WardrobeItemFull]) -> frozenset:
        return frozenset(item.id for item in outfit if item.id is not None)

    def _pick_outfit(self, top_candidates: List[Tuple[float, List[WardrobeItemFull]]]):
        """
        Pick a top candidate, preferring one that differs from the last pick
        so 'Regenerate' shows something new when alternatives exist.
        """
        if len(top_candidates) <= 1:
            chosen = top_candidates[0]
            self._last_outfit_ids = self._outfit_signature(chosen[1])
            return chosen

        distinct = [
            s for s in top_candidates
            if self._outfit_signature(s[1]) != self._last_outfit_ids
        ]
        if not distinct:
            distinct = top_candidates

        chosen = random.choice(distinct)
        self._last_outfit_ids = self._outfit_signature(chosen[1])
        return chosen

    @staticmethod
    def _by_zone(wardrobe: List[WardrobeItemFull]) -> dict:
        zones = {name: [] for name in ZONES}
        for item in wardrobe:
            zone = _zone(item)
            if zone is not None:
                zones[zone].append(item)
        return zones

    def _generate_candidates(
        self,
        wardrobe: List[WardrobeItemFull],
        with_shoes: bool,
    ) -> List[List[WardrobeItemFull]]:
        zones = self._by_zone(wardrobe)
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
            zone = _zone(swapped_item)
            if zone is not None:
                filled = any(
                    _zone(item) == zone and _item_key(item.id) != swap_id
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
    ) -> List[Tuple[float, List[WardrobeItemFull]]]:
        def keep(candidate: List[WardrobeItemFull], occasion_key: Optional[str]) -> bool:
            if not self._passes_lock_and_swap(candidate, lock_ids, swap_id, swapped_item):
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
                color_score, _, _ = self.rule_engine.score_colors(candidate)
                scored.append((color_score, candidate))
            return scored

        # Shoes complete a look, so shoeless outfits are only a fallback.
        has_shoes = any(_zone(item) == "shoes" for item in wardrobe)
        shoe_modes = (True, False) if has_shoes else (False,)
        candidate_sets = {}
        for with_shoes in shoe_modes:
            candidates = self._generate_candidates(wardrobe, with_shoes)
            random.shuffle(candidates)
            candidate_sets[with_shoes] = candidates

        for strict in ((True, False) if occasion else (True,)):
            for with_shoes in shoe_modes:
                scored = score(candidate_sets[with_shoes], strict)
                if scored:
                    return scored
        return []

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
        season: Optional[str],
        weather: Optional[str],
        style: Optional[str],
        style_relaxed: bool,
        color_reasons: List[str],
        color_codes: List[tuple],
        repeated: bool,
    ) -> tuple:
        reasons: List[str] = []
        codes: List[ReasonCode] = []
        if weather:
            reasons.append(f"Works in {weather} weather.")
            codes.append(ReasonCode(code="weather"))
        if season:
            reasons.append(f"Works for {season}.")
            codes.append(ReasonCode(code="season"))
        if style_relaxed:
            reasons.append("Style filter relaxed so an outfit could be made.")
            codes.append(ReasonCode(code="style_relaxed"))
        elif style:
            reasons.append(f"Matches {style}.")
            codes.append(ReasonCode(code="style_match"))
        if repeated:
            reasons.append("Repeated a piece because the wardrobe is small.")
            codes.append(ReasonCode(code="repeated_piece"))
        reasons.extend(color_reasons[:3])
        for code, colors in color_codes[:3]:
            codes.append(ReasonCode(code=code, colors=colors))
        if any(code == "color_neutral" for code, _ in color_codes):
            neutral = "Some colors have no rule, so they stay neutral."
            if neutral not in reasons:
                reasons.append(neutral)
            if not any(item.code == "color_neutral" for item in codes):
                codes.append(ReasonCode(code="color_neutral"))
        return reasons, codes

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

        by_key = {_item_key(item.id): item for item in wardrobe}
        core_locks = [key for key in locks if key in by_key and _zone(by_key[key]) in CORE_ZONES]
        core_swapped = swapped_item if swapped_item is not None and _zone(swapped_item) in CORE_ZONES else None

        def score_wardrobe(source: List[WardrobeItemFull]):
            styled = self._style_filtered(source, style)
            found = self._collect(styled, occasion, season, weather, core_locks, swap_key, core_swapped)
            if found or not style:
                return found, False, styled
            found = self._collect(source, occasion, season, weather, core_locks, swap_key, core_swapped)
            return found, bool(found), source

        active = [item for item in wardrobe if _item_key(item.id) not in excluded]
        scored, style_relaxed, extras_pool = score_wardrobe(active)
        repeated = False
        if not scored and excluded:
            scored, style_relaxed, extras_pool = score_wardrobe(wardrobe)
            repeated = bool(scored)

        if not scored:
            return RecommendResult([], 0.0, 0.0, [], [])

        best_score = max(row[0] for row in scored)
        top_candidates = [row for row in scored if row[0] == best_score]
        _, core_outfit = self._pick_outfit(top_candidates)
        best_outfit = self._attach_extras(
            core_outfit, extras_pool, occasion, season, weather, locks, swap_key,
        )
        color_score, color_reasons, color_codes = self.rule_engine.score_colors(best_outfit)
        reasons, reason_codes = self._reasons(
            best_outfit, season, weather, style, style_relaxed, color_reasons, color_codes, repeated,
        )

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
        return RecommendResult(outfit_items, color_score, color_score, reasons, reason_codes)
