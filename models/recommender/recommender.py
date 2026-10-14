from typing import List, NamedTuple, Optional, Sequence, Tuple, Union
import random

from api.schemas import OutfitItem, WardrobeItemFull
from models.recommender.rule_engine import RuleEngine, is_sport_item, normalise_label, normalise_type

TOPS = {
    "t_shirt", "hoodie", "dress_shirt", "blouse", "wool_sweater", "polo",
    "tank_top", "shirt", "top", "sweater", "jersey",
}
BOTTOMS = {"jeans", "trousers", "chinos", "shorts", "skirt", "leggings", "joggers"}
SHOES = {
    "sneakers", "oxford_shoes", "flip_flops", "loafers", "shoes", "heels",
    "boots", "sandals", "shoe", "boot", "sneaker", "sandal", "flats",
}
OUTERWEAR = {"blazer", "suit", "heavy_coat", "outerwear", "outwear", "jacket", "coat"}
ONE_PIECE = {"dress", "tracksuit", "tuxedo", "pyjama", "kurta"}
ZONES = {
    "top": TOPS,
    "bottom": BOTTOMS,
    "shoes": SHOES,
    "outerwear": OUTERWEAR,
    "one_piece": ONE_PIECE,
}

ItemId = Union[int, str]


class RecommendResult(NamedTuple):
    outfit: List[OutfitItem]
    score: float
    color_score: float
    reasons: List[str]


def _item_key(item_id: Optional[ItemId]) -> str:
    return "" if item_id is None else str(item_id)


def _zone(item: WardrobeItemFull) -> Optional[str]:
    label = normalise_type(item.type)
    for name, types in ZONES.items():
        if label in types:
            return name
    return None


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

    def _generate_candidates(self, wardrobe: List[WardrobeItemFull]) -> List[List[WardrobeItemFull]]:
        tops = [i for i in wardrobe if normalise_type(i.type) in TOPS]
        bottoms = [i for i in wardrobe if normalise_type(i.type) in BOTTOMS]
        shoes = [i for i in wardrobe if normalise_type(i.type) in SHOES]
        outerwear = [i for i in wardrobe if normalise_type(i.type) in OUTERWEAR]
        one_piece = [i for i in wardrobe if normalise_type(i.type) in ONE_PIECE]

        candidates = []

        if tops and bottoms:
            for t in tops:
                for b in bottoms:
                    combos = [[t, b]]
                    if outerwear:
                        combos.extend([[t, b, o] for o in outerwear])
                    if shoes:
                        combos.extend([[t, b, s] for s in shoes])
                        if outerwear:
                            combos.extend([[t, b, o, s] for o in outerwear for s in shoes])
                    candidates.extend(combos)

        if one_piece:
            for o in one_piece:
                combos = [[o]]
                if outerwear:
                    combos.extend([[o, out] for out in outerwear])
                if shoes:
                    combos.extend([[o, s] for s in shoes])
                    if outerwear:
                        combos.extend([[o, out, s] for out in outerwear for s in shoes])
                candidates.extend(combos)

        return candidates

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
        candidates = self._generate_candidates(wardrobe)
        random.shuffle(candidates)
        scored: List[Tuple[float, List[WardrobeItemFull]]] = []

        def keep(candidate: List[WardrobeItemFull], occasion_key: Optional[str]) -> bool:
            if not self._passes_lock_and_swap(candidate, lock_ids, swap_id, swapped_item):
                return False
            return self.rule_engine.validate_outfit(candidate, occasion_key, season, weather)

        for candidate in candidates:
            if not keep(candidate, occasion):
                continue
            color_score, _ = self.rule_engine.score_colors(candidate)
            scored.append((color_score, candidate))

        if not scored and occasion:
            for candidate in candidates:
                if normalise_label(occasion) != "gym" and any(is_sport_item(item) for item in candidate):
                    continue
                if not keep(candidate, None):
                    continue
                color_score, _ = self.rule_engine.score_colors(candidate)
                scored.append((color_score, candidate))

        return scored

    def _reasons(
        self,
        outfit: List[WardrobeItemFull],
        season: Optional[str],
        weather: Optional[str],
        style: Optional[str],
        style_relaxed: bool,
        color_reasons: List[str],
    ) -> List[str]:
        reasons: List[str] = []
        if weather:
            reasons.append(f"Works in {weather} weather.")
        if season:
            reasons.append(f"Works for {season}.")
        if style_relaxed:
            reasons.append("Style filter relaxed so an outfit could be made.")
        elif style:
            reasons.append(f"Matches {style}.")
        reasons.extend(color_reasons[:3])
        if any(reason.startswith("Some colors") for reason in color_reasons):
            neutral = "Some colors have no rule, so they stay neutral."
            if neutral not in reasons:
                reasons.append(neutral)
        return reasons

    def recommend(
        self,
        wardrobe: List[WardrobeItemFull],
        occasion: str = None,
        season: str = None,
        weather: str = None,
        style: str = None,
        lock_ids: Optional[Sequence[ItemId]] = None,
        swap_id: Optional[ItemId] = None,
    ) -> RecommendResult:
        locks = [_item_key(item_id) for item_id in (lock_ids or []) if _item_key(item_id)]
        swap_key = _item_key(swap_id) or None
        if swap_key:
            locks = [item_id for item_id in locks if item_id != swap_key]
        swapped_item = next((item for item in wardrobe if _item_key(item.id) == swap_key), None) if swap_key else None

        styled = self._style_filtered(wardrobe, style)
        scored = self._collect(styled, occasion, season, weather, locks, swap_key, swapped_item)
        style_relaxed = False
        if not scored and style:
            scored = self._collect(wardrobe, occasion, season, weather, locks, swap_key, swapped_item)
            style_relaxed = bool(scored)

        if not scored:
            return RecommendResult([], 0.0, 0.0, [])

        scored.sort(key=lambda row: (row[0], len(row[1])), reverse=True)
        pool = scored
        best_score = pool[0][0]
        best_len = len(pool[0][1])
        top_candidates = [
            row for row in pool
            if row[0] == best_score and len(row[1]) == best_len
        ]
        _, best_outfit = self._pick_outfit(top_candidates)
        color_score, color_reasons = self.rule_engine.score_colors(best_outfit)
        reasons = self._reasons(best_outfit, season, weather, style, style_relaxed, color_reasons)

        outfit_items = [
            OutfitItem(
                id=item.id,
                category=item.type,
                color=item.color,
                confidence=item.confidence,
                style=item.style,
                pattern=item.pattern,
                season=item.season,
                imageUrl=item.image_path,
            )
            for item in best_outfit
        ]
        return RecommendResult(outfit_items, color_score, color_score, reasons)
