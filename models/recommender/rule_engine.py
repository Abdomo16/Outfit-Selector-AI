import json
import os
from pathlib import Path
from typing import List
from api.schemas import WardrobeItemFull
from models.colors import family as color_family


# The classifier labels contain spaces and plurals (for example ``Sports Shoes``
# and ``Tshirts``), while the rule file uses snake-case labels.  Keeping the
# conversion in one place prevents a rule from silently being skipped.
TYPE_ALIASES = {
    "tshirts": "t-shirt",
    "tshirt": "t-shirt",
    "sports_tshirts": "t-shirt",
    "sports_tshirt": "t-shirt",
    "athletic_tshirt": "t-shirt",
    "shirts": "shirt",
    "tops": "top",
    "pants": "trousers",
    "pant": "trousers",
    "jean": "jeans",
    "kurtas": "kurta",
    "casual_shoes": "sneakers",
    "sports_shoes": "sneakers",
    "formal_shoes": "oxford_shoes",
    "flip_flop": "flip_flops",
    "puffer_jacket": "heavy_coat",
}


def normalise_label(value: str | None) -> str:
    """Return the form used by the recommendation rules."""
    if not value:
        return ""
    return "_".join(value.lower().strip().replace("-", " ").split())


def normalise_type(value: str | None) -> str:
    if not value:
        return ""
    label = normalise_label(value)
    canonical = TYPE_ALIASES.get(label, label)
    return canonical.replace("-", "_")


def is_sport_item(item: WardrobeItemFull) -> bool:
    """Identify sportwear without relying on one classifier field alone."""
    raw_type = normalise_label(item.type)
    return (
        normalise_label(item.style) in {"sport", "sports", "athletic", "sporty"}
        or normalise_label(item.occasion) == "gym"
        or raw_type.startswith(("sport_", "sports_", "athletic_", "sporty_"))
    )


class RuleEngine:
    def __init__(self, rules_path: str = None):
        base_dir = Path(__file__).resolve().parent.parent.parent
        rules_path = rules_path or os.environ.get("FASHION_RULES_PATH", str(base_dir / "rules" / "fashion_rules.json"))
        with open(rules_path, "r") as f:
            self.rules = json.load(f)
            
    def validate_outfit(
        self,
        items: List[WardrobeItemFull],
        occasion: str = None,
        season: str = None,
        weather: str = None,
    ) -> bool:
        item_types = [normalise_type(item.type) for item in items]

        occasion_key = normalise_label(occasion)

        
        # recommend gym wear.
        if occasion_key and occasion_key != "gym":
            for item in items:
                if is_sport_item(item):
                    return False

        # 1. Occasion Check
        if occasion_key and occasion_key in self.rules.get("occasion_rules", {}):
            occ_rule = self.rules["occasion_rules"][occasion_key]
            forbidden_types = {normalise_type(value) for value in occ_rule.get("forbidden_types", [])}
            forbidden_styles = {normalise_label(value) for value in occ_rule.get("forbidden_styles", [])}
            # Per-item CLIP-predicted occasions that are incompatible with this occasion
            forbidden_item_occasions = {
                normalise_label(value) for value in occ_rule.get("forbidden_item_occasions", [])
            }

            for item in items:
                t = normalise_type(item.type)
                s = normalise_label(item.style)
                # CLIP-predicted occasion tag on the item itself (e.g. "Gym", "Work")
                item_occ = normalise_label(item.occasion)

                if t in forbidden_types:
                    return False
                if s and s in forbidden_styles:
                    return False
                if item_occ and item_occ in forbidden_item_occasions:
                    return False
                
        # 2. Season Check
        season_key = normalise_label(season)
        if season_key and season_key in self.rules.get("season_rules", {}):
            sea_rule = self.rules["season_rules"][season_key]
            forbidden = {normalise_type(value) for value in sea_rule.get("forbidden_types", [])}
            for t in item_types:
                if t in forbidden:
                    return False

        # 3. Live weather. "everyday" has no occasion list, so this still applies.
        weather_key = normalise_label(weather)
        if weather_key and weather_key in self.rules.get("weather_rules", {}):
            weather_rule = self.rules["weather_rules"][weather_key]
            forbidden = {normalise_type(value) for value in weather_rule.get("forbidden_types", [])}
            for t in item_types:
                if t in forbidden:
                    return False

        return True

    def score_colors(self, items: List[WardrobeItemFull]) -> tuple[float, list[str], list[tuple[str, list[str]]]]:
        """Average pairwise color compatibility. Unknown colors score 0.5."""
        rules = self.rules.get("color_rules", {})
        if len(items) < 2:
            return 0.5, []

        scores: list[float] = []
        reasons: list[str] = []
        codes: list[tuple[str, list[str]]] = []
        unknown = False

        def allows(rule: list, other: str) -> bool:
            return "matches_all" in rule or other in rule

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left = items[i].color or ""
                right = items[j].color or ""
                left_key = color_family(left)
                right_key = color_family(right)
                left_rule = rules.get(left_key)
                right_rule = rules.get(right_key)
                if left_rule is None or right_rule is None:
                    scores.append(0.5)
                    unknown = True
                    continue
                if allows(left_rule, right_key) or allows(right_rule, left_key):
                    scores.append(1.0)
                    reasons.append(f"{left} pairs with {right}.")
                    codes.append(("color_pair", [left, right]))
                else:
                    scores.append(0.0)
                    reasons.append(f"{left} clashes with {right}.")
                    codes.append(("color_clash", [left, right]))

        if unknown:
            reasons.append("Some colors have no rule, so they stay neutral.")
            codes.append(("color_neutral", []))
        return sum(scores) / len(scores), reasons, codes
