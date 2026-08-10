import json
from typing import List
from api.schemas import WardrobeItemFull


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
    def __init__(self, rules_path: str = "rules/fashion_rules.json"):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)
            
    def validate_outfit(self, items: List[WardrobeItemFull], occasion: str = None, season: str = None) -> bool:
        item_types = [normalise_type(item.type) for item in items]

        occasion_key = normalise_label(occasion)

        # Sport-tagged items are deliberately gym-only.  This is evaluated
        # independently of the JSON rule file so a newly added occasion (such as
        # Travel) or an occasion missing from the rules can never accidentally
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
                    
        return True
