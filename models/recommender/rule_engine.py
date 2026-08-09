import json
from typing import List
from api.schemas import WardrobeItemFull

class RuleEngine:
    def __init__(self, rules_path: str = "rules/fashion_rules.json"):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)
            
    def validate_outfit(self, items: List[WardrobeItemFull], occasion: str = None, season: str = None) -> bool:
        item_types = [item.type.lower() for item in items]
        
        # 1. Occasion Check
        if occasion:
            occasion_key = occasion.lower().strip().replace(" ", "_")
            if occasion_key in self.rules.get("occasion_rules", {}):
                occ_rule = self.rules["occasion_rules"][occasion_key]
                forbidden_types   = occ_rule.get("forbidden_types", [])
                forbidden_styles  = occ_rule.get("forbidden_styles", [])
                # Per-item CLIP-predicted occasions that are incompatible with this occasion
                forbidden_item_occasions = [o.lower() for o in occ_rule.get("forbidden_item_occasions", [])]

                for item in items:
                    t = item.type.lower()
                    s = item.style.lower() if item.style else ""
                    # CLIP-predicted occasion tag on the item itself (e.g. "Gym", "Work")
                    item_occ = item.occasion.lower().strip().replace(" ", "_") if item.occasion else ""

                    if t in forbidden_types:
                        return False
                    if s and s in forbidden_styles:
                        return False
                    if item_occ and item_occ in forbidden_item_occasions:
                        return False
                
        # 2. Season Check
        if season and season in self.rules.get("season_rules", {}):
            sea_rule = self.rules["season_rules"][season]
            forbidden = sea_rule.get("forbidden_types", [])
            for t in item_types:
                if t in forbidden:
                    return False
                    
        return True
