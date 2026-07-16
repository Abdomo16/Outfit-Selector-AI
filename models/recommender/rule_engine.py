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
        if occasion and occasion in self.rules.get("occasion_rules", {}):
            occ_rule = self.rules["occasion_rules"][occasion]
            forbidden = occ_rule.get("forbidden_types", [])
            for t in item_types:
                if t in forbidden:
                    return False
                
        # 2. Season Check
        if season and season in self.rules.get("season_rules", {}):
            sea_rule = self.rules["season_rules"][season]
            forbidden = sea_rule.get("forbidden_types", [])
            for t in item_types:
                if t in forbidden:
                    return False
                    
        return True
