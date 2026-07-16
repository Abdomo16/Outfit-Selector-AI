from typing import List, Tuple
from api.schemas import WardrobeItemFull, OutfitItem
from models.recommender.rule_engine import RuleEngine
from models.recommender.similarity import cosine_similarity

class Recommender:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()

    def _generate_candidates(self, wardrobe: List[WardrobeItemFull]) -> List[List[WardrobeItemFull]]:
        # A simple wardrobe grammar
        tops = [i for i in wardrobe if i.type in ["t-shirt", "hoodie", "dress_shirt", "blouse", "wool_sweater", "polo", "tank_top"]]
        bottoms = [i for i in wardrobe if i.type in ["jeans", "trousers", "chinos", "shorts", "skirt", "leggings"]]
        shoes = [i for i in wardrobe if i.type in ["sneakers", "oxford_shoes", "flip_flops", "loafers", "shoes", "heels", "boots", "sandals"]]
        outerwear = [i for i in wardrobe if i.type in ["blazer", "suit", "heavy_coat", "outwear"]]
        one_piece = [i for i in wardrobe if i.type in ["dress", "tracksuit", "tuxedo", "pyjama"]]
        
        candidates = []
        
        # combinations: top + bottom + optional outerwear + optional shoes
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
                    
        # combinations: one-piece + optional shoes + optional outerwear
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

    def _score_outfit(self, outfit: List[WardrobeItemFull]) -> float:
        # Simple scoring: average cosine similarity between all pairs of items in the outfit
        score = 0.0
        pairs = 0
        
        if len(outfit) <= 1:
            return 1.0 # default score for single items
            
        for i in range(len(outfit)):
            for j in range(i+1, len(outfit)):
                emb1 = outfit[i].embedding
                emb2 = outfit[j].embedding
                if emb1 and emb2:
                    score += cosine_similarity(emb1, emb2)
                else:
                    score += 0.5
                pairs += 1
                
        if pairs > 0:
            return score / pairs
        return 0.5

    def recommend(self, wardrobe: List[WardrobeItemFull], occasion: str = None, season: str = None) -> Tuple[List[OutfitItem], float]:
        candidates = self._generate_candidates(wardrobe)
        
        best_outfit = []
        best_score = -1.0
        
        for candidate in candidates:
            # 1. Filter by rules
            if not self.rule_engine.validate_outfit(candidate, occasion, season):
                continue
                
            # 2. Score outfit
            score = self._score_outfit(candidate)
            
            # 3. Update best
            if score > best_score:
                best_score = score
                best_outfit = candidate
                
        outfit_items = [
            OutfitItem(id=item.id, type=item.type, color=item.color) 
            for item in best_outfit
        ]
        
        return outfit_items, max(best_score, 0.0)
