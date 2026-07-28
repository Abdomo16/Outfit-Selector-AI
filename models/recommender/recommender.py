from typing import List, Tuple
from api.schemas import WardrobeItemFull, OutfitItem
from models.recommender.rule_engine import RuleEngine
from models.recommender.similarity import cosine_similarity
import random

class Recommender:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()

    def _generate_candidates(self, wardrobe: List[WardrobeItemFull]) -> List[List[WardrobeItemFull]]:
        # A simple wardrobe grammar
        tops = [i for i in wardrobe if i.type.lower() in ["t-shirt", "hoodie", "dress_shirt", "blouse", "wool_sweater", "polo", "tank_top", "tshirts", "tshirt", "shirt", "top", "sweater"]]
        bottoms = [i for i in wardrobe if i.type.lower() in ["jeans", "trousers", "chinos", "shorts", "skirt", "leggings", "pant", "pants", "joggers", "jean", "trouser"]]
        shoes = [i for i in wardrobe if i.type.lower() in ["sneakers", "oxford_shoes", "flip_flops", "loafers", "shoes", "heels", "boots", "sandals", "shoe", "boot", "sneaker", "sandal"]]
        outerwear = [i for i in wardrobe if i.type.lower() in ["blazer", "suit", "heavy_coat", "outwear", "jacket", "coat"]]
        one_piece = [i for i in wardrobe if i.type.lower() in ["dress", "tracksuit", "tuxedo", "pyjama"]]
        
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
        """Score an outfit by pairwise cosine similarity + a completeness bonus.
        
        Single items are penalised (0.3 base) so they never beat a real combo.
        Multi-item outfits get a bonus for each extra item to reward completeness.
        """
        n = len(outfit)

        # Single item — penalise heavily so it only wins if nothing else passes rules
        if n <= 1:
            return 0.3

        # Pairwise cosine similarity
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                emb1 = outfit[i].embedding
                emb2 = outfit[j].embedding
                if emb1 and emb2:
                    total += cosine_similarity(emb1, emb2)
                else:
                    total += 0.5   # neutral fallback when embeddings are missing
                pairs += 1

        similarity_score = total / pairs if pairs > 0 else 0.5

        # Completeness bonus: +0.05 per item (encourages 3-piece > 2-piece)
        completeness_bonus = (n - 1) * 0.05

        return min(similarity_score + completeness_bonus, 1.0)

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
            
            # Add small random noise to prevent deterministic ties and allow slight variations
            score += random.uniform(0.0, 0.05)

            # 3. Prefer this candidate if:
            #    (a) it scores higher, OR
            #    (b) same score but has more items (completeness tie-break)
            if score > best_score or (
                score == best_score and len(candidate) > len(best_outfit)
            ):
                best_score = score
                best_outfit = candidate
                
        outfit_items = [
            OutfitItem(
                id=item.id,
                category=item.type,
                color=item.color,
                confidence=item.confidence,
                style=item.style,
                pattern=item.pattern,
                season=item.season,
                imageUrl=item.image_path
            ) 
            for item in best_outfit
        ]
        
        return outfit_items, max(best_score, 0.0)
