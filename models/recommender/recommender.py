from typing import List, Tuple
from api.schemas import WardrobeItemFull, OutfitItem
from models.recommender.rule_engine import RuleEngine, normalise_type, normalise_label, is_sport_item
from models.recommender.similarity import cosine_similarity
import random

class Recommender:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()
        self._last_outfit_ids = None

    @staticmethod
    def _outfit_signature(outfit: List[WardrobeItemFull]) -> frozenset:
        return frozenset(item.id for item in outfit if item.id is not None)

    def _pick_outfit(self, top_candidates: List[Tuple[float, List[WardrobeItemFull]]]):
        """
        Fix: Resolves the issue where regenerating recommends the exact same item.
        Pick a top candidate, preferring one that differs from the last pick
        so 'Regenerate' actually shows something new when alternatives exist.
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
        # A simple wardrobe grammar.  Always compare canonical classifier labels;
        # the uploaded type is retained in the response for the client.
        tops = [i for i in wardrobe if normalise_type(i.type) in {"t_shirt", "hoodie", "dress_shirt", "blouse", "wool_sweater", "polo", "tank_top", "shirt", "top", "sweater", "jersey"}]
        bottoms = [i for i in wardrobe if normalise_type(i.type) in {"jeans", "trousers", "chinos", "shorts", "skirt", "leggings", "joggers"}]
        shoes = [i for i in wardrobe if normalise_type(i.type) in {"sneakers", "oxford_shoes", "flip_flops", "loafers", "shoes", "heels", "boots", "sandals", "shoe", "boot", "sneaker", "sandal", "flats"}]
        outerwear = [i for i in wardrobe if normalise_type(i.type) in {"blazer", "suit", "heavy_coat", "outerwear", "outwear", "jacket", "coat"}]
        one_piece = [i for i in wardrobe if normalise_type(i.type) in {"dress", "tracksuit", "tuxedo", "pyjama", "kurta"}]
        
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
        
        # Shuffle first so equal-scored candidates aren't always in the same order
        random.shuffle(candidates)

        scored: List[Tuple[float, List[WardrobeItemFull]]] = []

        for candidate in candidates:
            # 1. Filter by rules
            if not self.rule_engine.validate_outfit(candidate, occasion, season):
                continue

            # 2. Score outfit
            score = self._score_outfit(candidate)
            scored.append((score, candidate))
            
        # Fallback: if no candidates passed rules, try ignoring the occasion constraint
        if not scored and occasion:
            for candidate in candidates:
                # Gym/sport wear must stay gym-only even on the relaxed fallback
                # path, otherwise an empty result quietly leaks sporty items.
                if normalise_label(occasion) != "gym" and any(is_sport_item(item) for item in candidate):
                    continue
                if not self.rule_engine.validate_outfit(candidate, occasion=None, season=season):
                    continue
                score = self._score_outfit(candidate)
                # Penalise fallback candidates so they don't seem as good as proper ones
                scored.append((score - 0.2, candidate))

        if not scored:
            return [], 0.0

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Prefer complete outfits: only consider multi-item candidates if any exist,
        # otherwise single items (e.g. a dress) are the only option.
        multi = [s for s in scored if len(s[1]) > 1]
        pool = multi if multi else scored

        # Select randomly from top candidates to increase variety on Regenerate
        best_overall_score = pool[0][0]
        # Include any candidate that scores within 0.1 of the top score
        top_candidates = [s for s in pool if s[0] >= best_overall_score - 0.1]

        best_score, best_outfit = self._pick_outfit(top_candidates)

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
