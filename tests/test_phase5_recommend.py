import random

import pytest
from fastapi.testclient import TestClient
from api.main import app
from models.recommender.rule_engine import RuleEngine
from models.recommender.similarity import cosine_similarity
from models.recommender.recommender import Recommender
from api.schemas import WardrobeItemFull

client = TestClient(app)

def test_cosine_similarity():
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_rule_engine():
    engine = RuleEngine("rules/fashion_rules.json")
    item_formal = WardrobeItemFull(id=1, type="suit", confidence=0.9, color="black", hex="#000000")
    item_casual = WardrobeItemFull(id=2, type="sneakers", confidence=0.9, color="white", hex="#FFFFFF")
    
    # Formal should reject casual items like sneakers
    assert engine.validate_outfit([item_formal], occasion="formal") == True
    assert engine.validate_outfit([item_formal, item_casual], occasion="formal") == False
    
    item_winter = WardrobeItemFull(id=3, type="flip_flops", confidence=0.9, color="blue", hex="#0000FF")
    # Winter season rejects flip_flops
    assert engine.validate_outfit([item_winter], season="winter") == False

    # The ML classifier emits labels with spaces/plurals.  They must still be
    # subject to the occasion rules, including the Gym/Sport restrictions.
    sports_tee = WardrobeItemFull(id=4, type="Tshirts", confidence=0.9, color="black", hex="#000", style="Sport")
    gym_tagged_tee = WardrobeItemFull(id=5, type="Shirts", confidence=0.9, color="black", hex="#000", occasion="Gym")
    assert engine.validate_outfit([sports_tee], occasion="University") is False
    assert engine.validate_outfit([sports_tee], occasion="Work") is False
    assert engine.validate_outfit([gym_tagged_tee], occasion="University") is False
    assert engine.validate_outfit([sports_tee], occasion="Gym") is True
    assert engine.validate_outfit([sports_tee], occasion="Travel") is False
    named_sports_tee = WardrobeItemFull(id=6, type="Sports Tshirts", confidence=0.9, color="black", hex="#000")
    assert engine.validate_outfit([named_sports_tee], occasion="Work") is False

def test_recommender():
    engine = RuleEngine("rules/fashion_rules.json")
    recommender = Recommender(engine)
    
    wardrobe = [
        WardrobeItemFull(id=1, type="jeans", confidence=0.9, color="blue", hex="#000", embedding=[1.0, 0.0]),
        WardrobeItemFull(id=2, type="t-shirt", confidence=0.9, color="white", hex="#fff", embedding=[1.0, 0.1]),
        WardrobeItemFull(id=3, type="sneakers", confidence=0.9, color="white", hex="#fff", embedding=[0.0, 1.0]),
        WardrobeItemFull(id=4, type="suit", confidence=0.9, color="black", hex="#000", embedding=[0.5, 0.5])
    ]
    
    outfit, score = recommender.recommend(wardrobe, occasion="casual")[:2]
    assert len(outfit) > 0
    for item in outfit:
        assert item.category != "suit"


def test_recommender_accepts_classifier_labels():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        WardrobeItemFull(id=1, type="Tshirts", confidence=0.9, color="white", hex="#fff", style="Minimal"),
        WardrobeItemFull(id=2, type="Jeans", confidence=0.9, color="blue", hex="#00f"),
        WardrobeItemFull(id=3, type="Casual Shoes", confidence=0.9, color="white", hex="#fff"),
    ]
    outfit, _ = recommender.recommend(wardrobe, occasion="University")[:2]
    assert {item.id for item in outfit} >= {1, 2}

def test_recommend_endpoint():
    payload = {
        "wardrobe": [
            {
                "id": 1,
                "type": "jeans",
                "confidence": 0.99,
                "color": "blue",
                "hex": "#0000FF",
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "id": 2,
                "type": "t-shirt",
                "confidence": 0.99,
                "color": "white",
                "hex": "#FFFFFF",
                "embedding": [0.1, 0.25, 0.3]
            }
        ],
        "occasion": "casual",
        "season": "summer"
    }
    
    response = client.post("/recommend/", json=payload)
    if response.status_code != 200:
        print(response.json())
        assert False
    data = response.json()
    assert "outfit" in data
    assert "score" in data
    assert "color_score" in data
    assert "reasons" in data
    assert len(data["outfit"]) == 2


def _item(item_id, item_type, color, style=None, category=None):
    return WardrobeItemFull(
        id=item_id,
        type=item_type,
        category=category,
        confidence=0.9,
        color=color,
        hex="#000000",
        style=style,
    )


def test_shoes_are_included_when_the_wardrobe_has_them():
    random.seed(1)
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "t-shirt", "white"),
        _item(2, "jeans", "navy"),
        _item(3, "sneakers", "burgundy"),
    ]
    for _ in range(5):
        ids = {item.id for item in recommender.recommend(wardrobe, occasion="everyday").outfit}
        assert 3 in ids


def test_shoeless_outfit_when_no_shoe_passes_the_rules():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "trousers", "navy"),
        _item(3, "flip_flops", "black"),
    ]
    result = recommender.recommend(wardrobe, occasion="work")
    assert {item.id for item in result.outfit} == {1, 2}


def test_accessory_and_bag_are_added():
    random.seed(2)
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "t-shirt", "white"),
        _item(2, "jeans", "navy"),
        _item(3, "sneakers", "white"),
        _item(4, "watch", "black"),
        _item(5, "backpack", "grey"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday")
    assert {item.id for item in result.outfit} == {1, 2, 3, 4, 5}
    watch = next(item for item in result.outfit if item.id == 4)
    assert watch.wardrobeCategory == "Accessories"


def test_accessory_uses_category_when_type_is_unknown():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "jeans", "navy"),
        _item(3, "bracelet", "black", category="Accessories"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday")
    assert 3 in {item.id for item in result.outfit}


def test_clashing_accessory_is_skipped():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "blouse", "burgundy"),
        _item(2, "trousers", "navy"),
        _item(3, "scarf", "olive"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday")
    assert {item.id for item in result.outfit} == {1, 2}


def test_dress_only_wardrobe_does_not_crash():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    result = recommender.recommend([_item(1, "dress", "black")], occasion="party")
    assert [item.id for item in result.outfit] == [1]
    # One piece has nothing to match, so its colors score a neutral 0.8.
    assert result.color_score == 0.8


def test_lock_and_swap_work_on_accessories():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "jeans", "navy"),
        _item(3, "watch", "black"),
        _item(4, "belt", "brown"),
    ]
    locked = recommender.recommend(wardrobe, occasion="everyday", lock_ids=[4])
    ids = {item.id for item in locked.outfit}
    assert 4 in ids
    assert 3 not in ids

    swapped = recommender.recommend(wardrobe, occasion="everyday", lock_ids=[1, 3], swap_id=3)
    ids = {item.id for item in swapped.outfit}
    assert 3 not in ids
    assert {1, 4} <= ids


def test_weather_blocks_hot_wool_and_rain_sandals():
    engine = RuleEngine("rules/fashion_rules.json")
    wool = _item(1, "wool_sweater", "grey")
    tee = _item(2, "t-shirt", "white")
    sandals = _item(3, "sandals", "brown")
    assert engine.validate_outfit([wool, tee], weather="hot") is False
    assert engine.validate_outfit([sandals], weather="rain") is False
    assert engine.validate_outfit([tee], weather="hot") is True

    recommender = Recommender(engine)
    wardrobe = [
        wool,
        tee,
        _item(4, "jeans", "navy"),
        _item(5, "flip_flops", "black"),
        _item(6, "sneakers", "white"),
    ]
    hot = recommender.recommend(wardrobe, occasion="everyday", weather="hot", season="summer")
    assert 1 not in {item.id for item in hot.outfit}
    rainy = recommender.recommend(wardrobe, occasion="everyday", weather="rain", season="spring")
    assert 5 not in {item.id for item in rainy.outfit}
    assert 6 in {item.id for item in rainy.outfit}


def test_everyday_blocks_sport_and_skips_occasion_lists():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "t-shirt", "white", style="Sport"),
        _item(2, "shirt", "white", style="Minimal"),
        _item(3, "jeans", "navy"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday", weather="mild")
    ids = {item.id for item in result.outfit}
    assert 1 not in ids
    assert 2 in ids


def test_style_filter_drops_conflicts_then_relaxes():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "blouse", "white", style="Elegant"),
        _item(2, "trousers", "navy", style="Elegant"),
        _item(3, "hoodie", "black", style="Streetwear"),
        _item(4, "joggers", "grey", style="Streetwear"),
    ]
    matched = recommender.recommend(wardrobe, occasion="everyday", style="Elegant")
    assert {item.id for item in matched.outfit} <= {1, 2}
    assert any(reason == "Matches Elegant." for reason in matched.reasons)

    relaxed = recommender.recommend(
        [_item(3, "hoodie", "black", style="Streetwear"), _item(4, "joggers", "grey", style="Streetwear")],
        occasion="everyday",
        style="Elegant",
    )
    assert relaxed.outfit
    assert any("relaxed" in reason for reason in relaxed.reasons)


def test_lock_keeps_piece_and_swap_replaces_only_that_slot():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "t-shirt", "white"),
        _item(2, "shirt", "navy"),
        _item(3, "jeans", "navy"),
        _item(4, "sneakers", "white"),
        _item(5, "loafers", "brown"),
    ]
    locked = recommender.recommend(wardrobe, occasion="everyday", lock_ids=[2])
    assert 2 in {item.id for item in locked.outfit}

    swapped = recommender.recommend(wardrobe, occasion="everyday", swap_id=4, lock_ids=[2, 4])
    ids = {item.id for item in swapped.outfit}
    assert 4 not in ids
    assert 5 in ids
    assert 2 in ids


def test_color_score_and_reasons():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    matched = recommender.recommend(
        [_item(1, "blouse", "white"), _item(2, "trousers", "navy")],
        occasion="everyday",
        weather="mild",
        season="spring",
    )
    assert matched.color_score == 1.0
    # The score weighs colors with formality, warmth, style and more.
    assert 0 < matched.score <= 1
    assert any("pairs with" in reason for reason in matched.reasons)
    assert any(reason == "Works in mild weather." for reason in matched.reasons)

    clashing = recommender.recommend(
        [_item(1, "blouse", "burgundy"), _item(2, "trousers", "olive")],
        occasion="everyday",
    )
    # Burgundy and olive sit far apart on the color wheel.
    assert clashing.color_score < 0.5
    assert any("clashes with" in reason for reason in clashing.reasons)


def test_string_ids_round_trip():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item("top-1", "shirt", "white"),
        _item("bottom-1", "jeans", "navy"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday", lock_ids=["top-1"])
    assert {item.id for item in result.outfit} == {"top-1", "bottom-1"}


def test_exclude_ids_drops_a_piece_when_another_exists():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "blouse", "white"),
        _item(3, "jeans", "navy"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday", exclude_ids=[1])
    assert 1 not in {item.id for item in result.outfit}
    assert 2 in {item.id for item in result.outfit}


def test_exclude_ids_retries_when_it_is_the_only_choice():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "jeans", "navy"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday", exclude_ids=[1])
    assert 1 in {item.id for item in result.outfit}
    assert any(code.code == "repeated_piece" for code in result.reason_codes)


def test_locked_id_is_not_excluded():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "shirt", "white"),
        _item(2, "blouse", "white"),
        _item(3, "jeans", "navy"),
    ]
    result = recommender.recommend(
        wardrobe, occasion="everyday", lock_ids=[1], exclude_ids=[1],
    )
    assert 1 in {item.id for item in result.outfit}


def test_reason_codes_travel_with_the_english_sentences():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    result = recommender.recommend(
        [_item(1, "blouse", "white"), _item(2, "trousers", "navy")],
        occasion="everyday",
        weather="mild",
    )
    pair = next(code for code in result.reason_codes if code.code == "color_pair")
    assert pair.colors == ["white", "navy"]
    assert any(code.code == "weather" for code in result.reason_codes)


def test_layer_prefers_a_look_with_a_jacket():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item(1, "t_shirt", "white"),
        _item(2, "jeans", "navy"),
        _item(3, "jacket", "black"),
    ]
    for _ in range(5):
        result = recommender.recommend(wardrobe, occasion="everyday", weather="mild", layer=True)
        assert 3 in {item.id for item in result.outfit}


def test_layer_still_dresses_a_wardrobe_without_jackets():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [_item(1, "t_shirt", "white"), _item(2, "jeans", "navy")]
    result = recommender.recommend(wardrobe, occasion="everyday", layer=True)
    assert {item.id for item in result.outfit} == {1, 2}


def test_trip_occasions():
    engine = RuleEngine("rules/fashion_rules.json")
    assert engine.validate_outfit([_item(1, "shorts", "beige")], occasion="beach") is True
    assert engine.validate_outfit([_item(1, "suit", "black")], occasion="beach") is False
    # Temples and mosques: knees covered, comfortable shoes.
    assert engine.validate_outfit([_item(1, "shorts", "beige")], occasion="sightseeing") is False
    assert engine.validate_outfit([_item(1, "heels", "black")], occasion="sightseeing") is False
    assert engine.validate_outfit([_item(1, "t_shirt", "white")], occasion="wedding") is False
    assert engine.validate_outfit([_item(1, "blazer", "navy")], occasion="wedding") is True


def test_recommend_route_accepts_layer():
    response = client.post("/recommend/", json={
        "wardrobe": [
            {"id": 1, "type": "t_shirt", "confidence": 0.9, "color": "white", "hex": "#fff"},
            {"id": 2, "type": "jeans", "confidence": 0.9, "color": "navy", "hex": "#000"},
            {"id": 3, "type": "jacket", "confidence": 0.9, "color": "black", "hex": "#000"},
        ],
        "occasion": "sightseeing",
        "weather": "mild",
        "layer": True,
    })
    assert response.status_code == 200
    assert 3 in {item["id"] for item in response.json()["outfit"]}
