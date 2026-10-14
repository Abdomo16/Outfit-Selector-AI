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


def _item(item_id, item_type, color, style=None):
    return WardrobeItemFull(
        id=item_id,
        type=item_type,
        confidence=0.9,
        color=color,
        hex="#000000",
        style=style,
    )


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
    assert matched.score == matched.color_score
    assert any("pairs with" in reason for reason in matched.reasons)
    assert any(reason == "Works in mild weather." for reason in matched.reasons)

    clashing = recommender.recommend(
        [_item(1, "blouse", "burgundy"), _item(2, "trousers", "olive")],
        occasion="everyday",
    )
    assert clashing.color_score == 0.0
    assert any("clashes with" in reason for reason in clashing.reasons)


def test_string_ids_round_trip():
    recommender = Recommender(RuleEngine("rules/fashion_rules.json"))
    wardrobe = [
        _item("top-1", "shirt", "white"),
        _item("bottom-1", "jeans", "navy"),
    ]
    result = recommender.recommend(wardrobe, occasion="everyday", lock_ids=["top-1"])
    assert {item.id for item in result.outfit} == {"top-1", "bottom-1"}
