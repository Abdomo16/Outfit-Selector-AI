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

def test_recommender():
    engine = RuleEngine("rules/fashion_rules.json")
    recommender = Recommender(engine)
    
    wardrobe = [
        WardrobeItemFull(id=1, type="jeans", confidence=0.9, color="blue", hex="#000", embedding=[1.0, 0.0]),
        WardrobeItemFull(id=2, type="t-shirt", confidence=0.9, color="white", hex="#fff", embedding=[1.0, 0.1]),
        WardrobeItemFull(id=3, type="sneakers", confidence=0.9, color="white", hex="#fff", embedding=[0.0, 1.0]),
        WardrobeItemFull(id=4, type="suit", confidence=0.9, color="black", hex="#000", embedding=[0.5, 0.5])
    ]
    
    outfit, score = recommender.recommend(wardrobe, occasion="casual")
    assert len(outfit) > 0
    for item in outfit:
        assert item.type != "suit"

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
    assert len(data["outfit"]) == 2
