from api.schemas import WardrobeItemFull
from models import taxonomy
from models.recommender.recommender import _zone

DF2_CLASSES = {
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
    "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress",
}

EXPECTED_ZONE = {
    "Tops": "top",
    "Bottoms": "bottom",
    "Outerwear": "outerwear",
    "Dresses": "one_piece",
    "Shoes": "shoes",
    "Bags": None,
    "Accessories": None,
}


def test_every_deepfashion2_label_has_a_category():
    assert set(taxonomy.DF2_HINTS) == DF2_CLASSES
    for hint in taxonomy.DF2_HINTS.values():
        assert hint and set(hint) <= set(taxonomy.CATEGORIES)


def test_every_yoloe_prompt_has_a_category():
    assert set(taxonomy.YOLOE_PROMPTS.values()) <= set(taxonomy.CATEGORIES)


def test_every_type_has_one_category_and_prompts():
    seen = set()
    for category in taxonomy.CATEGORIES:
        for type_name, prompts in taxonomy.TYPES[category].items():
            assert type_name not in seen, f"{type_name} is in two categories"
            seen.add(type_name)
            assert prompts, f"{type_name} has no prompts"
            assert taxonomy.category_for_type(type_name) == category


def test_every_type_lands_in_its_recommender_slot():
    for category, zone in EXPECTED_ZONE.items():
        for type_name in taxonomy.types_for_category(category):
            item = WardrobeItemFull(type=type_name, confidence=1.0, color="black", hex="#000000")
            assert _zone(item) == zone, f"{type_name} should be in {zone}"


def test_hint_for_label():
    assert taxonomy.hint_for_label("trousers") == ("Bottoms",)
    assert taxonomy.hint_for_label("sneaker") == ("Shoes",)
    assert taxonomy.hint_for_label("handbag") == ("Bags",)
    assert taxonomy.hint_for_label(None) == ()
    assert taxonomy.hint_for_label("unknown") == ()
