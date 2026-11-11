"""The outfit scorer and the complaints users have about wardrobe apps:
wool in summer, the same shoes every time, nothing when a category is
missing, and outfits that ignore the occasion."""
import json
import time
from datetime import date, timedelta

from api.schemas import WardrobeItemFull
from models.recommender.recommender import Recommender, zone_of
from models.recommender.rule_engine import RuleEngine, normalise_type
from models.recommender.scoring import OutfitScorer, warmth_target
from models.taxonomy import TYPES

RULES = "rules/fashion_rules.json"
HEX = {
    "white": "#f5f5f5", "black": "#111111", "navy": "#1f2a44", "beige": "#d9c8a9",
    "grey": "#808080", "brown": "#5c4033", "blue": "#2a52be", "olive": "#556b2f",
    "burgundy": "#6d2135", "red": "#c62828", "green": "#2e8b57", "light blue": "#a7c7e7",
}


def _item(item_id, item_type, color, style=None, pattern=None, season=None):
    return WardrobeItemFull(
        id=item_id, type=item_type, confidence=0.9, color=color,
        hex=HEX.get(color, "#808080"), style=style, pattern=pattern, season=season,
    )


def _ids(result):
    return {item.id for item in result.outfit}


def _scorer(**kwargs):
    rules = RuleEngine(RULES).rules
    return OutfitScorer(zone_of, rules.get("color_rules", {}), **kwargs)


def test_every_rule_names_a_type_the_classifier_knows():
    known = {normalise_type(t) for types in TYPES.values() for t in types}
    rules = json.load(open(RULES, encoding="utf-8"))
    unknown = [
        (section, key, name)
        for section in ("occasion_rules", "season_rules", "weather_rules")
        for key, rule in rules[section].items()
        for field in ("forbidden_types", "allowed_types")
        for name in rule.get(field, [])
        if normalise_type(name) not in known
    ]
    assert unknown == []


def test_old_labels_still_meet_the_rules():
    engine = RuleEngine(RULES)
    assert engine.validate_outfit([_item(1, "wool_sweater", "grey")], weather="hot") is False
    assert engine.validate_outfit([_item(1, "coat", "black")], weather="hot") is False


def test_no_coat_or_sweater_on_a_hot_day():
    wardrobe = [
        _item(1, "t-shirt", "white"), _item(2, "sweater", "navy"),
        _item(3, "jeans", "navy"), _item(4, "coat", "beige"),
        _item(5, "sneakers", "white"), _item(6, "heavy_coat", "black"),
    ]
    recommender = Recommender(RuleEngine(RULES))
    for _ in range(10):
        result = recommender.recommend(wardrobe, occasion="everyday", temp_high=37, temp_low=27)
        assert _ids(result) & {2, 4, 6} == set()
        assert any(code.code == "light_for_heat" for code in result.reason_codes)


def test_jacket_for_a_cool_night_but_not_a_warm_one():
    wardrobe = [
        _item(1, "t-shirt", "white"), _item(2, "jeans", "navy"),
        _item(3, "jacket", "black"), _item(4, "sneakers", "white"),
    ]
    recommender = Recommender(RuleEngine(RULES))
    cool = recommender.recommend(wardrobe, occasion="everyday", temp_high=24, temp_low=12)
    assert 3 in _ids(cool)
    assert any(code.code == "warm_enough_for_night" for code in cool.reason_codes)
    warm = recommender.recommend(wardrobe, occasion="everyday", temp_high=34, temp_low=27)
    assert 3 not in _ids(warm)


def test_warmth_target_rises_as_it_gets_colder():
    assert warmth_target(35) < warmth_target(22) < warmth_target(10)


def test_regenerate_shows_a_different_outfit():
    wardrobe = [
        _item(1, "t-shirt", "white"), _item(2, "t-shirt", "black"),
        _item(3, "jeans", "navy"), _item(4, "sneakers", "white"),
    ]
    recommender = Recommender(RuleEngine(RULES))
    first = recommender.recommend(wardrobe, occasion="everyday")
    for _ in range(10):
        again = recommender.recommend(
            wardrobe, occasion="everyday", previous_ids=[item.id for item in first.outfit],
        )
        assert _ids(again) != _ids(first)


def test_the_same_shoes_are_not_picked_every_time():
    wardrobe = [
        _item(1, "t-shirt", "white"), _item(2, "jeans", "navy"),
        _item(3, "sneakers", "white"), _item(4, "sneakers", "black"),
        _item(5, "sneakers", "grey"),
    ]
    recommender = Recommender(RuleEngine(RULES))
    shoes, previous = set(), None
    for _ in range(12):
        result = recommender.recommend(wardrobe, occasion="everyday", previous_ids=previous)
        previous = [item.id for item in result.outfit]
        shoes |= _ids(result) & {3, 4, 5}
    assert len(shoes) > 1


def test_recently_worn_pieces_score_lower():
    today = date(2026, 11, 6)
    tee, jeans = _item(1, "t-shirt", "white"), _item(2, "jeans", "navy")
    worn, fresh = _item(3, "sneakers", "white"), _item(4, "sneakers", "white")
    scorer = _scorer(
        occasion="everyday",
        last_worn={"3": (today - timedelta(days=2)).isoformat()},
        today=today,
    )
    assert scorer.score([tee, jeans, fresh]).total > scorer.score([tee, jeans, worn]).total
    _, codes = scorer.reasons([tee, jeans, fresh])
    assert any(code.code == "forgotten_piece" for code in codes)


def test_work_prefers_the_dressier_look():
    scorer = _scorer(occasion="work")
    smart = [_item(1, "shirt", "white"), _item(2, "trousers", "navy"), _item(3, "loafers", "brown")]
    casual = [_item(4, "t-shirt", "white"), _item(5, "jeans", "navy"), _item(3, "loafers", "brown")]
    assert scorer.score(smart).parts["formality"] > scorer.score(casual).parts["formality"]
    _, codes = scorer.reasons(smart)
    assert any(code.code == "formality_match" for code in codes)


def test_one_pattern_beats_two():
    scorer = _scorer(occasion="everyday")
    one = [_item(1, "shirt", "white", pattern="Striped"), _item(2, "jeans", "navy")]
    two = [_item(1, "shirt", "white", pattern="Striped"), _item(3, "skirt", "navy", pattern="Floral")]
    assert scorer.score(one).parts["pattern"] > scorer.score(two).parts["pattern"]


def test_styles_that_clash_score_lower():
    scorer = _scorer(occasion="everyday")
    matched = [_item(1, "blouse", "white", style="Elegant"), _item(2, "trousers", "black", style="Minimal")]
    clashing = [_item(1, "blouse", "white", style="Elegant"), _item(3, "joggers", "black", style="Sport")]
    assert scorer.score(matched).parts["style"] > scorer.score(clashing).parts["style"]


def test_colors_score_by_how_they_sit_together():
    scorer = _scorer()
    neutral = [_item(1, "t-shirt", "white"), _item(2, "jeans", "navy")]
    clash = [_item(1, "t-shirt", "red"), _item(2, "jeans", "green")]
    tonal = [_item(1, "shirt", "light blue"), _item(2, "jeans", "blue"), _item(3, "jacket", "blue")]
    assert scorer.score(neutral).parts["color"] > scorer.score(clash).parts["color"]
    _, codes = scorer.reasons(tonal)
    assert any(code.code == "tonal_outfit" for code in codes)


def test_missing_shoes_are_reported():
    recommender = Recommender(RuleEngine(RULES))
    result = recommender.recommend(
        [_item(1, "shirt", "white"), _item(2, "chinos", "beige"), _item(3, "sneakers", "white")],
        occasion="work",
    )
    assert _ids(result) == {1, 2}
    assert result.missing == ["shoes"]
    assert result.relaxed is False


def test_closest_look_when_nothing_fits_the_occasion():
    recommender = Recommender(RuleEngine(RULES))
    result = recommender.recommend(
        [_item(1, "t-shirt", "white"), _item(2, "jeans", "navy"), _item(3, "sneakers", "white")],
        occasion="wedding",
    )
    assert result.outfit
    assert result.relaxed is True
    assert result.missing == ["top", "shoes"]
    assert result.reason_codes[0].code == "relaxed_occasion"


def test_a_big_wardrobe_is_still_quick():
    colors = list(HEX)
    wardrobe = (
        [_item(f"t{i}", "t-shirt", colors[i % len(colors)]) for i in range(40)]
        + [_item(f"b{i}", "jeans", colors[i % len(colors)]) for i in range(30)]
        + [_item(f"o{i}", "jacket", colors[i % len(colors)]) for i in range(10)]
        + [_item(f"s{i}", "sneakers", colors[i % len(colors)]) for i in range(15)]
    )
    start = time.perf_counter()
    result = Recommender(RuleEngine(RULES)).recommend(wardrobe, occasion="everyday", temp_high=22)
    assert result.outfit
    assert time.perf_counter() - start < 10
