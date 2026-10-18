import numpy as np
import pytest
from PIL import Image

from models import colors
from models.color_detector.color_detector import ColorDetector


def test_ciede2000_matches_published_reference():
    # Sharma, Wu and Dalal (2005), test pair 1.
    assert colors.ciede2000([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485]) == pytest.approx(2.0425, abs=1e-4)
    assert colors.ciede2000([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]) == pytest.approx(0.0)


@pytest.mark.parametrize("hex_value, expected", [
    ("#e3e3e3", {"white", "light grey"}),   # white tee, was beige
    ("#ededeb", {"white"}),                 # white tee, was beige
    ("#a2a2a7", {"grey", "light grey"}),    # grey tee, was khaki
    ("#413f46", {"charcoal"}),              # charcoal trousers, was green
    ("#3662b9", {"blue"}),                  # blue tee, was teal
    ("#877452", {"tan", "khaki"}),          # tan chinos, was grey
    ("#2e1e16", {"brown"}),                 # dark brown chinos, was black
    ("#f1c0ca", {"pink"}),                  # pink shirt, was beige
    ("#161616", {"black"}),
    ("#2a2f42", {"navy"}),
    ("#f2d82c", {"yellow"}),
    ("#dc3249", {"red"}),
    ("#585846", {"olive"}),
])
def test_names_from_your_wardrobe(hex_value, expected):
    assert colors.name_for_rgb(colors.hex_to_rgb(hex_value)) in expected


def test_every_name_has_shades_and_a_family_the_rules_accept():
    for name in colors.COLOR_NAMES:
        assert colors.REFERENCE_SHADES[name]
        assert colors.family(name)
    assert colors.family("charcoal") == "grey"
    assert colors.family("tan") == "khaki"
    assert colors.family("Navy") == "navy"


def test_vote_is_trusted_only_when_the_fabric_could_be_that_color():
    near_black = colors.rgb_to_lab(np.asarray(colors.hex_to_rgb("#12171d"), dtype=np.float32))
    warm_white = colors.rgb_to_lab(np.asarray(colors.hex_to_rgb("#efe8dc"), dtype=np.float32))
    assert not colors.vote_agrees("olive", near_black)
    assert colors.vote_agrees("navy", near_black)
    assert colors.vote_agrees("white", warm_white)
    assert not colors.vote_agrees("red", warm_white)


def test_choose_name_needs_confidence_and_agreement():
    grey = colors.rgb_to_lab(np.asarray([165, 165, 170], dtype=np.float32))
    assert colors.choose_name("grey", grey, [("light grey", 0.7)]) == "light grey"
    assert colors.choose_name("grey", grey, [("light grey", 0.2)]) == "grey"
    assert colors.choose_name("grey", grey, [("red", 0.9)]) == "grey"
    assert colors.choose_name("grey", grey, []) == "grey"


def test_colorless_fabric_keeps_its_neutral_name():
    near_black = colors.rgb_to_lab(np.asarray(colors.hex_to_rgb("#222129"), dtype=np.float32))
    assert colors.choose_name("black", near_black, [("navy", 0.8)]) == "black"
    assert colors.choose_name("black", near_black, [("charcoal", 0.8)]) == "charcoal"


def _white_shirt_on_peach_wall():
    photo = Image.new("RGB", (200, 300), (231, 195, 160))
    photo.paste((246, 246, 248), (50, 40, 150, 260))
    alpha = np.zeros((300, 200), dtype=np.float32)
    alpha[40:260, 50:150] = 1.0
    return photo, alpha


def test_white_shirt_on_a_peach_wall_reads_white_with_background_removal():
    photo, alpha = _white_shirt_on_peach_wall()
    detector = ColorDetector(background_remover=lambda image: alpha)

    read = detector.read(photo)

    assert read["color"] == "white"
    assert read["secondary_color"] is None


def test_without_background_removal_the_centre_of_the_photo_is_used():
    photo, _ = _white_shirt_on_peach_wall()

    assert ColorDetector().read(photo)["color"] == "white"


def test_empty_rembg_result_falls_back_to_the_detector_mask():
    photo, alpha = _white_shirt_on_peach_wall()
    detector = ColorDetector(background_remover=lambda image: np.zeros_like(alpha))

    assert detector.read(photo, det_mask=(alpha * 255).astype(np.uint8))["color"] == "white"


def test_two_color_fabric_gets_a_secondary_color():
    shirt = Image.new("RGB", (200, 200), (31, 42, 68))      # navy
    for x in range(0, 200, 20):
        shirt.paste((245, 245, 245), (x, 0, x + 8, 200))    # white stripes, 40%

    read = ColorDetector().read(shirt, patterned=True)

    assert read["color"] == "navy"
    assert read["secondary_color"] == "white"
    assert read["secondary_hex"].startswith("#")


def test_another_garment_in_a_plain_crop_is_not_a_second_color():
    crop = Image.new("RGB", (200, 200), (18, 23, 29))       # black trousers
    crop.paste((92, 92, 73), (0, 0, 200, 50))               # olive shirt hem, 25%

    read = ColorDetector().read(crop)

    assert read["color"] == "black"
    assert read["secondary_color"] is None


def test_shading_on_one_color_is_not_a_second_color():
    pants = Image.new("RGB", (200, 200), (78, 57, 46))       # brown
    pants.paste((62, 45, 36), (0, 100, 200, 200))            # shadowed half

    read = ColorDetector().read(pants)

    assert read["color"] == "brown"
    assert read["secondary_color"] is None
