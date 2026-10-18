import numpy as np
from PIL import Image

from models.segmentation.segmenter import Segmenter


def test_crop_keeps_the_full_box_when_mask_is_a_sliver():
    image = Image.new("RGB", (100, 200), (10, 20, 30))
    mask = np.zeros((200, 100), dtype=np.float32)
    mask[90:110, 40:60] = 1.0
    box = (10, 20, 90, 180)

    crop, crop_mask = Segmenter._masked_crop(image, mask, box)

    assert crop.size == (80, 160)
    assert crop_mask is None


def test_crop_stays_box_sized_when_mask_covers_the_garment():
    image = Image.new("RGB", (100, 200), (10, 20, 30))
    mask = np.zeros((200, 100), dtype=np.float32)
    mask[20:180, 10:90] = 1.0
    box = (10, 20, 90, 180)

    crop, crop_mask = Segmenter._masked_crop(image, mask, box)

    assert crop.size == (80, 160)
    assert crop_mask is not None
    assert crop_mask.shape == (160, 80)


def test_one_item_short_box_saves_the_full_image():
    image = Image.new("RGB", (100, 200), (10, 20, 30))
    mask = np.zeros((200, 100), dtype=np.float32)
    mask[120:180, 20:80] = 1.0

    crop, crop_mask = Segmenter._full_photo(image, mask)

    assert crop.size == image.size
    assert crop.getpixel((50, 10)) == (10, 20, 30)
    assert crop_mask is None


def test_one_item_tall_mask_keeps_full_frame_and_whites_background():
    image = Image.new("RGB", (100, 200), (10, 20, 30))
    mask = np.zeros((200, 100), dtype=np.float32)
    mask[5:195, 15:85] = 1.0

    crop, crop_mask = Segmenter._full_photo(image, mask)

    assert crop.size == image.size
    assert crop.getpixel((0, 0)) == (255, 255, 255)
    assert crop.getpixel((50, 100)) == (10, 20, 30)
    assert crop_mask is not None
    assert crop_mask.shape == (200, 100)


def test_pad_box_stays_inside_the_photo():
    assert Segmenter._pad_box(0, 0, 50, 100, 80, 120)[0] == 0
    assert Segmenter._pad_box(0, 0, 50, 100, 80, 120)[1] == 0


def _det(label, box, conf=0.5):
    return {"label": label, "box": box, "conf": conf, "mask": None}


def test_left_and_right_shoe_become_one_pair():
    merged = Segmenter._merge_shoe_pairs([
        _det("sneaker", (10, 100, 40, 130), conf=0.6),
        _det("shoe", (200, 105, 230, 135), conf=0.4),
    ])

    assert len(merged) == 1
    assert merged[0]["box"] == (10, 100, 230, 135)
    assert merged[0]["label"] == "sneaker"


def test_shoes_on_different_rows_and_bags_stay_separate():
    merged = Segmenter._merge_shoe_pairs([
        _det("shoe", (10, 10, 40, 40)),
        _det("shoe", (10, 200, 40, 230)),
        _det("handbag", (50, 50, 90, 90)),
        _det("handbag", (100, 50, 140, 90)),
    ])

    assert len(merged) == 4


def test_inside_measures_the_share_of_the_inner_box():
    assert Segmenter._inside((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    assert Segmenter._inside((90, 90, 110, 110), (0, 0, 100, 100)) == 0.25
    assert Segmenter._inside((200, 200, 210, 210), (0, 0, 100, 100)) == 0.0


def _mask(rows, cols, shape=(200, 100)):
    mask = np.zeros(shape, dtype=np.float32)
    mask[rows, cols] = 1.0
    return mask


def test_item_running_off_the_bottom_edge_is_cut_off():
    trousers = _mask(slice(150, 200), slice(30, 70))
    assert Segmenter._partial_reason(trousers, (30, 150, 70, 200), []) == "cut_off"


def test_item_inside_the_photo_is_complete():
    shirt = _mask(slice(20, 120), slice(20, 80))
    assert Segmenter._partial_reason(shirt, (20, 20, 80, 120), []) is None


def test_corner_touch_is_not_cut_off():
    shirt = _mask(slice(20, 120), slice(20, 80))
    shirt[118:120, 20:24] = 1.0
    shirt[195:200, 20:24] = 1.0
    assert Segmenter._partial_reason(shirt, (20, 20, 80, 120), []) is None


def test_item_mostly_under_another_is_hidden():
    trousers = _mask(slice(120, 180), slice(30, 70))
    box = (30, 60, 70, 180)
    shirt = _mask(slice(20, 120), slice(20, 80))
    assert Segmenter._partial_reason(trousers, box, [shirt]) == "hidden"
    assert Segmenter._partial_reason(trousers, (30, 120, 70, 180), [shirt]) is None


def test_missing_mask_is_not_flagged():
    assert Segmenter._partial_reason(None, (0, 0, 10, 10), []) is None


def test_mask_box_trims_to_the_garment():
    mask = _mask(slice(60, 140), slice(30, 70))
    box = Segmenter._mask_box(mask, (10, 20, 90, 180), 100, 200)
    assert box == (29, 57, 71, 143)


def test_mask_box_keeps_the_box_for_a_sliver_mask():
    mask = _mask(slice(90, 110), slice(40, 60))
    assert Segmenter._mask_box(mask, (10, 20, 90, 180), 100, 200) == (10, 20, 90, 180)
