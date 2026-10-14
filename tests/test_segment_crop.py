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
