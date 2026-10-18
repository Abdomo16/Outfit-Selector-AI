"""
Wardrobe taxonomy shared by the detectors, the fashion encoder, the
recommender and the Flutter wardrobe tabs.

A category is a wardrobe tab. A type is the detailed label stored in
``wardrobe_items.type``. Every type belongs to exactly one category.
"""

CATEGORIES = (
    "Tops",
    "Bottoms",
    "Outerwear",
    "Dresses",
    "Shoes",
    "Bags",
    "Accessories",
)

# category -> {type: prompt phrasings}. Phrasings are averaged into one
# text embedding per type, so each should describe the item on its own.
TYPES = {
    "Tops": {
        "t-shirt": ["a t-shirt", "a short sleeve crew neck tee"],
        "shirt": ["a button-up collared shirt", "a dress shirt"],
        "blouse": ["a women's blouse", "a flowy blouse top"],
        "polo": ["a polo shirt", "a short sleeve polo with a collar"],
        "tank top": ["a sleeveless tank top", "a camisole top"],
        "sweater": ["a knit sweater", "a wool jumper"],
        "hoodie": ["a hoodie", "a hooded sweatshirt"],
        "sweatshirt": ["a crewneck sweatshirt", "a fleece pullover sweatshirt"],
        "cardigan": ["a knit cardigan", "a button-front cardigan sweater"],
    },
    "Bottoms": {
        "jeans": ["denim jeans", "a pair of blue jeans"],
        "trousers": ["formal trousers", "tailored dress pants"],
        "chinos": ["casual chino pants", "cotton khaki chinos"],
        "joggers": ["jogger sweatpants", "track pants"],
        "shorts": ["a pair of shorts", "casual shorts"],
        "skirt": ["a skirt", "a midi skirt"],
        "leggings": ["tight athletic leggings", "yoga leggings"],
    },
    "Outerwear": {
        "jacket": ["a casual jacket", "a zip-up bomber jacket"],
        "denim jacket": ["a denim jacket", "a jean jacket"],
        "leather jacket": ["a leather jacket", "a biker leather jacket"],
        "puffer jacket": ["a puffer jacket", "a quilted down jacket"],
        "coat": ["a long overcoat", "a wool trench coat"],
        "blazer": ["a blazer", "a suit jacket"],
        "vest": ["a puffer vest gilet", "a suit waistcoat"],
    },
    "Dresses": {
        "dress": ["a dress", "a women's dress"],
        "jumpsuit": ["a jumpsuit", "a one-piece romper"],
    },
    "Shoes": {
        "sneakers": ["a pair of sneakers", "athletic running shoes"],
        "boots": ["a pair of boots", "ankle boots"],
        "heels": ["high heel shoes", "women's pumps with heels"],
        "flats": ["ballet flat shoes", "women's flat shoes"],
        "sandals": ["a pair of sandals", "strappy summer sandals"],
        "flip flops": ["flip flops", "thong slippers"],
        "loafers": ["a pair of loafers", "slip-on loafer shoes"],
        "formal shoes": ["formal leather dress shoes", "oxford shoes"],
    },
    "Bags": {
        "handbag": ["a handbag", "a women's purse"],
        "backpack": ["a backpack", "a school rucksack"],
        "crossbody bag": ["a crossbody bag", "a small shoulder bag"],
        "tote bag": ["a tote bag", "a large shopping tote"],
        "wallet": ["a wallet", "a leather card holder"],
    },
    "Accessories": {
        "cap": ["a baseball cap", "a snapback cap"],
        "hat": ["a hat", "a wide-brim hat"],
        "sunglasses": ["a pair of sunglasses", "sunglasses eyewear"],
        "watch": ["a wrist watch", "a wristwatch"],
        "belt": ["a leather belt", "a belt with a buckle"],
        "scarf": ["a scarf", "a knit winter scarf"],
        "tie": ["a necktie", "a silk tie"],
        "socks": ["a pair of socks", "ankle socks"],
        "jewelry": ["jewelry", "a necklace or bracelet"],
    },
}

# DeepFashion2 segmenter labels -> categories the item can belong to.
# Several labels allow two categories because DF2 lumps them together
# (its "vest" is a sleeveless top, a jacket is often a long-sleeved shirt).
DF2_HINTS = {
    "short_sleeved_shirt": ("Tops",),
    "long_sleeved_shirt": ("Tops", "Outerwear"),
    "short_sleeved_outwear": ("Outerwear", "Tops"),
    "long_sleeved_outwear": ("Outerwear", "Tops"),
    "vest": ("Tops", "Outerwear"),
    "sling": ("Tops",),
    "shorts": ("Bottoms",),
    "trousers": ("Bottoms",),
    "skirt": ("Bottoms",),
    "short_sleeved_dress": ("Dresses",),
    "long_sleeved_dress": ("Dresses",),
    "vest_dress": ("Dresses",),
    "sling_dress": ("Dresses",),
}

# YOLOE text prompts for items DeepFashion2 cannot detect, and their category.
YOLOE_PROMPTS = {
    "sneaker": "Shoes",
    "boot": "Shoes",
    "high heel shoe": "Shoes",
    "sandal": "Shoes",
    "shoe": "Shoes",
    "handbag": "Bags",
    "backpack": "Bags",
    "wallet": "Bags",
    "hat": "Accessories",
    "baseball cap": "Accessories",
    "sunglasses": "Accessories",
    "wristwatch": "Accessories",
    "belt": "Accessories",
    "scarf": "Accessories",
    "necktie": "Accessories",
}

STYLES = ("Streetwear", "Elegant", "Minimal", "Vintage", "Sport")
OCCASIONS = ("University", "Work", "Date Night", "Gym", "Party", "Travel")
SEASONS = ("Summer", "Winter", "Spring", "Autumn")
PATTERNS = (
    "Plain",
    "Striped",
    "Plaid",
    "Floral",
    "Polka Dot",
    "Graphic Print",
    "Animal Print",
    "Camouflage",
)

_TYPE_TO_CATEGORY = {
    type_name: category
    for category, types in TYPES.items()
    for type_name in types
}


def all_types() -> list:
    return list(_TYPE_TO_CATEGORY)


def category_for_type(type_name: str | None) -> str | None:
    if not type_name:
        return None
    return _TYPE_TO_CATEGORY.get(type_name.strip().lower())


def types_for_category(category: str) -> list:
    return list(TYPES.get(category, {}))


def hint_for_label(label: str | None) -> tuple:
    """Categories a detector label allows. Empty means any category."""
    if not label:
        return ()
    if label in DF2_HINTS:
        return DF2_HINTS[label]
    if label in YOLOE_PROMPTS:
        return (YOLOE_PROMPTS[label],)
    return ()
