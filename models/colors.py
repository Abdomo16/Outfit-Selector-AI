"""
Garment color names shared by the color detector, the fashion encoder vote
and the outfit rules.

Colors are compared in CIELAB with CIEDE2000, which tracks how people see
color differences far better than RGB distance.
"""
import cv2
import numpy as np

# name -> reference shades (sRGB hex). Several shades per name cover the
# range a garment of that color shows under normal lighting.
REFERENCE_SHADES = {
    "white":      ["#ffffff", "#f5f5f5", "#f8f8f2"],
    "cream":      ["#fffdd0", "#f3e9d2", "#efe6cf"],
    "beige":      ["#f5f5dc", "#e8dcc8", "#d9c8a9"],
    "tan":        ["#d2b48c", "#c19a6b", "#a9865b", "#967b5a"],
    "khaki":      ["#c3b091", "#bdb08a", "#a89f73"],
    "brown":      ["#8b4513", "#6f4e37", "#5c4033", "#3d2b1f"],
    "light grey": ["#d3d3d3", "#c0c0c0"],
    "grey":       ["#808080", "#9e9e9e"],
    "charcoal":   ["#36454f", "#404040"],
    "black":      ["#000000", "#1a1a1a"],
    "navy":       ["#000080", "#1f2a44", "#232b3e", "#14213d"],
    "blue":       ["#1e90ff", "#4169e1", "#2a52be", "#0047ab"],
    "light blue": ["#add8e6", "#87ceeb", "#a7c7e7"],
    "teal":       ["#008080", "#2a7f86", "#367588"],
    "green":      ["#228b22", "#2e8b57", "#3cb371", "#006400"],
    "olive":      ["#808000", "#556b2f", "#6b6b47"],
    "yellow":     ["#ffd700", "#ffeb3b", "#f4d03f", "#e1ad01"],
    "orange":     ["#ff8c00", "#f28c28", "#e86a33"],
    "red":        ["#dc143c", "#c62828", "#e53935"],
    "burgundy":   ["#800020", "#6d2135", "#4a1c2c"],
    "pink":       ["#ffc0cb", "#ffb6c1", "#f4a6b8", "#ff69b4"],
    "purple":     ["#800080", "#6a0dad", "#4b2142", "#9370db"],
}
COLOR_NAMES = tuple(REFERENCE_SHADES)

# Colors this close to grey are named by lightness alone. Hue is noise there,
# which is how a white shirt used to read as beige and charcoal as green.
NEUTRAL_CHROMA = 8.0
NEUTRAL_LIGHTNESS = (
    (88.0, "white"),
    (68.0, "light grey"),
    (42.0, "grey"),
    (20.0, "charcoal"),
    (0.0, "black"),
)
NEUTRAL_NAMES = {name for _, name in NEUTRAL_LIGHTNESS}

# The outfit rules in rules/fashion_rules.json know these family names.
FAMILY = {
    "light grey": "grey",
    "charcoal": "grey",
    "tan": "khaki",
    "light blue": "blue",
    "maroon": "burgundy",
}

# A vote from the fashion model is trusted when its color is this close to
# the measured fabric (CIEDE2000), or in the same family.
VOTE_MAX_DIFFERENCE = 20.0
VOTE_MIN_CONFIDENCE = 0.4


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB (..., 3) in 0..255 -> CIELAB (..., 3) with L in 0..100."""
    arr = np.asarray(rgb, dtype=np.float32)
    shape = arr.shape
    lab = cv2.cvtColor((arr.reshape(-1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    return lab.reshape(shape)


def lab_to_rgb(lab) -> np.ndarray:
    arr = np.asarray(lab, dtype=np.float32)
    shape = arr.shape
    rgb = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).reshape(shape)


def ciede2000(lab1, lab2) -> np.ndarray:
    """CIEDE2000 color difference. Inputs broadcast over the last axis (L, a, b)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma_zero = (c1p * c2p) == 0

    dlp = L2 - L1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(chroma_zero, 0, dhp)
    dHp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2))

    lbp = (L1 + L2) / 2
    cbp = (c1p + c2p) / 2
    hsum = h1p + h2p
    hbp = np.where(
        np.abs(h1p - h2p) <= 180,
        hsum / 2,
        np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
    )
    hbp = np.where(chroma_zero, hsum, hbp)

    t = (
        1
        - 0.17 * np.cos(np.radians(hbp - 30))
        + 0.24 * np.cos(np.radians(2 * hbp))
        + 0.32 * np.cos(np.radians(3 * hbp + 6))
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    )
    d_theta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc = 1 + 0.045 * cbp
    sh = 1 + 0.015 * cbp * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt(
        (dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dcp / sc) * (dHp / sh)
    )


_REF_NAMES = [name for name, shades in REFERENCE_SHADES.items() for _ in shades]
_REF_LAB = rgb_to_lab([hex_to_rgb(h) for shades in REFERENCE_SHADES.values() for h in shades])


def is_neutral(lab) -> bool:
    return float(np.hypot(float(lab[1]), float(lab[2]))) < NEUTRAL_CHROMA


def name_for_lab(lab) -> str:
    """The garment color name a person would use for this Lab color."""
    L, a, b = (float(v) for v in lab)
    if np.hypot(a, b) < NEUTRAL_CHROMA:
        for threshold, name in NEUTRAL_LIGHTNESS:
            if L >= threshold:
                return name
    return _REF_NAMES[int(np.argmin(ciede2000(np.asarray([L, a, b]), _REF_LAB)))]


def name_for_rgb(rgb) -> str:
    return name_for_lab(rgb_to_lab(np.asarray(rgb, dtype=np.float32)))


def family(name: str | None) -> str:
    """The name the outfit rules use for this color."""
    key = (name or "").strip().lower()
    return FAMILY.get(key, key)


def difference_to(name: str, lab) -> float:
    """Smallest CIEDE2000 difference between a color and a name's shades."""
    refs = _REF_LAB[[i for i, n in enumerate(_REF_NAMES) if n == name]]
    return float(ciede2000(np.asarray(lab, dtype=np.float64), refs).min())


def vote_agrees(voted: str, lab) -> bool:
    """A fashion-model color is believable if the fabric could be that color."""
    if family(voted) == family(name_for_lab(lab)):
        return True
    return difference_to(voted, lab) < VOTE_MAX_DIFFERENCE


def choose_name(pixel_name: str, lab, scores: list) -> str:
    """
    The fashion model names the color when it is confident and the measured
    fabric agrees; it reads "white shirt" correctly under warm light. Pixels
    decide otherwise, which stops votes like "olive" for near-black chinos.
    """
    if scores:
        voted, probability = scores[0]
        # Fabric with no measurable color stays black, grey or white; near
        # black the model cannot tell navy from black.
        if is_neutral(lab) and voted not in NEUTRAL_NAMES:
            return pixel_name
        if probability >= VOTE_MIN_CONFIDENCE and vote_agrees(voted, lab):
            return voted
    return pixel_name
