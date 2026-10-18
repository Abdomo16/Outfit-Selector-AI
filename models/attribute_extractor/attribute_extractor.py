import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from models import colors, taxonomy

DEFAULT_MODEL_ID = "srpone/zooclaw-fashionsiglip2"
FALLBACK_MODEL_ID = "Marqo/marqo-fashionSigLIP"
# SigLIP text towers are trained on fixed-length padded prompts.
TEXT_MAX_LENGTH = 64
# How much more likely a type is when the detector's label points to its category.
HINT_BOOST = 4.0

STYLE_PROMPTS = {
    "Streetwear": ["streetwear style clothing", "urban street fashion"],
    "Elegant": ["elegant formal clothing", "classy dressy fashion"],
    "Minimal": ["minimalist clothing", "simple clean basic fashion"],
    "Vintage": ["vintage retro clothing", "old-fashioned classic fashion"],
    "Sport": ["sportswear athletic clothing", "activewear for training"],
}
OCCASION_PROMPTS = {
    "University": ["casual clothing for university classes", "a student campus outfit"],
    "Work": ["office work attire", "business professional clothing"],
    "Date Night": ["an outfit for a date night", "a stylish evening dinner outfit"],
    "Gym": ["gym workout clothing", "clothing for exercise"],
    "Party": ["party clothing", "a going-out night party outfit"],
    "Travel": ["comfortable travel clothing", "an outfit for a trip"],
}
SEASON_PROMPTS = {
    "Summer": ["light summer clothing", "clothing for hot weather"],
    "Winter": ["warm winter clothing", "clothing for cold weather"],
    "Spring": ["light spring clothing", "clothing for mild spring weather"],
    "Autumn": ["autumn fall clothing", "layered clothing for cool weather"],
}
PATTERN_PROMPTS = {
    "Plain": ["a plain solid color item", "a single color item with no pattern"],
    "Striped": ["a striped item", "an item with stripes"],
    "Plaid": ["a plaid checked item", "a tartan check pattern"],
    "Floral": ["a floral print item", "an item with flowers printed on it"],
    "Polka Dot": ["a polka dot item", "an item with dots"],
    "Graphic Print": ["an item with a graphic print or logo", "a printed graphic design"],
    "Animal Print": ["a leopard or zebra animal print item", "an animal print pattern"],
    "Camouflage": ["a camouflage print item", "a military camo pattern"],
}


def _as_tensor(output) -> torch.Tensor:
    """get_*_features returns a tensor or a model output depending on version."""
    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    raise TypeError(f"Unexpected feature output: {type(output).__name__}")


class _TransformersBackend:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        scale = getattr(self.model, "logit_scale", None)
        self.logit_scale = float(scale.detach().exp()) if scale is not None else 100.0

    def text(self, texts: list) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return _as_tensor(self.model.get_text_features(**inputs))

    def image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return _as_tensor(self.model.get_image_features(pixel_values=inputs["pixel_values"]))


class _OpenClipBackend:
    """Marqo models. Their Transformers wrapper does not load on Transformers 5."""

    def __init__(self, model_id: str, device: str):
        import open_clip

        self.device = device
        name = f"hf-hub:{model_id}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name)
        self.tokenizer = open_clip.get_tokenizer(name)
        self.model.to(device).eval()
        self.logit_scale = float(self.model.logit_scale.detach().exp())

    def text(self, texts: list) -> torch.Tensor:
        return self.model.encode_text(self.tokenizer(texts).to(self.device))

    def image(self, image: Image.Image) -> torch.Tensor:
        return self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))


class FashionEncoder:
    """
    Zero-shot fashion classifier on a fashion-tuned SigLIP model.

    Every prompt is encoded once at startup. Each crop is encoded once and
    compared against the type, style, occasion, season and pattern prompts.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get("FASHION_MODEL_ID", DEFAULT_MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FashionEncoder] Loading {self.model_id} on {self.device}...")
        backend = _OpenClipBackend if self.model_id.startswith("Marqo/") else _TransformersBackend
        self.backend = backend(self.model_id, self.device)
        self.logit_scale = self.backend.logit_scale

        self.type_names = taxonomy.all_types()
        type_prompts = {
            name: phrasings
            for types in taxonomy.TYPES.values()
            for name, phrasings in types.items()
        }
        self.type_embeds = self._encode_groups(type_prompts, self.type_names)
        self.styles, self.style_embeds = self._encode_named(STYLE_PROMPTS)
        self.occasions, self.occasion_embeds = self._encode_named(OCCASION_PROMPTS)
        self.seasons, self.season_embeds = self._encode_named(SEASON_PROMPTS)
        self.patterns, self.pattern_embeds = self._encode_named(PATTERN_PROMPTS)
        self._color_embeds = {}
        print(f"[FashionEncoder] Ready with {len(self.type_names)} types.")

    @torch.no_grad()
    def _encode_text(self, texts: list) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), 32):
            features = self.backend.text(texts[start:start + 32])
            chunks.append(F.normalize(features.float(), dim=-1))
        return torch.cat(chunks)

    def _encode_groups(self, prompts: dict, names: list) -> torch.Tensor:
        """One embedding per name, averaged over its phrasings."""
        flat = [p for name in names for p in prompts[name]]
        embeds = self._encode_text(flat)
        rows, offset = [], 0
        for name in names:
            count = len(prompts[name])
            rows.append(embeds[offset:offset + count].mean(dim=0))
            offset += count
        return F.normalize(torch.stack(rows), dim=-1)

    def _encode_named(self, prompts: dict) -> tuple:
        names = list(prompts)
        return names, self._encode_groups(prompts, names)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        features = self.backend.image(image.convert("RGB"))
        return F.normalize(features.float(), dim=-1)[0]

    def _rank(self, image_embed: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return (self.logit_scale * text_embeds @ image_embed).softmax(dim=0)

    def classify(self, image_embed: torch.Tensor, hint: tuple = ()) -> tuple:
        """
        Pick the best type, favouring the hinted categories when given.

        The hint is a prior, not a filter: detectors sometimes label a
        neighbouring garment, and a clear image match should still win.
        Returns (category, type, confidence).
        """
        probs = self._rank(image_embed, self.type_embeds)
        if hint:
            allowed = {t for category in hint for t in taxonomy.types_for_category(category)}
            weights = torch.tensor(
                [HINT_BOOST if t in allowed else 1.0 for t in self.type_names],
                device=probs.device,
            )
            probs = probs * weights
            probs = probs / probs.sum()
        best = int(probs.argmax())
        type_name = self.type_names[best]
        return taxonomy.category_for_type(type_name), type_name, float(probs[best])

    def _best(self, image_embed: torch.Tensor, names: list, embeds: torch.Tensor) -> str:
        return names[int(self._rank(image_embed, embeds).argmax())]

    def attributes(self, image_embed: torch.Tensor) -> dict:
        return {
            "pattern": self._best(image_embed, self.patterns, self.pattern_embeds),
            "style": self._best(image_embed, self.styles, self.style_embeds),
            "occasion": self._best(image_embed, self.occasions, self.occasion_embeds),
            "season": self._best(image_embed, self.seasons, self.season_embeds),
        }

    def color_scores(self, image_embed, type_name: str) -> list:
        """
        [(color, probability)] for prompts like "a light grey t-shirt",
        best first. Prompts are encoded once per type.
        """
        if type_name not in self._color_embeds:
            self._color_embeds[type_name] = self._encode_text(
                [f"a {color} {type_name}" for color in colors.COLOR_NAMES]
            )
        embed = torch.as_tensor(image_embed, dtype=torch.float32, device=self.device)
        probs = self._rank(embed, self._color_embeds[type_name])
        return sorted(zip(colors.COLOR_NAMES, probs.tolist()), key=lambda c: c[1], reverse=True)

    def predict(self, image: Image.Image, hint: tuple = ()) -> dict:
        """Type, category, attributes and the image embedding for one crop."""
        embed = self.encode_image(image)
        category, type_name, confidence = self.classify(embed, hint)
        return {
            "category": category,
            "type": type_name,
            "confidence": round(confidence, 4),
            **self.attributes(embed),
            "embedding": embed.cpu().tolist(),
        }
