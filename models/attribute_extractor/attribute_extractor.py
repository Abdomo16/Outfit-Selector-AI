import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class AttributeExtractor:
    def __init__(self):
        print("[AttributeExtractor] Loading OpenAI CLIP Model (this might take a moment on first run)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/clip-vit-base-patch32"
        # Download and cache the weights
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        print(f"[AttributeExtractor] CLIP loaded successfully onto {self.device}!")
        
        # 1. Styles exactly matching Flutter App
        self.styles = ["Streetwear", "Elegant", "Minimal", "Vintage"]
        self.style_prompts = [f"a photo of {s.lower()} fashion clothing" for s in self.styles]
        
        # 2. Occasions exactly matching Flutter App
        self.occasions = ["University", "Work", "Date Night", "Gym", "Party", "Travel"]
        self.occasion_prompts = [f"a photo of clothing suited for {o.lower()}" for o in self.occasions]
        
        # 3. Basic Patterns
        self.patterns = ["Plain", "Striped", "Floral", "Plaid"]
        self.pattern_prompts = [f"a photo of {p.lower()} patterned clothing" for p in self.patterns]

        # 4. Seasons / Weather (Summer, Winter, etc.)
        self.seasons = ["Summer", "Winter", "Spring", "Autumn"]
        self.season_prompts = [f"a photo of clothing suited for {s.lower()} weather" for s in self.seasons]

    def _get_best_match(self, image: Image.Image, choices: list, prompts: list) -> str:
        # Pass the image and text prompts to the model
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            best_idx = probs.argmax().item()
        
        return choices[best_idx]

    def predict(self, image: Image.Image) -> dict:
        """
        Use Zero-Shot CLIP classification to extract stylistic attributes.
        """
        # We ensure it's RGB
        img = image.convert("RGB")
        
        style = self._get_best_match(img, self.styles, self.style_prompts)
        occasion = self._get_best_match(img, self.occasions, self.occasion_prompts)
        pattern = self._get_best_match(img, self.patterns, self.pattern_prompts)
        season = self._get_best_match(img, self.seasons, self.season_prompts)
        
        return {
            "pattern": pattern,
            "style": style,
            "occasion": occasion,
            "season": season
        }
