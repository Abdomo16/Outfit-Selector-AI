import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
except ImportError:
    tf = None

from PIL import Image

class ClothingClassifier:
    def __init__(self, weights_path: str = "weights/best_model_phase2 (3).keras", classes_json: str = "weights/classes_best_model_phase2.json"):
        self.weights_path = weights_path
        self.classes_json = classes_json
        self.class_names = self._load_classes()
        self.model = self._load_model()
        
    def _load_classes(self) -> list:
        if os.path.exists(self.classes_json):
            with open(self.classes_json, "r") as f:
                data = json.load(f)
                # Sort elements by key index
                classes = [data[str(i)] for i in range(len(data))]
                return classes
        else:
            print(f"[Classifier] WARNING: No classes file found at '{self.classes_json}'.")
            return ["unknown"] * 23

    def _load_model(self):
        if tf is None:
            print("[Classifier] WARNING: tensorflow is not installed.")
            return None
            
        if os.path.exists(self.weights_path):
            model = tf.keras.models.load_model(self.weights_path)
            print(f"[Classifier] Loaded weights from {self.weights_path}")
            return model
        else:
            print(f"[Classifier] WARNING: No weights found at '{self.weights_path}'.")
            return None

    def predict(self, image: Image.Image) -> dict:
        if self.model is None or tf is None:
            return {"type": "unknown", "confidence": 0.0}

        image = image.convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image, dtype=np.float32)
        img_tensor = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_tensor, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        
        return {
            "type": self.class_names[top_idx].lower(),
            "confidence": round(float(predictions[top_idx]), 4)
        }
