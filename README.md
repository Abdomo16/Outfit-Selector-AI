# Outfit Selector AI

Outfit Selector AI is an intelligent wardrobe management and outfit recommendation backend API. It uses machine learning and computer vision to analyze user-uploaded photos, isolate clothing items, classify their features, and automatically recommend outfits tailored to specific occasions, seasons, and user preferences.

## Core Features

- Item Classification and Color Detection: Classifies the specific clothing type and extracts the dominant color hex values using K-Means clustering.
- Image Segmentation: Utilizes YOLOv8 driven instance segmentation to detect and cleanly crop multiple distinct articles of clothing from a single photograph.
- Attribute Extraction and Embeddings: Generates dense feature embeddings and detects high-level clothing attributes such as pattern and style.
- Wardrobe Database Management: Maintains user wardrobes with a persistent SQLite storage mechanism that tracks the isolated image and enriched attribute sets.
- Rule-based Outfit Recommendations: Applies predefined fashion constraints alongside cosine similarity measures to validate and present aesthetically cohesive outfit combinations.

## Project Structure

- /api: FastAPI application logic, including routing endpoints, database engine definitions, and Pydantic schemas.
- /datasets: Scripts handling dataset parsing and conversion for model training frameworks.
- /inference: Top-level orchestration pipeline connecting segmentation, classification, and embeddings logic.
- /models: The specialized module implementations for the rule engine, recommenders, image segmenters, and classifiers.
- /notebook_training: Exploratory and programmatic notebooks used for training detection and classification models.
- /rules: Configuration files governing valid fashion aesthetics based on occasion, color, and season.
- /tests: Dedicated unit tests and scripts for pipeline integration and logic validation.
- /weights: Dedicated storage for machine learning artifacts and parameter files necessary for inference.

## Quick Start Guide

1. Navigate to the project root directory and install all required dependencies:
   pip install -r requirements.txt

2. Ensure all corresponding model weights are correctly placed inside the /weights directory for inference resolution.

3. Start the application backend locally:
   uvicorn api.main:app --reload

The server will be available by default at http://127.0.0.1:8000. You can investigate the API routing structure and invoke the endpoints directly from the interactive docs available at http://127.0.0.1:8000/docs.

## Testing

To run the recommendation integration validations, invoke the corresponding test script:
python tests/test_phase5_recommend_script.py
