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

## Setup Guide

### Local Development
1. Clone the repository and navigate to the project directory.
2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and fill in your Supabase credentials (`SUPABASE_SERVICE_KEY`, `SUPABASE_ANON_KEY`).
4. Ensure all corresponding model weights are correctly placed inside the `weights/` directory (`yolov8m-seg.pt`, etc.).
5. Start the application backend locally:
   ```bash
   uvicorn api.main:app --reload
   ```

### Supabase Setup
The backend uses Supabase for persistence and image storage:
- The `wardrobe_items` table stores segmented clothing metadata.
- The `wardrobe-images` storage bucket holds the cropped item images.
- On the first upload, the bucket is created automatically if it does not exist.

### Docker Deployment (Production)
Build and run the production-ready Docker container. Make sure your `.env` file contains the Supabase credentials:
```bash
docker build -t outfit-selector .
docker run -p 8000:8000 --env-file .env outfit-selector
```

The server will be available at `http://127.0.0.1:8000`.

## API Documentation

- `POST /wardrobe/upload`: Accepts an image upload. Returns a JSON schema of all segmented items identified in the image, computing their category, color, attributes, and 512-dimensional embeddings.
- `POST /recommend`: Accepts a collection of wardrobe items and constraints (occasion/season). Applies fashion rules and similarity scoring to output the best outfit combination.

Full interactive documentation is available dynamically at `http://127.0.0.1:8000/docs`.

## Datasets & Training

### Dataset Requirements
- **Segmentation**: We use the Roboflow DeepFashion2 dataset for YOLOv8 instance segmentation. Ensure the data is properly formatted for YOLOv8.
- **Classification**: Uses standard labeled clothing datasets categorized by `type` and hierarchical attributes.

### Training the Models
Training notebooks and utilities are located in `notebook_training/`.
1. Run the YOLOv8 training sequence to output `.pt` files.
2. Export the final weights configuration as `best.pt` and move them to the `weights/` directory prior to deployment.

## Testing

To run the recommendation integration validations, invoke the corresponding test script:
python tests/test_phase5_recommend_script.py
