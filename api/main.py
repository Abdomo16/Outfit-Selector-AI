import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import chat, recommend, visualization, wardrobe
from api.routes import auth as auth_routes
from api.routes import outfit_plans, saved_outfits, wear_logs
from inference.pipeline import InferencePipeline

load_dotenv()

_STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure photo storage directories exist before the pipeline starts
    (_STORAGE_ROOT / "wardrobe-images").mkdir(parents=True, exist_ok=True)
    app.state.pipeline = InferencePipeline()
    yield
    app.state.pipeline = None


app = FastAPI(title="Outfit Selector AI", lifespan=lifespan)

# Serve local garment photos at /static/
_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STORAGE_ROOT)), name="static")

app.include_router(auth_routes.router,   prefix="/auth",          tags=["auth"])
app.include_router(wardrobe.router,      prefix="/wardrobe",      tags=["wardrobe"])
app.include_router(recommend.router,     prefix="/recommend",     tags=["recommend"])
app.include_router(visualization.router, prefix="/visualize",     tags=["virtual try-on"])
app.include_router(chat.router,          prefix="/chat",          tags=["chat"])
app.include_router(saved_outfits.router, prefix="/saved-outfits", tags=["saved outfits"])
app.include_router(wear_logs.router,     prefix="/wear-logs",     tags=["wear logs"])
app.include_router(outfit_plans.router,  prefix="/outfit-plans",  tags=["outfit plans"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
