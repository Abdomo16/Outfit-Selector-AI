from fastapi import FastAPI
from api.routes import wardrobe, recommend
from inference.pipeline import InferencePipeline
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = InferencePipeline()
    yield
    app.state.pipeline = None

app = FastAPI(title="Outfit Selector AI", lifespan=lifespan)

app.include_router(wardrobe.router, prefix="/wardrobe", tags=["wardrobe"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
