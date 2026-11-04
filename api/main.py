from fastapi import FastAPI
from api.routes import wardrobe, recommend, visualization, chat
from inference.pipeline import InferencePipeline
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = InferencePipeline()
    yield
    app.state.pipeline = None

app = FastAPI(title="Outfit Selector AI", lifespan=lifespan)

app.include_router(wardrobe.router, prefix="/wardrobe", tags=["wardrobe"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
app.include_router(visualization.router, prefix="/visualize", tags=["virtual try-on"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
