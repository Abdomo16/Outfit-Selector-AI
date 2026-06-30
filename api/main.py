from fastapi import FastAPI
from api.routes import wardrobe, recommend

app = FastAPI(title="Outfit Selector AI")

app.include_router(wardrobe.router, prefix="/wardrobe", tags=["wardrobe"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
