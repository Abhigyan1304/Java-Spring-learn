from fastapi import FastAPI
from app.api import sql_api, summary_api, other_api
from app.startup import on_startup, on_shutdown

app = FastAPI(title="Text2SQL Service with RAG + Guardrails")

# Register routers
app.include_router(sql_api.router, prefix="/sql", tags=["SQL"])
app.include_router(summary_api.router, prefix="/summary", tags=["Summary"])
app.include_router(other_api.router, prefix="/other", tags=["Other"])

# Startup / Shutdown hooks
@app.on_event("startup")
async def startup_event():
    await on_startup()

@app.on_event("shutdown")
async def shutdown_event():
    await on_shutdown()

@app.get("/")
def health_check():
    return {"status": "ok"}
