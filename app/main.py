from fastapi import FastAPI
from app.api.routes import router as api_router
from app.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )
    
    app.include_router(api_router)
    
    return app

app = create_app()
