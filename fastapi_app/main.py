from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging

from core.config import settings
from core.database import engine, Base
from shared.websocket import WebSocketManager

# Import all app routers
from apps.dashboard.routes import router as dashboard_router
from apps.pipeline.routes import router as pipeline_router
from apps.signals.routes import router as signals_router
from apps.infrastructure.routes import router as infrastructure_router
from apps.commands.routes import router as commands_router
from apps.training.routes import router as training_router
from apps.parameters.routes import router as parameters_router

# Import all models to ensure tables are created
from apps.dashboard.models import *
from apps.pipeline.models import *
from apps.signals.models import *
from apps.commands.models import *
from apps.training.models import *
from apps.parameters.models import *

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# WebSocket manager
websocket_manager = WebSocketManager()

# Set WebSocket manager for tasks
from apps.pipeline.tasks import set_websocket_manager

set_websocket_manager(websocket_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting BreadthFlow API v2.0...")
    logger.info(f"📊 Database: {settings.database_url}")
    logger.info(f"🔗 Spark Server: {settings.spark_command_server_url}")
    yield
    # Shutdown
    logger.info("🛑 Shutting down BreadthFlow API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Include routers
app.include_router(dashboard_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(signals_router, prefix="/api")
app.include_router(infrastructure_router, prefix="/api")
app.include_router(commands_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(parameters_router, prefix="/api")


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(30)
            await websocket_manager.send_personal_message(
                {"type": "heartbeat", "timestamp": "2024-01-01T00:00:00Z"}, websocket
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/")
async def root():
    return {
        "message": "BreadthFlow API v2.0",
        "version": settings.api_version,
        "docs": "/docs",
        "websocket": "/ws",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.api_version, "database": "connected", "spark_server": "connected"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=settings.debug)
