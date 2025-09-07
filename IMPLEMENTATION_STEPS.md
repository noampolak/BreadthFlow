# 🚀 BreadthFlow Dashboard Implementation Steps
## FastAPI + React Modern Dashboard

> **Goal**: Step-by-step implementation of the modern FastAPI + React dashboard with proper Docker configuration and port management.

---

## 📋 **Implementation Overview**

### **Port Configuration**
- **Current Dashboard**: Port 8083 (Python HTTP server)
- **New FastAPI Backend**: Port 8005 (FastAPI)
- **New React Frontend**: Port 3005 (React dev server)
- **Production React**: Port 80 (Nginx)

### **Docker Services**
- **breadthflow-api**: FastAPI backend (port 8005)
- **breadthflow-frontend**: React frontend (port 3005)
- **breadthflow-nginx**: Production frontend (port 80)

---

## 🛠️ **Step 1: Create FastAPI Backend Structure**

### **1.1 Create Directory Structure**
```bash
# Navigate to BreadthFlow root
cd /Users/noampolak/MyPrivateRepos/BreadthFlow

# Create FastAPI app structure
mkdir -p fastapi_app/{core,apps/{dashboard,pipeline,signals,infrastructure,commands,training,parameters},shared,migrations,tests/apps}

# Create app subdirectories
for app in dashboard pipeline signals infrastructure commands training parameters; do
    mkdir -p fastapi_app/apps/$app
done
```

### **1.2 Create Core Configuration**
```bash
# Create core files
touch fastapi_app/core/{__init__.py,config.py,database.py,security.py,dependencies.py,exceptions.py}
touch fastapi_app/shared/{__init__.py,models.py,schemas.py,utils.py,websocket.py}
touch fastapi_app/main.py
touch fastapi_app/requirements.txt
touch fastapi_app/Dockerfile
touch fastapi_app/.env
```

### **1.3 Create App Files**
```bash
# Create files for each app
for app in dashboard pipeline signals infrastructure commands training parameters; do
    touch fastapi_app/apps/$app/{__init__.py,models.py,schemas.py,routes.py,utils.py,tasks.py}
done
```

---

## 🐍 **Step 2: Implement FastAPI Backend**

### **2.1 Core Configuration**
```python
# fastapi_app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://pipeline:pipeline123@postgres:5432/breadthflow"
    
    # API
    api_title: str = "BreadthFlow API v2.0"
    api_version: str = "2.0.0"
    api_description: str = "Modern FastAPI backend for BreadthFlow dashboard"
    
    # Security
    secret_key: str = "breadthflow-secret-key-2024"
    access_token_expire_minutes: int = 30
    
    # External Services
    spark_command_server_url: str = "http://spark-master:8081"
    redis_url: str = "redis://redis:6379"
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:3005",  # React dev server
        "http://localhost:8083",  # Legacy dashboard
        "http://localhost:80"     # Production frontend
    ]
    
    # App Settings
    debug: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### **2.2 Database Configuration**
```python
# fastapi_app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from core.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### **2.3 Main FastAPI Application**
```python
# fastapi_app/main.py
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

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# WebSocket manager
websocket_manager = WebSocketManager()

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
    openapi_url="/openapi.json"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

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
                {"type": "heartbeat", "timestamp": "2024-01-01T00:00:00Z"}, 
                websocket
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
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": settings.api_version,
        "database": "connected",
        "spark_server": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8005, 
        reload=settings.debug
    )
```

### **2.4 WebSocket Manager**
```python
# fastapi_app/shared/websocket.py
from fastapi import WebSocket
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, metadata: Dict[str, Any] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = metadata or {}
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_pipeline_update(self, pipeline_id: str, status: str):
        message = {
            "type": "pipeline_update",
            "pipeline_id": pipeline_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    async def broadcast_signal_update(self, signal_data: dict):
        message = {
            "type": "signal_update",
            "data": signal_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    async def broadcast_dashboard_update(self, dashboard_data: dict):
        message = {
            "type": "dashboard_update",
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
```

---

## 🎯 **Step 3: Implement Pipeline App (Example)**

### **3.1 Pipeline Models**
```python
# fastapi_app/apps/pipeline/models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Float
from sqlalchemy.sql import func
from shared.models import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(255), unique=True, index=True)
    command = Column(String(500))
    status = Column(String(50), default="pending")
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Float, nullable=True)
    error_message = Column(String(1000), nullable=True)
    metadata = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<PipelineRun(id={self.id}, run_id='{self.run_id}', status='{self.status}')>"
```

### **3.2 Pipeline Schemas**
```python
# fastapi_app/apps/pipeline/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class PipelineConfig(BaseModel):
    mode: str = Field(..., description="Pipeline mode (demo, small, medium, full)")
    interval: str = Field(..., description="Execution interval")
    timeframe: str = Field(..., description="Data timeframe")
    symbols: Optional[str] = Field(None, description="Custom symbols list")
    data_source: str = Field("yfinance", description="Data source")

class PipelineRunBase(BaseModel):
    command: str = Field(..., description="Pipeline command to execute")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class PipelineRunCreate(PipelineRunBase):
    pass

class PipelineRunResponse(PipelineRunBase):
    id: int
    run_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True

class PipelineStatusResponse(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    running_runs: int
    success_rate: float
    average_duration: float
```

### **3.3 Pipeline Routes**
```python
# fastapi_app/apps/pipeline/routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from core.dependencies import get_db
from .schemas import PipelineRunCreate, PipelineRunResponse, PipelineConfig, PipelineStatusResponse
from .utils import PipelineService
from .tasks import start_pipeline_task, stop_pipeline_task

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

@router.get("/runs", response_model=List[PipelineRunResponse])
async def get_pipeline_runs(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get paginated pipeline runs with optional status filter"""
    service = PipelineService(db)
    return await service.get_runs(skip=skip, limit=limit, status=status)

@router.post("/start", response_model=PipelineRunResponse)
async def start_pipeline(
    config: PipelineConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new pipeline with background task execution"""
    service = PipelineService(db)
    
    # Check if pipeline is already running
    running_pipeline = await service.get_running_pipeline()
    if running_pipeline:
        raise HTTPException(
            status_code=400, 
            detail=f"Pipeline {running_pipeline.run_id} is already running"
        )
    
    # Create pipeline run record
    pipeline_run = await service.create_pipeline_run(config)
    
    # Start pipeline in background
    background_tasks.add_task(
        start_pipeline_task, 
        pipeline_run.id, 
        config.dict()
    )
    
    return pipeline_run

@router.post("/stop/{run_id}")
async def stop_pipeline(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Stop a running pipeline"""
    service = PipelineService(db)
    pipeline_run = await service.get_run_by_id(run_id)
    
    if not pipeline_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    if pipeline_run.status != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")
    
    # Stop pipeline in background
    background_tasks.add_task(stop_pipeline_task, run_id)
    
    return {"message": f"Pipeline {run_id} stop requested"}

@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(db: Session = Depends(get_db)):
    """Get current pipeline status and statistics"""
    service = PipelineService(db)
    return await service.get_status()

@router.get("/runs/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Get specific pipeline run details"""
    service = PipelineService(db)
    pipeline_run = await service.get_run_by_id(run_id)
    
    if not pipeline_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    return pipeline_run
```

### **3.4 Pipeline Business Logic**
```python
# fastapi_app/apps/pipeline/utils.py
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from .models import PipelineRun
from .schemas import PipelineRunCreate, PipelineConfig, PipelineStatus, PipelineStatusResponse
import uuid
from datetime import datetime

class PipelineService:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_pipeline_run(self, config: PipelineConfig) -> PipelineRun:
        """Create a new pipeline run record"""
        run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        pipeline_run = PipelineRun(
            run_id=run_id,
            command=f"spark_streaming_start_{config.mode}",
            status=PipelineStatus.PENDING,
            metadata=config.dict()
        )
        
        self.db.add(pipeline_run)
        self.db.commit()
        self.db.refresh(pipeline_run)
        
        return pipeline_run
    
    async def get_runs(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        status: Optional[str] = None
    ) -> List[PipelineRun]:
        """Get paginated pipeline runs"""
        query = self.db.query(PipelineRun)
        
        if status:
            query = query.filter(PipelineRun.status == status)
        
        return query.order_by(desc(PipelineRun.start_time)).offset(skip).limit(limit).all()
    
    async def get_run_by_id(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run by ID"""
        return self.db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()
    
    async def get_running_pipeline(self) -> Optional[PipelineRun]:
        """Get currently running pipeline"""
        return self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.RUNNING
        ).first()
    
    async def get_status(self) -> PipelineStatusResponse:
        """Get pipeline status and statistics"""
        total_runs = self.db.query(PipelineRun).count()
        successful_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.COMPLETED
        ).count()
        failed_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.FAILED
        ).count()
        running_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.RUNNING
        ).count()
        
        avg_duration = self.db.query(func.avg(PipelineRun.duration)).filter(
            PipelineRun.duration.isnot(None)
        ).scalar() or 0
        
        return PipelineStatusResponse(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            running_runs=running_runs,
            success_rate=round((successful_runs / total_runs * 100) if total_runs > 0 else 0, 2),
            average_duration=round(avg_duration, 2)
        )
```

### **3.5 Pipeline Background Tasks**
```python
# fastapi_app/apps/pipeline/tasks.py
import asyncio
import httpx
from sqlalchemy.orm import Session
from core.database import get_db
from .models import PipelineRun
from .schemas import PipelineConfig
from shared.websocket import WebSocketManager
from datetime import datetime

# Global WebSocket manager (will be injected)
websocket_manager = None

def set_websocket_manager(manager: WebSocketManager):
    global websocket_manager
    websocket_manager = manager

async def start_pipeline_task(pipeline_id: int, config: dict):
    """Background task to start pipeline execution"""
    db = next(get_db())
    try:
        # Update status to running
        pipeline_run = db.query(PipelineRun).filter(PipelineRun.id == pipeline_id).first()
        if pipeline_run:
            pipeline_run.status = "running"
            pipeline_run.is_active = True
            db.commit()
            
            # Broadcast update via WebSocket
            if websocket_manager:
                await websocket_manager.broadcast_pipeline_update(
                    pipeline_run.run_id, 
                    "running"
                )
        
        # Execute pipeline via Spark command server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://spark-master:8081/execute",
                json={
                    "command": "spark_streaming_start_demo",
                    "parameters": config
                },
                timeout=300.0
            )
            
            if response.status_code == 200:
                # Update status to completed
                pipeline_run.status = "completed"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "completed"
                    )
            else:
                # Update status to failed
                pipeline_run.status = "failed"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.error_message = f"HTTP {response.status_code}: {response.text}"
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "failed"
                    )
            
            db.commit()
            
    except Exception as e:
        # Update status to failed
        pipeline_run = db.query(PipelineRun).filter(PipelineRun.id == pipeline_id).first()
        if pipeline_run:
            pipeline_run.status = "failed"
            pipeline_run.is_active = False
            pipeline_run.end_time = datetime.now()
            pipeline_run.error_message = str(e)
            db.commit()
            
            # Broadcast update via WebSocket
            if websocket_manager:
                await websocket_manager.broadcast_pipeline_update(
                    pipeline_run.run_id, 
                    "failed"
                )
    finally:
        db.close()

async def stop_pipeline_task(run_id: str):
    """Background task to stop pipeline execution"""
    db = next(get_db())
    try:
        # Stop pipeline via Spark command server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://spark-master:8081/execute",
                json={
                    "command": "spark_streaming_stop",
                    "parameters": {"run_id": run_id}
                },
                timeout=30.0
            )
            
            # Update status to stopped
            pipeline_run = db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()
            if pipeline_run:
                pipeline_run.status = "stopped"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                db.commit()
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "stopped"
                    )
                
    except Exception as e:
        # Log error but don't fail the task
        print(f"Error stopping pipeline {run_id}: {e}")
    finally:
        db.close()
```

---

## 🐳 **Step 4: Docker Configuration**

### **4.1 FastAPI Dockerfile**
```dockerfile
# fastapi_app/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
```

### **4.2 FastAPI Requirements**
```txt
# fastapi_app/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
pydantic==2.5.0
pydantic-settings==2.1.0
httpx==0.25.2
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
redis==5.0.1
celery==5.3.4
```

### **4.3 React Frontend Setup**
```bash
# Create React app
cd /Users/noampolak/MyPrivateRepos/BreadthFlow
npx create-react-app frontend --template typescript
cd frontend

# Install additional dependencies
npm install @reduxjs/toolkit react-redux
npm install react-router-dom
npm install axios
npm install @types/react-router-dom
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
```

### **4.4 React Dockerfile**
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Install serve to run the app
RUN npm install -g serve

# Expose port
EXPOSE 3005

# Start the app
CMD ["serve", "-s", "build", "-l", "3005"]
```

---

## 🐳 **Step 5: Docker Compose Integration**

### **5.1 Update Docker Compose**
```yaml
# Add to existing docker-compose.yml
services:
  # FastAPI Backend
  breadthflow-api:
    build:
      context: ./fastapi_app
      dockerfile: Dockerfile
    ports:
      - "8005:8005"
    environment:
      - DATABASE_URL=postgresql://pipeline:pipeline123@postgres:5432/breadthflow
      - REDIS_URL=redis://redis:6379
      - SPARK_COMMAND_SERVER_URL=http://spark-master:8081
      - DEBUG=true
    depends_on:
      - postgres
      - redis
    volumes:
      - ./fastapi_app:/app
    networks:
      - breadthflow-network
    restart: unless-stopped

  # React Frontend (Development)
  breadthflow-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3005:3005"
    environment:
      - REACT_APP_API_URL=http://localhost:8005
      - REACT_APP_WS_URL=ws://localhost:8005/ws
    volumes:
      - ./frontend:/app
      - /app/node_modules
    networks:
      - breadthflow-network
    restart: unless-stopped

  # Production Frontend (Nginx)
  breadthflow-nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend/build:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - breadthflow-frontend
    networks:
      - breadthflow-network
    restart: unless-stopped

  # Redis for caching and WebSocket
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - breadthflow-network
    restart: unless-stopped

volumes:
  redis_data:

networks:
  breadthflow-network:
    driver: bridge
```

### **5.2 Nginx Configuration**
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server breadthflow-api:8005;
    }

    server {
        listen 80;
        server_name localhost;

        # Serve React app
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;
        }

        # Proxy API requests to FastAPI
        location /api/ {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Proxy WebSocket requests
        location /ws {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

---

## 🚀 **Step 6: Build and Run**

### **6.1 Build Commands**
```bash
# Build FastAPI backend
cd fastapi_app
docker build -t breadthflow-api .

# Build React frontend
cd ../frontend
docker build -t breadthflow-frontend .

# Build all services
cd ..
docker-compose build
```

### **6.2 Run Commands**
```bash
# Start all services
docker-compose up -d

# Start only new services
docker-compose up -d breadthflow-api breadthflow-frontend redis

# View logs
docker-compose logs -f breadthflow-api
docker-compose logs -f breadthflow-frontend
```

### **6.3 Access Points**
- **FastAPI Backend**: http://localhost:8005
- **FastAPI Docs**: http://localhost:8005/docs
- **React Frontend**: http://localhost:3005
- **Production Frontend**: http://localhost:80
- **Legacy Dashboard**: http://localhost:8083 (still running)

---

## 🧪 **Step 7: Testing**

### **7.1 Test FastAPI Endpoints**
```bash
# Health check
curl http://localhost:8005/health

# Get pipeline runs
curl http://localhost:8005/api/pipeline/runs

# Start pipeline
curl -X POST http://localhost:8005/api/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "demo", "interval": "5m", "timeframe": "1day"}'

# Get pipeline status
curl http://localhost:8005/api/pipeline/status
```

### **7.2 Test WebSocket**
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8005/ws');
ws.onopen = () => console.log('Connected to WebSocket');
ws.onmessage = (event) => console.log('Received:', JSON.parse(event.data));
```

---

## 📊 **Step 8: Port Management**

### **8.1 Port Allocation**
| Service | Port | Purpose |
|---------|------|---------|
| **Legacy Dashboard** | 8083 | Current Python HTTP server |
| **FastAPI Backend** | 8005 | New API backend |
| **React Dev** | 3005 | Development frontend |
| **Production Frontend** | 80 | Production frontend (Nginx) |
| **Redis** | 6379 | Caching and WebSocket |
| **PostgreSQL** | 5432 | Database (existing) |
| **Spark Master** | 8080 | Spark UI (existing) |
| **Spark Command API** | 8081 | Command execution (existing) |

### **8.2 Migration Strategy**
1. **Phase 1**: Run both dashboards in parallel (ports 8083 + 8005/3005)
2. **Phase 2**: Test new dashboard thoroughly
3. **Phase 3**: Switch traffic to new dashboard (port 80)
4. **Phase 4**: Deprecate legacy dashboard (port 8083)

---

## 🎯 **Next Steps**

1. **Create the directory structure** using the commands above
2. **Implement the core FastAPI files** (config, database, main.py)
3. **Build the pipeline app** as shown in the example
4. **Create Docker files** and test locally
5. **Update docker-compose.yml** with new services
6. **Test the integration** with existing BreadthFlow system

This implementation gives you a modern, scalable, and maintainable dashboard system while preserving all existing functionality!
