"""
Aplicación principal para la interfaz web de Control Agency.
"""

import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uvicorn

from web.routers import platforms, tasks, auth, dashboard
from web.config import settings
from web.dependencies import get_token_manager, get_current_user

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    
    Returns:
        FastAPI: Aplicación FastAPI configurada.
    """
    # Crear la aplicación FastAPI
    app = FastAPI(
        title="Control Agency",
        description="Interfaz web para gestionar recursos en la nube",
        version="1.0.0"
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Montar archivos estáticos
    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
    
    # Configurar plantillas Jinja2
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
    
    # Incluir routers
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(platforms.router, prefix="/platforms", tags=["platforms"])
    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    
    # Ruta principal
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """
        Ruta principal que muestra la página de inicio.
        
        Args:
            request: Objeto Request de FastAPI.
        
        Returns:
            HTMLResponse: Respuesta HTML con la página de inicio.
        """
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Manejador de errores
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """
        Manejador de excepciones HTTP.
        
        Args:
            request: Objeto Request de FastAPI.
            exc: Excepción HTTP.
        
        Returns:
            JSONResponse: Respuesta JSON con el error.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    
    return app

def start_server():
    """
    Inicia el servidor web.
    """
    app = create_app()
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    start_server()
