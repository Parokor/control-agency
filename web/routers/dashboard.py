"""
Router para el dashboard en la interfaz web de Control Agency.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import Dict, List, Optional, Any

from web.dependencies import get_current_active_user
from web.models import User, SchedulerStatus, TaskStatusEnum
from cloud.resource_scheduler import ResourceScheduler

router = APIRouter()

# Configurar plantillas Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))

# Crear instancia del programador de recursos
scheduler = ResourceScheduler()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_active_user)):
    """
    Muestra el dashboard principal.
    
    Args:
        request: Objeto Request de FastAPI.
        current_user: Usuario actual.
    
    Returns:
        HTMLResponse: Respuesta HTML con el dashboard.
    """
    # Obtener el estado del programador de recursos
    scheduler_status = scheduler.get_scheduler_status()
    
    # Obtener las tareas
    tasks = scheduler.get_tasks()
    
    # Contar tareas por estado
    task_counts = {
        "pending": len(scheduler.get_tasks(TaskStatusEnum.PENDING)),
        "running": len(scheduler.get_tasks(TaskStatusEnum.RUNNING)),
        "completed": len(scheduler.get_tasks(TaskStatusEnum.COMPLETED)),
        "failed": len(scheduler.get_tasks(TaskStatusEnum.FAILED)),
        "cancelled": len(scheduler.get_tasks(TaskStatusEnum.CANCELLED))
    }
    
    # Obtener el estado de las plataformas
    platforms_status = {}
    for platform in scheduler_status["platforms"]:
        platforms_status[platform] = scheduler.get_platform_status(platform)
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "scheduler_status": scheduler_status,
            "tasks": tasks,
            "task_counts": task_counts,
            "platforms_status": platforms_status
        }
    )

@router.get("/status", response_model=SchedulerStatus)
async def get_scheduler_status(current_user: User = Depends(get_current_active_user)):
    """
    Obtiene el estado del programador de recursos.
    
    Args:
        current_user: Usuario actual.
    
    Returns:
        SchedulerStatus: Estado del programador de recursos.
    """
    return scheduler.get_scheduler_status()

@router.post("/start")
async def start_scheduler(current_user: User = Depends(get_current_active_user)):
    """
    Inicia el programador de recursos.
    
    Args:
        current_user: Usuario actual.
    
    Returns:
        Dict[str, Any]: Resultado de la operación.
    """
    scheduler.start()
    return {"status": "success", "message": "Scheduler started"}

@router.post("/stop")
async def stop_scheduler(current_user: User = Depends(get_current_active_user)):
    """
    Detiene el programador de recursos.
    
    Args:
        current_user: Usuario actual.
    
    Returns:
        Dict[str, Any]: Resultado de la operación.
    """
    scheduler.stop()
    return {"status": "success", "message": "Scheduler stopped"}
