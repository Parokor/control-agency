"""
Router para la gestión de tareas en la interfaz web de Control Agency.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import Dict, List, Optional, Any

from web.dependencies import get_current_active_user
from web.models import User, TaskCreate, TaskResponse, TaskStatusEnum, TaskPriorityEnum, PerformancePrediction
from cloud.resource_scheduler import ResourceScheduler, Task, TaskPriority, TaskStatus, PerformancePredictor

router = APIRouter()

# Configurar plantillas Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))

# Crear instancia del programador de recursos
scheduler = ResourceScheduler()

# Crear instancia del predictor de rendimiento
predictor = PerformancePredictor()

@router.get("/", response_class=HTMLResponse)
async def tasks_page(
    request: Request,
    status: Optional[TaskStatusEnum] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Muestra la página de gestión de tareas.
    
    Args:
        request: Objeto Request de FastAPI.
        status: Estado de las tareas a mostrar.
        current_user: Usuario actual.
    
    Returns:
        HTMLResponse: Respuesta HTML con la página de gestión de tareas.
    """
    # Convertir el estado de string a enum
    task_status = None
    if status:
        task_status = TaskStatus[status]
    
    # Obtener las tareas
    tasks = scheduler.get_tasks(task_status)
    
    return templates.TemplateResponse(
        "tasks.html",
        {
            "request": request,
            "user": current_user,
            "tasks": tasks,
            "status_filter": status
        }
    )

@router.get("/create", response_class=HTMLResponse)
async def create_task_page(request: Request, current_user: User = Depends(get_current_active_user)):
    """
    Muestra la página de creación de tareas.
    
    Args:
        request: Objeto Request de FastAPI.
        current_user: Usuario actual.
    
    Returns:
        HTMLResponse: Respuesta HTML con la página de creación de tareas.
    """
    return templates.TemplateResponse(
        "create_task.html",
        {
            "request": request,
            "user": current_user,
            "priorities": TaskPriorityEnum,
            "platforms": ["colab", "paperspace", "runpod"]
        }
    )

@router.post("/", response_model=TaskResponse)
async def create_task(
    task_create: TaskCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Crea una nueva tarea.
    
    Args:
        task_create: Datos para crear la tarea.
        current_user: Usuario actual.
    
    Returns:
        TaskResponse: Datos de la tarea creada.
    
    Raises:
        HTTPException: Si la tarea no se pudo crear.
    """
    # Convertir los datos del modelo Pydantic a la clase Task
    task = Task(
        task_id=task_create.task_id,
        code=task_create.code,
        platform_preference=task_create.platform_preference or ["colab", "paperspace", "runpod"],
        priority=TaskPriority[task_create.priority],
        timeout=task_create.timeout or 3600
    )
    
    # Enviar la tarea al programador de recursos
    if scheduler.submit_task(task):
        # Obtener la tarea actualizada
        updated_task = scheduler.get_task(task_create.task_id)
        
        # Convertir la tarea a TaskResponse
        return TaskResponse(
            task_id=updated_task.task_id,
            status=TaskStatusEnum[updated_task.status.name],
            platform=updated_task.platform,
            instance_id=updated_task.instance_id,
            start_time=updated_task.start_time,
            end_time=updated_task.end_time,
            result=updated_task.result,
            error=updated_task.error
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task {task_create.task_id}"
        )

@router.post("/upload", response_model=TaskResponse)
async def upload_task(
    task_id: str = Form(...),
    code_file: UploadFile = File(...),
    priority: TaskPriorityEnum = Form(TaskPriorityEnum.MEDIUM),
    platforms: str = Form("colab,paperspace,runpod"),
    timeout: int = Form(3600),
    current_user: User = Depends(get_current_active_user)
):
    """
    Crea una nueva tarea a partir de un archivo.
    
    Args:
        task_id: ID de la tarea.
        code_file: Archivo con el código de la tarea.
        priority: Prioridad de la tarea.
        platforms: Plataformas preferidas (separadas por comas).
        timeout: Tiempo máximo de ejecución en segundos.
        current_user: Usuario actual.
    
    Returns:
        TaskResponse: Datos de la tarea creada.
    
    Raises:
        HTTPException: Si la tarea no se pudo crear.
    """
    # Leer el contenido del archivo
    code = await code_file.read()
    code_str = code.decode("utf-8")
    
    # Convertir las plataformas a una lista
    platform_list = platforms.split(",")
    
    # Crear la tarea
    task_create = TaskCreate(
        task_id=task_id,
        code=code_str,
        platform_preference=platform_list,
        priority=priority,
        timeout=timeout
    )
    
    # Llamar a la función de creación de tareas
    return await create_task(task_create, current_user)

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Obtiene los datos de una tarea.
    
    Args:
        task_id: ID de la tarea.
        current_user: Usuario actual.
    
    Returns:
        TaskResponse: Datos de la tarea.
    
    Raises:
        HTTPException: Si la tarea no existe.
    """
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Convertir la tarea a TaskResponse
    return TaskResponse(
        task_id=task.task_id,
        status=TaskStatusEnum[task.status.name],
        platform=task.platform,
        instance_id=task.instance_id,
        start_time=task.start_time,
        end_time=task.end_time,
        result=task.result,
        error=task.error
    )

@router.delete("/{task_id}", response_model=Dict[str, str])
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Cancela una tarea.
    
    Args:
        task_id: ID de la tarea.
        current_user: Usuario actual.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    
    Raises:
        HTTPException: Si la tarea no se pudo cancelar.
    """
    if scheduler.cancel_task(task_id):
        return {"status": "success", "message": f"Task {task_id} cancelled successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task {task_id}"
        )

@router.get("/predict/{task_id}", response_model=List[PerformancePrediction])
async def predict_task_performance(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Predice el rendimiento de una tarea en diferentes plataformas.
    
    Args:
        task_id: ID de la tarea.
        current_user: Usuario actual.
    
    Returns:
        List[PerformancePrediction]: Predicciones de rendimiento.
    
    Raises:
        HTTPException: Si la tarea no existe.
    """
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Predecir el rendimiento en diferentes plataformas
    predictions = []
    for platform in ["colab", "paperspace", "runpod"]:
        predicted_time = predictor.predict_execution_time(task, platform)
        if predicted_time:
            predictions.append(
                PerformancePrediction(
                    task_id=task_id,
                    platform=platform,
                    predicted_time=predicted_time,
                    confidence=0.8  # Valor de confianza fijo por ahora
                )
            )
    
    return predictions
