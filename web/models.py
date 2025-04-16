"""
Modelos para la interfaz web de Control Agency.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class TokenData(BaseModel):
    """
    Datos del token de autenticación.
    """
    username: Optional[str] = None

class Token(BaseModel):
    """
    Token de autenticación.
    """
    access_token: str
    token_type: str

class User(BaseModel):
    """
    Usuario de la aplicación.
    """
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    hashed_password: str

class UserInDB(User):
    """
    Usuario en la base de datos.
    """
    hashed_password: str

class UserCreate(BaseModel):
    """
    Datos para crear un usuario.
    """
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    """
    Datos para iniciar sesión.
    """
    username: str
    password: str

class PlatformType(str, Enum):
    """
    Tipos de plataformas en la nube.
    """
    COLAB = "colab"
    PAPERSPACE = "paperspace"
    RUNPOD = "runpod"

class PlatformToken(BaseModel):
    """
    Token de autenticación para una plataforma en la nube.
    """
    platform: PlatformType
    token: str

class PlatformStatus(BaseModel):
    """
    Estado de una plataforma en la nube.
    """
    platform: PlatformType
    status: str
    message: Optional[str] = None
    instances: Optional[List[Dict[str, Any]]] = None

class TaskPriorityEnum(str, Enum):
    """
    Prioridades de las tareas.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TaskStatusEnum(str, Enum):
    """
    Estados de las tareas.
    """
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class TaskCreate(BaseModel):
    """
    Datos para crear una tarea.
    """
    task_id: str
    code: str
    platform_preference: Optional[List[PlatformType]] = None
    priority: TaskPriorityEnum = TaskPriorityEnum.MEDIUM
    timeout: Optional[int] = 3600

class TaskResponse(BaseModel):
    """
    Respuesta con los datos de una tarea.
    """
    task_id: str
    status: TaskStatusEnum
    platform: Optional[PlatformType] = None
    instance_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SchedulerStatus(BaseModel):
    """
    Estado del programador de recursos.
    """
    running: bool
    max_concurrent_tasks: int
    pending_tasks: int
    total_tasks: int
    platforms: List[PlatformType]
    instances: Dict[PlatformType, List[str]]

class PerformancePrediction(BaseModel):
    """
    Predicción de rendimiento para una tarea.
    """
    task_id: str
    platform: PlatformType
    predicted_time: float
    confidence: float
