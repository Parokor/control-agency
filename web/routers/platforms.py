"""
Router para la gestión de plataformas en la nube en la interfaz web de Control Agency.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import Dict, List, Optional, Any

from web.dependencies import get_current_active_user, get_token_manager
from web.models import User, PlatformToken, PlatformStatus, PlatformType
from auth.token_manager import TokenManager
from cloud.platform_manager import get_platform_instance

router = APIRouter()

# Configurar plantillas Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))

@router.get("/", response_class=HTMLResponse)
async def platforms_page(request: Request, current_user: User = Depends(get_current_active_user)):
    """
    Muestra la página de gestión de plataformas.
    
    Args:
        request: Objeto Request de FastAPI.
        current_user: Usuario actual.
    
    Returns:
        HTMLResponse: Respuesta HTML con la página de gestión de plataformas.
    """
    # Obtener el gestor de tokens
    token_manager = get_token_manager()
    
    # Obtener las plataformas con tokens guardados
    platforms = token_manager.list_platforms()
    
    # Obtener el estado de las plataformas
    platforms_status = {}
    for platform in platforms:
        try:
            platform_instance = get_platform_instance(platform, token_manager)
            if platform_instance:
                connection_status = platform_instance.test_connection()
                platforms_status[platform] = {
                    "connected": connection_status,
                    "message": "Connected" if connection_status else "Connection failed"
                }
            else:
                platforms_status[platform] = {
                    "connected": False,
                    "message": "Platform not supported"
                }
        except Exception as e:
            platforms_status[platform] = {
                "connected": False,
                "message": str(e)
            }
    
    return templates.TemplateResponse(
        "platforms.html",
        {
            "request": request,
            "user": current_user,
            "platforms": platforms,
            "platforms_status": platforms_status
        }
    )

@router.post("/token", response_model=Dict[str, str])
async def save_platform_token(
    platform_token: PlatformToken,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Guarda un token para una plataforma en la nube.
    
    Args:
        platform_token: Token de la plataforma.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    """
    if token_manager.save_token(platform_token.platform, platform_token.token):
        return {"status": "success", "message": f"Token for {platform_token.platform} saved successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save token for {platform_token.platform}"
        )

@router.get("/token/{platform}", response_model=Dict[str, str])
async def verify_platform_token(
    platform: PlatformType,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Verifica un token para una plataforma en la nube.
    
    Args:
        platform: Plataforma en la nube.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    """
    if token_manager.verify_token(platform):
        return {"status": "success", "message": f"Token for {platform} is valid"}
    else:
        return {"status": "error", "message": f"Token for {platform} is invalid or not found"}

@router.delete("/token/{platform}", response_model=Dict[str, str])
async def delete_platform_token(
    platform: PlatformType,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Elimina un token para una plataforma en la nube.
    
    Args:
        platform: Plataforma en la nube.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    """
    if token_manager.delete_token(platform):
        return {"status": "success", "message": f"Token for {platform} deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete token for {platform}"
        )

@router.get("/status/{platform}", response_model=PlatformStatus)
async def get_platform_status(
    platform: PlatformType,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Obtiene el estado de una plataforma en la nube.
    
    Args:
        platform: Plataforma en la nube.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        PlatformStatus: Estado de la plataforma.
    """
    platform_instance = get_platform_instance(platform, token_manager)
    if not platform_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Platform {platform} not found"
        )
    
    # Verificar la conexión
    if not platform_instance.test_connection():
        return PlatformStatus(
            platform=platform,
            status="ERROR",
            message=f"Failed to connect to {platform}"
        )
    
    # Obtener las instancias
    instances = platform_instance.list_instances()
    
    return PlatformStatus(
        platform=platform,
        status="OK",
        instances=instances or []
    )

@router.post("/instances/{platform}", response_model=Dict[str, Any])
async def create_platform_instance(
    platform: PlatformType,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Crea una instancia en una plataforma en la nube.
    
    Args:
        platform: Plataforma en la nube.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, Any]: Resultado de la operación.
    """
    platform_instance = get_platform_instance(platform, token_manager)
    if not platform_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Platform {platform} not found"
        )
    
    instance_id = platform_instance.create_instance()
    if not instance_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create instance on {platform}"
        )
    
    return {
        "status": "success",
        "message": f"Instance created successfully on {platform}",
        "instance_id": instance_id
    }

@router.delete("/instances/{platform}/{instance_id}", response_model=Dict[str, str])
async def stop_platform_instance(
    platform: PlatformType,
    instance_id: str,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Detiene una instancia en una plataforma en la nube.
    
    Args:
        platform: Plataforma en la nube.
        instance_id: ID de la instancia.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    """
    platform_instance = get_platform_instance(platform, token_manager)
    if not platform_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Platform {platform} not found"
        )
    
    if platform_instance.stop_instance(instance_id):
        return {"status": "success", "message": f"Instance {instance_id} stopped successfully on {platform}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop instance {instance_id} on {platform}"
        )

@router.post("/anti-timeout/{platform}/{instance_id}", response_model=Dict[str, str])
async def implement_anti_timeout(
    platform: PlatformType,
    instance_id: str,
    current_user: User = Depends(get_current_active_user),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Implementa mecanismos anti-timeout en una instancia.
    
    Args:
        platform: Plataforma en la nube.
        instance_id: ID de la instancia.
        current_user: Usuario actual.
        token_manager: Gestor de tokens.
    
    Returns:
        Dict[str, str]: Resultado de la operación.
    """
    platform_instance = get_platform_instance(platform, token_manager)
    if not platform_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Platform {platform} not found"
        )
    
    if platform_instance.implement_anti_timeout(instance_id):
        return {"status": "success", "message": f"Anti-timeout mechanisms implemented successfully on instance {instance_id} on {platform}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to implement anti-timeout mechanisms on instance {instance_id} on {platform}"
        )
