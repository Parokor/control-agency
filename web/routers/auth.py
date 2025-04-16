"""
Router para la autenticación en la interfaz web de Control Agency.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
from datetime import timedelta
from typing import Dict, List, Optional, Any

from web.dependencies import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    fake_users_db
)
from web.models import Token, User, UserCreate, UserLogin
from web.config import settings

router = APIRouter()

# Configurar plantillas Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Obtiene un token de acceso para un usuario.
    
    Args:
        form_data: Datos del formulario de inicio de sesión.
    
    Returns:
        Token: Token de acceso.
    
    Raises:
        HTTPException: Si las credenciales son incorrectas.
    """
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Muestra la página de inicio de sesión.
    
    Args:
        request: Objeto Request de FastAPI.
    
    Returns:
        HTMLResponse: Respuesta HTML con la página de inicio de sesión.
    """
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login", response_class=HTMLResponse)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Procesa el inicio de sesión.
    
    Args:
        request: Objeto Request de FastAPI.
        form_data: Datos del formulario de inicio de sesión.
    
    Returns:
        HTMLResponse: Respuesta HTML con la página de inicio de sesión o redirección al dashboard.
    """
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect username or password"}
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@router.get("/logout")
async def logout():
    """
    Cierra la sesión del usuario.
    
    Returns:
        RedirectResponse: Redirección a la página de inicio.
    """
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="access_token")
    return response

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Obtiene los datos del usuario actual.
    
    Args:
        current_user: Usuario actual.
    
    Returns:
        User: Datos del usuario actual.
    """
    return current_user
