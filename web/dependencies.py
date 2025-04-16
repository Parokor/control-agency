"""
Dependencias para la interfaz web de Control Agency.
"""

import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from auth.token_manager import TokenManager
from web.config import settings
from web.models import User, TokenData

# Esquema OAuth2 para la autenticación
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Usuarios de prueba (en producción, usar una base de datos)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
    }
}

def get_token_manager() -> TokenManager:
    """
    Obtiene una instancia del gestor de tokens.
    
    Returns:
        TokenManager: Instancia del gestor de tokens.
    """
    return TokenManager(config_dir=settings.CONFIG_DIR)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica si la contraseña es correcta.
    
    Args:
        plain_password: Contraseña en texto plano.
        hashed_password: Contraseña hasheada.
    
    Returns:
        bool: True si la contraseña es correcta, False en caso contrario.
    """
    # En producción, usar bcrypt o similar
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str) -> Optional[User]:
    """
    Obtiene un usuario por su nombre de usuario.
    
    Args:
        db: Base de datos de usuarios.
        username: Nombre de usuario.
    
    Returns:
        User: Usuario o None si no existe.
    """
    if username in db:
        user_dict = db[username]
        return User(**user_dict)
    return None

def authenticate_user(db, username: str, password: str) -> Optional[User]:
    """
    Autentica un usuario.
    
    Args:
        db: Base de datos de usuarios.
        username: Nombre de usuario.
        password: Contraseña.
    
    Returns:
        User: Usuario autenticado o None si la autenticación falla.
    """
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Crea un token de acceso.
    
    Args:
        data: Datos a incluir en el token.
        expires_delta: Tiempo de expiración del token.
    
    Returns:
        str: Token de acceso.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Obtiene el usuario actual a partir del token.
    
    Args:
        token: Token de acceso.
    
    Returns:
        User: Usuario actual.
    
    Raises:
        HTTPException: Si el token es inválido o el usuario no existe.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Obtiene el usuario actual activo.
    
    Args:
        current_user: Usuario actual.
    
    Returns:
        User: Usuario actual activo.
    
    Raises:
        HTTPException: Si el usuario está deshabilitado.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
