"""
Configuración para la interfaz web de Control Agency.
"""

import os
from pydantic import BaseSettings
from typing import Optional, Dict, Any, List

class Settings(BaseSettings):
    """
    Configuración para la aplicación web.
    """
    # Configuración del servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Configuración de la base de datos
    DATABASE_URL: Optional[str] = None
    
    # Configuración de autenticación
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "supersecretkey")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Configuración de plataformas en la nube
    CLOUD_PLATFORMS: List[str] = ["colab", "paperspace", "runpod"]
    
    # Configuración de GitHub
    GITHUB_REPO_OWNER: Optional[str] = os.environ.get("GITHUB_REPO_OWNER")
    GITHUB_REPO_NAME: Optional[str] = os.environ.get("GITHUB_REPO_NAME")
    
    # Configuración de directorios
    CONFIG_DIR: str = os.path.join(os.path.expanduser("~"), ".control-agency")
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Crear instancia de configuración
settings = Settings()
