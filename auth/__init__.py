"""
Módulo de autenticación para Control Agency.
Este módulo proporciona clases y funciones para autenticar con diferentes plataformas.
"""

from auth.github_auth import GitHubAuth, save_token_to_file, load_token_from_file
from auth.cloud_auth import get_auth_instance, CloudPlatformAuth, ColabAuth, PaperspaceAuth, RunPodAuth
from auth.token_manager import TokenManager

__all__ = [
    'GitHubAuth',
    'save_token_to_file',
    'load_token_from_file',
    'get_auth_instance',
    'CloudPlatformAuth',
    'ColabAuth',
    'PaperspaceAuth',
    'RunPodAuth',
    'TokenManager'
]
