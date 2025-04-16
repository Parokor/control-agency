#!/usr/bin/env python3
"""
Módulo para autenticación con GitHub usando tokens.
Este módulo proporciona funciones para autenticar con la API de GitHub
y realizar operaciones básicas como clonar repositorios, crear pull requests, etc.
"""

import os
import sys
import json
import requests
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubAuth:
    """Clase para manejar la autenticación con GitHub."""
    
    def __init__(self, token: str = None, token_file: str = None):
        """
        Inicializa la autenticación con GitHub.
        
        Args:
            token: Token de autenticación de GitHub.
            token_file: Ruta al archivo que contiene el token de GitHub.
        """
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = None
        
        # Si no se proporciona un token, intentar obtenerlo del archivo o de las variables de entorno
        if not self.token:
            if token_file and os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    self.token = f.read().strip()
            else:
                self.token = os.environ.get('GITHUB_TOKEN')
        
        if not self.token:
            logger.warning("No se ha proporcionado un token de GitHub. Algunas funciones pueden no estar disponibles.")
        else:
            self.headers = {
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            }
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con GitHub.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return False
        
        try:
            response = requests.get(f"{self.api_url}/user", headers=self.headers)
            response.raise_for_status()
            user_data = response.json()
            logger.info(f"Conexión exitosa con GitHub. Usuario: {user_data.get('login')}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al conectar con GitHub: {str(e)}")
            return False
    
    def get_repo_info(self, repo_owner: str, repo_name: str) -> Optional[Dict]:
        """
        Obtiene información sobre un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
        
        Returns:
            Dict: Información del repositorio o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener información del repositorio: {str(e)}")
            return None
    
    def get_file_content(self, repo_owner: str, repo_name: str, file_path: str, ref: str = "main") -> Optional[str]:
        """
        Obtiene el contenido de un archivo en un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            file_path: Ruta al archivo en el repositorio.
            ref: Referencia (rama, tag, commit) del repositorio.
        
        Returns:
            str: Contenido del archivo o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={ref}",
                headers=self.headers
            )
            response.raise_for_status()
            content_data = response.json()
            
            if content_data.get('type') != 'file':
                logger.error(f"La ruta {file_path} no corresponde a un archivo.")
                return None
            
            content = base64.b64decode(content_data.get('content')).decode('utf-8')
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener el contenido del archivo: {str(e)}")
            return None
    
    def create_file(self, repo_owner: str, repo_name: str, file_path: str, content: str, commit_message: str, branch: str = "main") -> bool:
        """
        Crea un archivo en un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            file_path: Ruta al archivo en el repositorio.
            content: Contenido del archivo.
            commit_message: Mensaje del commit.
            branch: Rama en la que se creará el archivo.
        
        Returns:
            bool: True si la creación es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return False
        
        try:
            # Codificar el contenido en base64
            content_bytes = content.encode('utf-8')
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            
            # Crear el archivo
            data = {
                'message': commit_message,
                'content': content_base64,
                'branch': branch
            }
            
            response = requests.put(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            logger.info(f"Archivo {file_path} creado correctamente en {repo_owner}/{repo_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear el archivo: {str(e)}")
            return False
    
    def update_file(self, repo_owner: str, repo_name: str, file_path: str, content: str, commit_message: str, branch: str = "main") -> bool:
        """
        Actualiza un archivo en un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            file_path: Ruta al archivo en el repositorio.
            content: Nuevo contenido del archivo.
            commit_message: Mensaje del commit.
            branch: Rama en la que se actualizará el archivo.
        
        Returns:
            bool: True si la actualización es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return False
        
        try:
            # Obtener el SHA del archivo actual
            response = requests.get(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch}",
                headers=self.headers
            )
            response.raise_for_status()
            file_sha = response.json().get('sha')
            
            # Codificar el nuevo contenido en base64
            content_bytes = content.encode('utf-8')
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            
            # Actualizar el archivo
            data = {
                'message': commit_message,
                'content': content_base64,
                'sha': file_sha,
                'branch': branch
            }
            
            response = requests.put(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            logger.info(f"Archivo {file_path} actualizado correctamente en {repo_owner}/{repo_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al actualizar el archivo: {str(e)}")
            return False
    
    def create_pull_request(self, repo_owner: str, repo_name: str, title: str, body: str, head: str, base: str = "main") -> Optional[Dict]:
        """
        Crea un pull request en un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            title: Título del pull request.
            body: Descripción del pull request.
            head: Rama de origen.
            base: Rama de destino.
        
        Returns:
            Dict: Información del pull request creado o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return None
        
        try:
            data = {
                'title': title,
                'body': body,
                'head': head,
                'base': base
            }
            
            response = requests.post(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/pulls",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            pr_data = response.json()
            logger.info(f"Pull request creado correctamente: {pr_data.get('html_url')}")
            return pr_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear el pull request: {str(e)}")
            return None
    
    def list_branches(self, repo_owner: str, repo_name: str) -> Optional[List[Dict]]:
        """
        Lista las ramas de un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
        
        Returns:
            List[Dict]: Lista de ramas o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/branches",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al listar las ramas: {str(e)}")
            return None
    
    def create_branch(self, repo_owner: str, repo_name: str, branch_name: str, base_sha: str) -> bool:
        """
        Crea una rama en un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            branch_name: Nombre de la nueva rama.
            base_sha: SHA del commit base.
        
        Returns:
            bool: True si la creación es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return False
        
        try:
            data = {
                'ref': f"refs/heads/{branch_name}",
                'sha': base_sha
            }
            
            response = requests.post(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/git/refs",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            logger.info(f"Rama {branch_name} creada correctamente en {repo_owner}/{repo_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear la rama: {str(e)}")
            return False
    
    def get_default_branch(self, repo_owner: str, repo_name: str) -> Optional[str]:
        """
        Obtiene la rama por defecto de un repositorio.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
        
        Returns:
            str: Nombre de la rama por defecto o None si hay un error.
        """
        repo_info = self.get_repo_info(repo_owner, repo_name)
        if repo_info:
            return repo_info.get('default_branch')
        return None
    
    def get_latest_commit_sha(self, repo_owner: str, repo_name: str, branch: str = "main") -> Optional[str]:
        """
        Obtiene el SHA del último commit en una rama.
        
        Args:
            repo_owner: Propietario del repositorio.
            repo_name: Nombre del repositorio.
            branch: Nombre de la rama.
        
        Returns:
            str: SHA del último commit o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de GitHub.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/repos/{repo_owner}/{repo_name}/commits/{branch}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json().get('sha')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener el SHA del último commit: {str(e)}")
            return None

def save_token_to_file(token: str, token_file: str) -> bool:
    """
    Guarda un token de GitHub en un archivo.
    
    Args:
        token: Token de GitHub.
        token_file: Ruta al archivo donde se guardará el token.
    
    Returns:
        bool: True si el token se guardó correctamente, False en caso contrario.
    """
    try:
        # Crear el directorio si no existe
        token_dir = os.path.dirname(token_file)
        if token_dir and not os.path.exists(token_dir):
            os.makedirs(token_dir)
        
        # Guardar el token en el archivo
        with open(token_file, 'w') as f:
            f.write(token)
        
        # Establecer permisos restrictivos
        os.chmod(token_file, 0o600)
        
        logger.info(f"Token guardado correctamente en {token_file}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar el token: {str(e)}")
        return False

def load_token_from_file(token_file: str) -> Optional[str]:
    """
    Carga un token de GitHub desde un archivo.
    
    Args:
        token_file: Ruta al archivo que contiene el token.
    
    Returns:
        str: Token de GitHub o None si hay un error.
    """
    try:
        if not os.path.exists(token_file):
            logger.error(f"El archivo {token_file} no existe.")
            return None
        
        with open(token_file, 'r') as f:
            token = f.read().strip()
        
        if not token:
            logger.error(f"El archivo {token_file} está vacío.")
            return None
        
        return token
    except Exception as e:
        logger.error(f"Error al cargar el token: {str(e)}")
        return None

def main():
    """Función principal para probar la autenticación con GitHub."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autenticación con GitHub")
    parser.add_argument("--token", help="Token de autenticación de GitHub")
    parser.add_argument("--token-file", help="Ruta al archivo que contiene el token de GitHub")
    parser.add_argument("--save-token", action="store_true", help="Guardar el token en un archivo")
    parser.add_argument("--test", action="store_true", help="Probar la conexión con GitHub")
    parser.add_argument("--repo-owner", help="Propietario del repositorio")
    parser.add_argument("--repo-name", help="Nombre del repositorio")
    
    args = parser.parse_args()
    
    # Si se proporciona un token y se quiere guardar
    if args.token and args.save_token and args.token_file:
        if save_token_to_file(args.token, args.token_file):
            print(f"Token guardado correctamente en {args.token_file}")
        else:
            print(f"Error al guardar el token en {args.token_file}")
        return
    
    # Crear instancia de GitHubAuth
    github_auth = GitHubAuth(token=args.token, token_file=args.token_file)
    
    # Probar la conexión
    if args.test:
        if github_auth.test_connection():
            print("Conexión exitosa con GitHub")
        else:
            print("Error al conectar con GitHub")
    
    # Obtener información de un repositorio
    if args.repo_owner and args.repo_name:
        repo_info = github_auth.get_repo_info(args.repo_owner, args.repo_name)
        if repo_info:
            print(f"Repositorio: {repo_info.get('full_name')}")
            print(f"Descripción: {repo_info.get('description')}")
            print(f"Estrellas: {repo_info.get('stargazers_count')}")
            print(f"Forks: {repo_info.get('forks_count')}")
            print(f"Rama por defecto: {repo_info.get('default_branch')}")
        else:
            print(f"Error al obtener información del repositorio {args.repo_owner}/{args.repo_name}")

if __name__ == "__main__":
    main()
