#!/usr/bin/env python3
"""
Módulo para autenticación con plataformas en la nube usando tokens.
Este módulo proporciona clases para autenticar con diferentes plataformas en la nube
como Google Colab, Paperspace Gradient, RunPod, etc.
"""

import os
import sys
import json
import requests
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CloudPlatformAuth(ABC):
    """Clase base abstracta para autenticación con plataformas en la nube."""
    
    def __init__(self, token: str = None, token_file: str = None):
        """
        Inicializa la autenticación con la plataforma en la nube.
        
        Args:
            token: Token de autenticación.
            token_file: Ruta al archivo que contiene el token.
        """
        self.token = token
        self.headers = None
        
        # Si no se proporciona un token, intentar obtenerlo del archivo o de las variables de entorno
        if not self.token:
            if token_file and os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    self.token = f.read().strip()
            else:
                self.token = self._get_token_from_env()
        
        if not self.token:
            logger.warning(f"No se ha proporcionado un token para {self.__class__.__name__}. Algunas funciones pueden no estar disponibles.")
        else:
            self.headers = self._get_headers()
    
    @abstractmethod
    def _get_token_from_env(self) -> Optional[str]:
        """
        Obtiene el token de las variables de entorno.
        
        Returns:
            str: Token de autenticación o None si no está disponible.
        """
        pass
    
    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """
        Obtiene los headers para las peticiones a la API.
        
        Returns:
            Dict[str, str]: Headers para las peticiones.
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Prueba la conexión con la plataforma en la nube.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        pass
    
    def save_token_to_file(self, token_file: str) -> bool:
        """
        Guarda el token en un archivo.
        
        Args:
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
                f.write(self.token)
            
            # Establecer permisos restrictivos
            os.chmod(token_file, 0o600)
            
            logger.info(f"Token guardado correctamente en {token_file}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar el token: {str(e)}")
            return False

class ColabAuth(CloudPlatformAuth):
    """Clase para autenticación con Google Colab."""
    
    def __init__(self, token: str = None, token_file: str = None):
        """
        Inicializa la autenticación con Google Colab.
        
        Args:
            token: Token de autenticación de Google Colab.
            token_file: Ruta al archivo que contiene el token de Google Colab.
        """
        super().__init__(token, token_file)
        self.api_url = "https://colab.research.google.com/api"
    
    def _get_token_from_env(self) -> Optional[str]:
        """
        Obtiene el token de Google Colab de las variables de entorno.
        
        Returns:
            str: Token de Google Colab o None si no está disponible.
        """
        return os.environ.get('COLAB_TOKEN')
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Obtiene los headers para las peticiones a la API de Google Colab.
        
        Returns:
            Dict[str, str]: Headers para las peticiones.
        """
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con Google Colab.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Google Colab.")
            return False
        
        try:
            # Google Colab no tiene un endpoint específico para probar la conexión
            # Intentamos obtener la lista de notebooks como prueba
            response = requests.get(
                f"{self.api_url}/notebooks",
                headers=self.headers
            )
            response.raise_for_status()
            logger.info("Conexión exitosa con Google Colab.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al conectar con Google Colab: {str(e)}")
            return False
    
    def create_notebook(self, notebook_content: Dict) -> Optional[str]:
        """
        Crea un nuevo notebook en Google Colab.
        
        Args:
            notebook_content: Contenido del notebook en formato JSON.
        
        Returns:
            str: ID del notebook creado o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Google Colab.")
            return None
        
        try:
            response = requests.post(
                f"{self.api_url}/notebooks",
                headers=self.headers,
                json=notebook_content
            )
            response.raise_for_status()
            notebook_id = response.json().get('id')
            logger.info(f"Notebook creado correctamente con ID: {notebook_id}")
            return notebook_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear el notebook: {str(e)}")
            return None
    
    def get_notebook(self, notebook_id: str) -> Optional[Dict]:
        """
        Obtiene un notebook de Google Colab.
        
        Args:
            notebook_id: ID del notebook.
        
        Returns:
            Dict: Contenido del notebook o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Google Colab.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/notebooks/{notebook_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener el notebook: {str(e)}")
            return None
    
    def execute_notebook(self, notebook_id: str) -> bool:
        """
        Ejecuta un notebook en Google Colab.
        
        Args:
            notebook_id: ID del notebook.
        
        Returns:
            bool: True si la ejecución es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Google Colab.")
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/notebooks/{notebook_id}/execute",
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Notebook {notebook_id} ejecutado correctamente.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al ejecutar el notebook: {str(e)}")
            return False

class PaperspaceAuth(CloudPlatformAuth):
    """Clase para autenticación con Paperspace Gradient."""
    
    def __init__(self, token: str = None, token_file: str = None):
        """
        Inicializa la autenticación con Paperspace Gradient.
        
        Args:
            token: Token de autenticación de Paperspace Gradient.
            token_file: Ruta al archivo que contiene el token de Paperspace Gradient.
        """
        super().__init__(token, token_file)
        self.api_url = "https://api.paperspace.io"
    
    def _get_token_from_env(self) -> Optional[str]:
        """
        Obtiene el token de Paperspace Gradient de las variables de entorno.
        
        Returns:
            str: Token de Paperspace Gradient o None si no está disponible.
        """
        return os.environ.get('PAPERSPACE_API_KEY')
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Obtiene los headers para las peticiones a la API de Paperspace Gradient.
        
        Returns:
            Dict[str, str]: Headers para las peticiones.
        """
        return {
            'X-API-Key': self.token,
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con Paperspace Gradient.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Paperspace Gradient.")
            return False
        
        try:
            response = requests.get(
                f"{self.api_url}/users/me",
                headers=self.headers
            )
            response.raise_for_status()
            user_data = response.json()
            logger.info(f"Conexión exitosa con Paperspace Gradient. Usuario: {user_data.get('email')}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al conectar con Paperspace Gradient: {str(e)}")
            return False
    
    def list_notebooks(self) -> Optional[List[Dict]]:
        """
        Lista los notebooks disponibles en Paperspace Gradient.
        
        Returns:
            List[Dict]: Lista de notebooks o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Paperspace Gradient.")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/notebooks",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al listar los notebooks: {str(e)}")
            return None
    
    def create_notebook(self, name: str, machine_type: str, container: str) -> Optional[Dict]:
        """
        Crea un nuevo notebook en Paperspace Gradient.
        
        Args:
            name: Nombre del notebook.
            machine_type: Tipo de máquina (p.ej., "K80", "P4000").
            container: Contenedor a utilizar.
        
        Returns:
            Dict: Información del notebook creado o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de Paperspace Gradient.")
            return None
        
        try:
            data = {
                'name': name,
                'machineType': machine_type,
                'container': container
            }
            
            response = requests.post(
                f"{self.api_url}/notebooks",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            notebook_data = response.json()
            logger.info(f"Notebook creado correctamente: {notebook_data.get('id')}")
            return notebook_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear el notebook: {str(e)}")
            return None

class RunPodAuth(CloudPlatformAuth):
    """Clase para autenticación con RunPod."""
    
    def __init__(self, token: str = None, token_file: str = None):
        """
        Inicializa la autenticación con RunPod.
        
        Args:
            token: Token de autenticación de RunPod.
            token_file: Ruta al archivo que contiene el token de RunPod.
        """
        super().__init__(token, token_file)
        self.api_url = "https://api.runpod.io/graphql"
    
    def _get_token_from_env(self) -> Optional[str]:
        """
        Obtiene el token de RunPod de las variables de entorno.
        
        Returns:
            str: Token de RunPod o None si no está disponible.
        """
        return os.environ.get('RUNPOD_API_KEY')
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Obtiene los headers para las peticiones a la API de RunPod.
        
        Returns:
            Dict[str, str]: Headers para las peticiones.
        """
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con RunPod.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de RunPod.")
            return False
        
        try:
            # Consulta GraphQL para obtener información del usuario
            query = """
            query {
                myself {
                    id
                    email
                }
            }
            """
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': query}
            )
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"Error en la consulta GraphQL: {data['errors']}")
                return False
            
            user_data = data.get('data', {}).get('myself', {})
            logger.info(f"Conexión exitosa con RunPod. Usuario: {user_data.get('email')}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al conectar con RunPod: {str(e)}")
            return False
    
    def list_pods(self) -> Optional[List[Dict]]:
        """
        Lista los pods disponibles en RunPod.
        
        Returns:
            List[Dict]: Lista de pods o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de RunPod.")
            return None
        
        try:
            # Consulta GraphQL para obtener la lista de pods
            query = """
            query {
                myself {
                    pods {
                        id
                        name
                        status
                        runtime {
                            uptimeInSeconds
                        }
                    }
                }
            }
            """
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': query}
            )
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"Error en la consulta GraphQL: {data['errors']}")
                return None
            
            pods = data.get('data', {}).get('myself', {}).get('pods', [])
            return pods
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al listar los pods: {str(e)}")
            return None
    
    def create_pod(self, name: str, gpu_type_id: str, container_disk_in_gb: int, docker_image: str) -> Optional[Dict]:
        """
        Crea un nuevo pod en RunPod.
        
        Args:
            name: Nombre del pod.
            gpu_type_id: ID del tipo de GPU.
            container_disk_in_gb: Tamaño del disco del contenedor en GB.
            docker_image: Imagen de Docker a utilizar.
        
        Returns:
            Dict: Información del pod creado o None si hay un error.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de RunPod.")
            return None
        
        try:
            # Consulta GraphQL para crear un pod
            query = """
            mutation ($input: PodDeployInput!) {
                podDeploy(input: $input) {
                    id
                    name
                    status
                }
            }
            """
            
            variables = {
                'input': {
                    'name': name,
                    'gpuTypeId': gpu_type_id,
                    'containerDiskInGb': container_disk_in_gb,
                    'dockerImage': docker_image
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': query, 'variables': variables}
            )
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"Error en la consulta GraphQL: {data['errors']}")
                return None
            
            pod_data = data.get('data', {}).get('podDeploy', {})
            logger.info(f"Pod creado correctamente: {pod_data.get('id')}")
            return pod_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al crear el pod: {str(e)}")
            return None
    
    def stop_pod(self, pod_id: str) -> bool:
        """
        Detiene un pod en RunPod.
        
        Args:
            pod_id: ID del pod.
        
        Returns:
            bool: True si la detención es exitosa, False en caso contrario.
        """
        if not self.token:
            logger.error("No se ha proporcionado un token de RunPod.")
            return False
        
        try:
            # Consulta GraphQL para detener un pod
            query = """
            mutation ($input: PodStopInput!) {
                podStop(input: $input) {
                    id
                    status
                }
            }
            """
            
            variables = {
                'input': {
                    'podId': pod_id
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': query, 'variables': variables}
            )
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"Error en la consulta GraphQL: {data['errors']}")
                return False
            
            logger.info(f"Pod {pod_id} detenido correctamente.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al detener el pod: {str(e)}")
            return False

def get_auth_instance(platform: str, token: str = None, token_file: str = None) -> Optional[CloudPlatformAuth]:
    """
    Obtiene una instancia de autenticación para la plataforma especificada.
    
    Args:
        platform: Nombre de la plataforma ("colab", "paperspace", "runpod").
        token: Token de autenticación.
        token_file: Ruta al archivo que contiene el token.
    
    Returns:
        CloudPlatformAuth: Instancia de autenticación o None si la plataforma no es soportada.
    """
    platform = platform.lower()
    
    if platform == "colab":
        return ColabAuth(token, token_file)
    elif platform == "paperspace":
        return PaperspaceAuth(token, token_file)
    elif platform == "runpod":
        return RunPodAuth(token, token_file)
    else:
        logger.error(f"Plataforma no soportada: {platform}")
        return None

def main():
    """Función principal para probar la autenticación con plataformas en la nube."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autenticación con plataformas en la nube")
    parser.add_argument("--platform", required=True, choices=["colab", "paperspace", "runpod"], help="Plataforma en la nube")
    parser.add_argument("--token", help="Token de autenticación")
    parser.add_argument("--token-file", help="Ruta al archivo que contiene el token")
    parser.add_argument("--save-token", action="store_true", help="Guardar el token en un archivo")
    parser.add_argument("--test", action="store_true", help="Probar la conexión con la plataforma")
    
    args = parser.parse_args()
    
    # Obtener instancia de autenticación
    auth = get_auth_instance(args.platform, args.token, args.token_file)
    
    if not auth:
        print(f"Error al crear la instancia de autenticación para {args.platform}")
        return
    
    # Si se quiere guardar el token
    if args.save_token and args.token_file:
        if auth.save_token_to_file(args.token_file):
            print(f"Token guardado correctamente en {args.token_file}")
        else:
            print(f"Error al guardar el token en {args.token_file}")
    
    # Probar la conexión
    if args.test:
        if auth.test_connection():
            print(f"Conexión exitosa con {args.platform}")
        else:
            print(f"Error al conectar con {args.platform}")

if __name__ == "__main__":
    main()
