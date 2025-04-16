#!/usr/bin/env python3
"""
Módulo para la gestión de tokens de autenticación.
Este módulo proporciona funciones para gestionar tokens de autenticación
de diferentes plataformas de forma segura.
"""

import os
import sys
import json
import base64
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenManager:
    """Clase para la gestión de tokens de autenticación."""
    
    def __init__(self, config_dir: str = None, master_password: str = None):
        """
        Inicializa el gestor de tokens.
        
        Args:
            config_dir: Directorio de configuración donde se guardarán los tokens.
            master_password: Contraseña maestra para cifrar los tokens.
        """
        # Directorio de configuración por defecto
        if not config_dir:
            config_dir = os.path.join(str(Path.home()), ".control-agency")
        
        self.config_dir = config_dir
        self.tokens_file = os.path.join(config_dir, "tokens.json")
        self.salt_file = os.path.join(config_dir, "salt")
        self.master_password = master_password
        self.fernet = None
        
        # Crear directorio de configuración si no existe
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            logger.info(f"Directorio de configuración creado: {config_dir}")
        
        # Inicializar el cifrado
        self._init_encryption()
    
    def _init_encryption(self):
        """Inicializa el cifrado para los tokens."""
        # Si no hay contraseña maestra, pedirla al usuario
        if not self.master_password:
            if os.path.exists(self.tokens_file):
                self.master_password = getpass.getpass("Introduce la contraseña maestra para descifrar los tokens: ")
            else:
                self.master_password = getpass.getpass("Introduce una contraseña maestra para cifrar los tokens: ")
                confirm_password = getpass.getpass("Confirma la contraseña maestra: ")
                if self.master_password != confirm_password:
                    logger.error("Las contraseñas no coinciden.")
                    sys.exit(1)
        
        # Generar o cargar la sal
        if os.path.exists(self.salt_file):
            with open(self.salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(self.salt_file, 'wb') as f:
                f.write(salt)
            # Establecer permisos restrictivos
            os.chmod(self.salt_file, 0o600)
        
        # Derivar la clave a partir de la contraseña y la sal
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        
        # Crear el objeto Fernet para cifrar/descifrar
        self.fernet = Fernet(key)
    
    def save_token(self, platform: str, token: str) -> bool:
        """
        Guarda un token para una plataforma.
        
        Args:
            platform: Nombre de la plataforma.
            token: Token de autenticación.
        
        Returns:
            bool: True si el token se guardó correctamente, False en caso contrario.
        """
        try:
            # Cargar tokens existentes
            tokens = self.load_tokens()
            
            # Cifrar el token
            encrypted_token = self.fernet.encrypt(token.encode()).decode()
            
            # Guardar el token
            tokens[platform] = encrypted_token
            
            # Guardar los tokens en el archivo
            with open(self.tokens_file, 'w') as f:
                json.dump(tokens, f)
            
            # Establecer permisos restrictivos
            os.chmod(self.tokens_file, 0o600)
            
            logger.info(f"Token para {platform} guardado correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al guardar el token para {platform}: {str(e)}")
            return False
    
    def load_token(self, platform: str) -> Optional[str]:
        """
        Carga un token para una plataforma.
        
        Args:
            platform: Nombre de la plataforma.
        
        Returns:
            str: Token de autenticación o None si no está disponible.
        """
        try:
            # Cargar tokens existentes
            tokens = self.load_tokens()
            
            # Verificar si existe un token para la plataforma
            if platform not in tokens:
                logger.warning(f"No hay token guardado para {platform}.")
                return None
            
            # Descifrar el token
            encrypted_token = tokens[platform]
            token = self.fernet.decrypt(encrypted_token.encode()).decode()
            
            return token
        except Exception as e:
            logger.error(f"Error al cargar el token para {platform}: {str(e)}")
            return None
    
    def load_tokens(self) -> Dict[str, str]:
        """
        Carga todos los tokens.
        
        Returns:
            Dict[str, str]: Diccionario con los tokens cifrados.
        """
        if not os.path.exists(self.tokens_file):
            return {}
        
        try:
            with open(self.tokens_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar los tokens: {str(e)}")
            return {}
    
    def delete_token(self, platform: str) -> bool:
        """
        Elimina un token para una plataforma.
        
        Args:
            platform: Nombre de la plataforma.
        
        Returns:
            bool: True si el token se eliminó correctamente, False en caso contrario.
        """
        try:
            # Cargar tokens existentes
            tokens = self.load_tokens()
            
            # Verificar si existe un token para la plataforma
            if platform not in tokens:
                logger.warning(f"No hay token guardado para {platform}.")
                return False
            
            # Eliminar el token
            del tokens[platform]
            
            # Guardar los tokens en el archivo
            with open(self.tokens_file, 'w') as f:
                json.dump(tokens, f)
            
            logger.info(f"Token para {platform} eliminado correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar el token para {platform}: {str(e)}")
            return False
    
    def list_platforms(self) -> List[str]:
        """
        Lista las plataformas con tokens guardados.
        
        Returns:
            List[str]: Lista de plataformas.
        """
        try:
            # Cargar tokens existentes
            tokens = self.load_tokens()
            
            return list(tokens.keys())
        except Exception as e:
            logger.error(f"Error al listar las plataformas: {str(e)}")
            return []
    
    def verify_token(self, platform: str) -> bool:
        """
        Verifica si un token para una plataforma es válido.
        
        Args:
            platform: Nombre de la plataforma.
        
        Returns:
            bool: True si el token es válido, False en caso contrario.
        """
        # Cargar el token
        token = self.load_token(platform)
        
        if not token:
            return False
        
        # Verificar el token según la plataforma
        if platform.lower() == "github":
            from auth.github_auth import GitHubAuth
            github_auth = GitHubAuth(token=token)
            return github_auth.test_connection()
        elif platform.lower() in ["colab", "paperspace", "runpod"]:
            from auth.cloud_auth import get_auth_instance
            cloud_auth = get_auth_instance(platform, token=token)
            return cloud_auth.test_connection() if cloud_auth else False
        else:
            logger.warning(f"No se puede verificar el token para la plataforma {platform}.")
            return False

def main():
    """Función principal para gestionar tokens de autenticación."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestión de tokens de autenticación")
    parser.add_argument("--config-dir", help="Directorio de configuración")
    parser.add_argument("--save", action="store_true", help="Guardar un token")
    parser.add_argument("--load", action="store_true", help="Cargar un token")
    parser.add_argument("--delete", action="store_true", help="Eliminar un token")
    parser.add_argument("--list", action="store_true", help="Listar plataformas con tokens guardados")
    parser.add_argument("--verify", action="store_true", help="Verificar un token")
    parser.add_argument("--platform", help="Nombre de la plataforma")
    parser.add_argument("--token", help="Token de autenticación")
    
    args = parser.parse_args()
    
    # Crear instancia de TokenManager
    token_manager = TokenManager(config_dir=args.config_dir)
    
    # Guardar un token
    if args.save and args.platform:
        token = args.token
        if not token:
            token = getpass.getpass(f"Introduce el token para {args.platform}: ")
        
        if token_manager.save_token(args.platform, token):
            print(f"Token para {args.platform} guardado correctamente.")
        else:
            print(f"Error al guardar el token para {args.platform}.")
    
    # Cargar un token
    elif args.load and args.platform:
        token = token_manager.load_token(args.platform)
        if token:
            print(f"Token para {args.platform}: {token}")
        else:
            print(f"No se pudo cargar el token para {args.platform}.")
    
    # Eliminar un token
    elif args.delete and args.platform:
        if token_manager.delete_token(args.platform):
            print(f"Token para {args.platform} eliminado correctamente.")
        else:
            print(f"Error al eliminar el token para {args.platform}.")
    
    # Listar plataformas
    elif args.list:
        platforms = token_manager.list_platforms()
        if platforms:
            print("Plataformas con tokens guardados:")
            for platform in platforms:
                print(f"- {platform}")
        else:
            print("No hay tokens guardados.")
    
    # Verificar un token
    elif args.verify and args.platform:
        if token_manager.verify_token(args.platform):
            print(f"El token para {args.platform} es válido.")
        else:
            print(f"El token para {args.platform} no es válido o no existe.")
    
    # Mostrar ayuda si no se proporciona ninguna opción
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
