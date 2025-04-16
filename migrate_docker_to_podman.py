#!/usr/bin/env python3
"""
Script para migrar imágenes y contenedores de Docker a Podman.
Este script facilita la transición de Docker a Podman migrando imágenes,
volúmenes y configuraciones.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import re
import yaml

# Colores para la salida
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_command(command, check=True, capture_output=True):
    """Ejecuta un comando y devuelve su salida."""
    try:
        if capture_output:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=check,
                shell=True
            )
            return result.stdout.strip(), result.returncode
        else:
            result = subprocess.run(
                command,
                check=check,
                shell=True
            )
            return "", result.returncode
    except subprocess.CalledProcessError as e:
        if capture_output:
            return e.stderr.strip(), e.returncode
        return "", e.returncode

def check_docker_installed():
    """Verifica si Docker está instalado."""
    print(f"{Colors.HEADER}Verificando instalación de Docker...{Colors.ENDC}")
    output, return_code = run_command("docker --version", check=False)
    
    if return_code == 0:
        print(f"{Colors.OKGREEN}✓ Docker está instalado: {output}{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Docker no está instalado o no se encuentra en el PATH{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return False

def check_podman_installed():
    """Verifica si Podman está instalado."""
    print(f"{Colors.HEADER}Verificando instalación de Podman...{Colors.ENDC}")
    output, return_code = run_command("podman --version", check=False)
    
    if return_code == 0:
        print(f"{Colors.OKGREEN}✓ Podman está instalado: {output}{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Podman no está instalado o no se encuentra en el PATH{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return False

def list_docker_images():
    """Lista todas las imágenes de Docker."""
    print(f"{Colors.HEADER}Listando imágenes de Docker...{Colors.ENDC}")
    output, return_code = run_command("docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}'", check=False)
    
    if return_code == 0:
        images = []
        for line in output.splitlines():
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    image_id = parts[1]
                    images.append((image_name, image_id))
        
        if images:
            print(f"{Colors.OKGREEN}✓ Se encontraron {len(images)} imágenes de Docker{Colors.ENDC}")
            for i, (name, image_id) in enumerate(images, 1):
                print(f"{Colors.OKBLUE}{i}. {name} ({image_id}){Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠ No se encontraron imágenes de Docker{Colors.ENDC}")
        
        return images
    else:
        print(f"{Colors.FAIL}✗ Error al listar imágenes de Docker{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return []

def list_docker_containers():
    """Lista todos los contenedores de Docker."""
    print(f"{Colors.HEADER}Listando contenedores de Docker...{Colors.ENDC}")
    output, return_code = run_command("docker ps -a --format '{{.Names}} {{.Image}} {{.ID}}'", check=False)
    
    if return_code == 0:
        containers = []
        for line in output.splitlines():
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    container_name = parts[0]
                    container_image = parts[1]
                    container_id = parts[2]
                    containers.append((container_name, container_image, container_id))
        
        if containers:
            print(f"{Colors.OKGREEN}✓ Se encontraron {len(containers)} contenedores de Docker{Colors.ENDC}")
            for i, (name, image, container_id) in enumerate(containers, 1):
                print(f"{Colors.OKBLUE}{i}. {name} (Imagen: {image}, ID: {container_id}){Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠ No se encontraron contenedores de Docker{Colors.ENDC}")
        
        return containers
    else:
        print(f"{Colors.FAIL}✗ Error al listar contenedores de Docker{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return []

def migrate_image(image_name, image_id, output_dir):
    """Migra una imagen de Docker a Podman."""
    print(f"{Colors.HEADER}Migrando imagen {image_name} ({image_id})...{Colors.ENDC}")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Nombre del archivo tar para la imagen
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', image_name)
    tar_file = os.path.join(output_dir, f"{safe_name}.tar")
    
    # Guardar la imagen de Docker como archivo tar
    print(f"{Colors.OKBLUE}Guardando imagen como archivo tar...{Colors.ENDC}")
    save_output, save_code = run_command(f"docker save -o {tar_file} {image_name}", check=False)
    
    if save_code == 0:
        print(f"{Colors.OKGREEN}✓ Imagen guardada en {tar_file}{Colors.ENDC}")
        
        # Cargar la imagen en Podman
        print(f"{Colors.OKBLUE}Cargando imagen en Podman...{Colors.ENDC}")
        load_output, load_code = run_command(f"podman load -i {tar_file}", check=False)
        
        if load_code == 0:
            print(f"{Colors.OKGREEN}✓ Imagen cargada en Podman: {load_output}{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}✗ Error al cargar la imagen en Podman{Colors.ENDC}")
            print(f"{Colors.FAIL}Error: {load_output}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ Error al guardar la imagen de Docker{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {save_output}{Colors.ENDC}")
        return False

def get_container_config(container_id):
    """Obtiene la configuración de un contenedor Docker."""
    print(f"{Colors.OKBLUE}Obteniendo configuración del contenedor {container_id}...{Colors.ENDC}")
    
    # Obtener la configuración del contenedor en formato JSON
    output, return_code = run_command(f"docker inspect {container_id}", check=False)
    
    if return_code == 0:
        try:
            config = json.loads(output)
            if isinstance(config, list) and len(config) > 0:
                return config[0]
            else:
                print(f"{Colors.FAIL}✗ Formato de configuración inesperado{Colors.ENDC}")
                return None
        except json.JSONDecodeError:
            print(f"{Colors.FAIL}✗ Error al parsear la configuración JSON{Colors.ENDC}")
            return None
    else:
        print(f"{Colors.FAIL}✗ Error al obtener la configuración del contenedor{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return None

def generate_podman_run_command(container_config, container_name):
    """Genera un comando podman run basado en la configuración del contenedor Docker."""
    if not container_config:
        return None
    
    cmd = ["podman", "run"]
    
    # Nombre del contenedor
    cmd.extend(["--name", container_name])
    
    # Modo de ejecución (detached o interactivo)
    if container_config.get("Config", {}).get("Tty", False):
        cmd.append("-it")
    else:
        cmd.append("-d")
    
    # Puertos
    ports = container_config.get("HostConfig", {}).get("PortBindings", {})
    for container_port, host_bindings in ports.items():
        for binding in host_bindings:
            host_ip = binding.get("HostIp", "")
            host_port = binding.get("HostPort", "")
            
            if host_ip and host_port:
                cmd.extend(["-p", f"{host_ip}:{host_port}:{container_port.split('/')[0]}"])
            elif host_port:
                cmd.extend(["-p", f"{host_port}:{container_port.split('/')[0]}"])
    
    # Volúmenes
    mounts = container_config.get("Mounts", [])
    for mount in mounts:
        source = mount.get("Source", "")
        destination = mount.get("Destination", "")
        if source and destination:
            cmd.extend(["-v", f"{source}:{destination}"])
    
    # Variables de entorno
    env_vars = container_config.get("Config", {}).get("Env", [])
    for env in env_vars:
        cmd.extend(["-e", env])
    
    # Redes
    network_mode = container_config.get("HostConfig", {}).get("NetworkMode", "")
    if network_mode and network_mode != "default":
        cmd.extend(["--network", network_mode])
    
    # Imagen
    image = container_config.get("Config", {}).get("Image", "")
    cmd.append(image)
    
    # Comando
    entrypoint = container_config.get("Config", {}).get("Entrypoint", [])
    cmd_args = container_config.get("Config", {}).get("Cmd", [])
    
    if entrypoint:
        if isinstance(entrypoint, list):
            cmd.extend(entrypoint)
        else:
            cmd.append(entrypoint)
    
    if cmd_args:
        if isinstance(cmd_args, list):
            cmd.extend(cmd_args)
        else:
            cmd.append(cmd_args)
    
    return " ".join(cmd)

def migrate_container(container_name, container_image, container_id, output_dir):
    """Migra un contenedor de Docker a Podman."""
    print(f"{Colors.HEADER}Migrando contenedor {container_name} (ID: {container_id})...{Colors.ENDC}")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener la configuración del contenedor
    container_config = get_container_config(container_id)
    
    if container_config:
        # Generar comando podman run
        podman_cmd = generate_podman_run_command(container_config, container_name)
        
        if podman_cmd:
            # Guardar el comando en un archivo
            script_file = os.path.join(output_dir, f"{container_name}_podman_run.sh")
            with open(script_file, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write(f"# Comando para recrear el contenedor {container_name} en Podman\n")
                f.write(f"{podman_cmd}\n")
            
            # Hacer el archivo ejecutable
            os.chmod(script_file, 0o755)
            
            print(f"{Colors.OKGREEN}✓ Comando de Podman generado y guardado en {script_file}{Colors.ENDC}")
            
            # Preguntar si se desea ejecutar el comando
            print(f"{Colors.OKBLUE}Comando generado:{Colors.ENDC}")
            print(f"{Colors.OKBLUE}{podman_cmd}{Colors.ENDC}")
            
            return True
        else:
            print(f"{Colors.FAIL}✗ No se pudo generar el comando podman run{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ No se pudo obtener la configuración del contenedor{Colors.ENDC}")
        return False

def migrate_docker_compose(compose_file, output_dir):
    """Migra un archivo docker-compose.yml a podman-compose.yml."""
    print(f"{Colors.HEADER}Migrando archivo docker-compose {compose_file} a podman-compose...{Colors.ENDC}")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Leer el archivo docker-compose.yml
    try:
        with open(compose_file, "r") as f:
            compose_data = yaml.safe_load(f)
        
        # Verificar si es un archivo docker-compose válido
        if not isinstance(compose_data, dict) or "services" not in compose_data:
            print(f"{Colors.FAIL}✗ El archivo {compose_file} no parece ser un archivo docker-compose válido{Colors.ENDC}")
            return False
        
        # Nombre del archivo de salida
        output_file = os.path.join(output_dir, "podman-compose.yml")
        
        # Guardar el archivo podman-compose.yml
        with open(output_file, "w") as f:
            yaml.dump(compose_data, f, default_flow_style=False)
        
        print(f"{Colors.OKGREEN}✓ Archivo podman-compose generado y guardado en {output_file}{Colors.ENDC}")
        
        # Crear un script para ejecutar podman-compose
        script_file = os.path.join(output_dir, "run_podman_compose.sh")
        with open(script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Instalar podman-compose si no está instalado\n")
            f.write("if ! command -v podman-compose &> /dev/null; then\n")
            f.write("    echo \"Instalando podman-compose...\"\n")
            f.write("    pip install podman-compose\n")
            f.write("fi\n\n")
            f.write("# Ejecutar podman-compose\n")
            f.write(f"cd {output_dir}\n")
            f.write("podman-compose up -d\n")
        
        # Hacer el archivo ejecutable
        os.chmod(script_file, 0o755)
        
        print(f"{Colors.OKGREEN}✓ Script para ejecutar podman-compose generado y guardado en {script_file}{Colors.ENDC}")
        
        return True
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error al migrar el archivo docker-compose: {str(e)}{Colors.ENDC}")
        return False

def create_docker_alias():
    """Crea un alias de Docker a Podman."""
    print(f"{Colors.HEADER}Creando alias de Docker a Podman...{Colors.ENDC}")
    
    # Determinar el shell actual
    shell_rc = ""
    if "ZSH_VERSION" in os.environ:
        shell_rc = os.path.expanduser("~/.zshrc")
    else:
        shell_rc = os.path.expanduser("~/.bashrc")
    
    # Verificar si el alias ya existe
    try:
        with open(shell_rc, "r") as f:
            content = f.read()
            if "alias docker=podman" in content:
                print(f"{Colors.WARNING}⚠ El alias 'docker=podman' ya existe en {shell_rc}{Colors.ENDC}")
                return True
    except FileNotFoundError:
        pass
    
    # Añadir el alias
    try:
        with open(shell_rc, "a") as f:
            f.write("\n# Alias para usar Podman en lugar de Docker\n")
            f.write("alias docker=podman\n")
        
        print(f"{Colors.OKGREEN}✓ Alias 'docker=podman' añadido a {shell_rc}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Para activar el alias, ejecute: source {shell_rc}{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error al crear el alias: {str(e)}{Colors.ENDC}")
        return False

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Migrar de Docker a Podman")
    parser.add_argument("--output-dir", default="./podman_migration", help="Directorio de salida para los archivos migrados")
    parser.add_argument("--migrate-images", action="store_true", help="Migrar imágenes de Docker a Podman")
    parser.add_argument("--migrate-containers", action="store_true", help="Migrar contenedores de Docker a Podman")
    parser.add_argument("--migrate-compose", help="Migrar archivo docker-compose.yml a podman-compose.yml")
    parser.add_argument("--create-alias", action="store_true", help="Crear alias de Docker a Podman")
    parser.add_argument("--all", action="store_true", help="Realizar todas las migraciones")
    
    args = parser.parse_args()
    
    # Verificar instalaciones
    docker_installed = check_docker_installed()
    podman_installed = check_podman_installed()
    
    if not docker_installed:
        print(f"{Colors.FAIL}✗ Docker no está instalado. No se puede realizar la migración.{Colors.ENDC}")
        return 1
    
    if not podman_installed:
        print(f"{Colors.FAIL}✗ Podman no está instalado. No se puede realizar la migración.{Colors.ENDC}")
        return 1
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Migrar imágenes
    if args.migrate_images or args.all:
        images = list_docker_images()
        if images:
            print(f"{Colors.HEADER}Migrando imágenes de Docker a Podman...{Colors.ENDC}")
            for image_name, image_id in images:
                migrate_image(image_name, image_id, args.output_dir)
    
    # Migrar contenedores
    if args.migrate_containers or args.all:
        containers = list_docker_containers()
        if containers:
            print(f"{Colors.HEADER}Migrando contenedores de Docker a Podman...{Colors.ENDC}")
            for container_name, container_image, container_id in containers:
                migrate_container(container_name, container_image, container_id, args.output_dir)
    
    # Migrar docker-compose
    if args.migrate_compose or args.all:
        compose_file = args.migrate_compose if args.migrate_compose else "docker-compose.yml"
        if os.path.exists(compose_file):
            migrate_docker_compose(compose_file, args.output_dir)
        else:
            print(f"{Colors.WARNING}⚠ No se encontró el archivo {compose_file}{Colors.ENDC}")
    
    # Crear alias
    if args.create_alias or args.all:
        create_docker_alias()
    
    print(f"{Colors.OKGREEN}✓ Migración completada. Los archivos se han guardado en {args.output_dir}{Colors.ENDC}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
