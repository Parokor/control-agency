#!/usr/bin/env python3
"""
Script principal para coordinar la migración de Docker a Podman.
Este script orquesta todo el proceso de migración, desde la instalación de Podman
hasta la migración de imágenes, contenedores y archivos de configuración.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

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

def check_prerequisites():
    """Verifica los requisitos previos para la migración."""
    print(f"{Colors.HEADER}Verificando requisitos previos...{Colors.ENDC}")
    
    # Verificar Python 3.6+
    python_version, _ = run_command("python3 --version", check=False)
    if not python_version:
        print(f"{Colors.FAIL}✗ Python 3 no está instalado{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✓ {python_version} está instalado{Colors.ENDC}")
    
    # Verificar pip
    pip_version, _ = run_command("pip3 --version", check=False)
    if not pip_version:
        print(f"{Colors.FAIL}✗ pip3 no está instalado{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✓ pip está instalado: {pip_version}{Colors.ENDC}")
    
    # Verificar Docker (opcional)
    docker_version, docker_code = run_command("docker --version", check=False)
    if docker_code == 0:
        print(f"{Colors.OKGREEN}✓ Docker está instalado: {docker_version}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠ Docker no está instalado. La migración de imágenes y contenedores no será posible.{Colors.ENDC}")
    
    # Verificar Podman (opcional)
    podman_version, podman_code = run_command("podman --version", check=False)
    if podman_code == 0:
        print(f"{Colors.OKGREEN}✓ Podman ya está instalado: {podman_version}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠ Podman no está instalado. Se instalará durante el proceso de migración.{Colors.ENDC}")
    
    # Verificar dependencias de Python
    print(f"{Colors.OKBLUE}Instalando dependencias de Python...{Colors.ENDC}")
    _, pip_code = run_command("pip3 install pyyaml", check=False)
    if pip_code != 0:
        print(f"{Colors.FAIL}✗ No se pudieron instalar las dependencias de Python{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✓ Dependencias de Python instaladas correctamente{Colors.ENDC}")
    
    return True

def run_tests():
    """Ejecuta las pruebas unitarias."""
    print(f"{Colors.HEADER}Ejecutando pruebas unitarias...{Colors.ENDC}")
    
    # Hacer ejecutables los scripts
    run_command("chmod +x install_podman.sh", check=False)
    
    # Ejecutar las pruebas
    output, code = run_command("python3 run_migration_tests.py", check=False)
    
    if code == 0:
        print(f"{Colors.OKGREEN}✓ Todas las pruebas pasaron correctamente{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Algunas pruebas fallaron{Colors.ENDC}")
        print(output)
        return False

def install_podman():
    """Instala Podman."""
    print(f"{Colors.HEADER}Instalando Podman...{Colors.ENDC}")
    
    # Verificar si Podman ya está instalado
    podman_version, podman_code = run_command("podman --version", check=False)
    if podman_code == 0:
        print(f"{Colors.OKGREEN}✓ Podman ya está instalado: {podman_version}{Colors.ENDC}")
        return True
    
    # Ejecutar el script de instalación
    print(f"{Colors.OKBLUE}Ejecutando script de instalación de Podman...{Colors.ENDC}")
    _, code = run_command("./install_podman.sh", check=False, capture_output=False)
    
    if code == 0:
        print(f"{Colors.OKGREEN}✓ Podman instalado correctamente{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Error al instalar Podman{Colors.ENDC}")
        return False

def test_podman_installation():
    """Prueba la instalación de Podman."""
    print(f"{Colors.HEADER}Probando la instalación de Podman...{Colors.ENDC}")
    
    # Ejecutar el script de prueba
    _, code = run_command("python3 test_podman_installation.py", check=False, capture_output=False)
    
    if code == 0:
        print(f"{Colors.OKGREEN}✓ Instalación de Podman verificada correctamente{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Error al verificar la instalación de Podman{Colors.ENDC}")
        return False

def migrate_docker_to_podman(output_dir, migrate_images=True, migrate_containers=True, migrate_compose=None, create_alias=True):
    """Migra de Docker a Podman."""
    print(f"{Colors.HEADER}Migrando de Docker a Podman...{Colors.ENDC}")
    
    # Construir el comando
    command = f"python3 migrate_docker_to_podman.py --output-dir {output_dir}"
    
    if migrate_images:
        command += " --migrate-images"
    
    if migrate_containers:
        command += " --migrate-containers"
    
    if migrate_compose:
        command += f" --migrate-compose {migrate_compose}"
    
    if create_alias:
        command += " --create-alias"
    
    # Ejecutar el comando
    _, code = run_command(command, check=False, capture_output=False)
    
    if code == 0:
        print(f"{Colors.OKGREEN}✓ Migración completada correctamente{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Error durante la migración{Colors.ENDC}")
        return False

def convert_dockerfiles(dockerfiles, output_dir, optimize=True):
    """Convierte Dockerfiles a Containerfiles."""
    print(f"{Colors.HEADER}Convirtiendo Dockerfiles a Containerfiles...{Colors.ENDC}")
    
    success = True
    
    for dockerfile in dockerfiles:
        if not os.path.exists(dockerfile):
            print(f"{Colors.WARNING}⚠ El archivo {dockerfile} no existe{Colors.ENDC}")
            continue
        
        # Determinar el nombre de salida
        basename = os.path.basename(dockerfile)
        dirname = os.path.dirname(dockerfile)
        output_path = os.path.join(output_dir, dirname, "Containerfile")
        
        # Crear el directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construir el comando
        command = f"python3 convert_dockerfile_to_containerfile.py {dockerfile} --output {output_path}"
        
        if optimize:
            command += " --optimize"
        
        # Ejecutar el comando
        output, code = run_command(command, check=False)
        
        if code == 0:
            print(f"{Colors.OKGREEN}✓ Dockerfile {dockerfile} convertido correctamente a {output_path}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Error al convertir Dockerfile {dockerfile}{Colors.ENDC}")
            print(output)
            success = False
    
    return success

def show_migration_summary(output_dir):
    """Muestra un resumen de la migración."""
    print(f"{Colors.BOLD}{Colors.HEADER}=== Resumen de la Migración ==={Colors.ENDC}")
    
    # Verificar si el directorio de salida existe
    if not os.path.exists(output_dir):
        print(f"{Colors.WARNING}⚠ El directorio de salida {output_dir} no existe{Colors.ENDC}")
        return
    
    # Contar archivos migrados
    containerfiles = list(Path(output_dir).glob("**/Containerfile"))
    podman_compose_files = list(Path(output_dir).glob("**/podman-compose.yml"))
    podman_run_scripts = list(Path(output_dir).glob("**/*_podman_run.sh"))
    
    print(f"{Colors.OKBLUE}Directorio de salida: {output_dir}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Containerfiles: {len(containerfiles)}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Archivos podman-compose: {len(podman_compose_files)}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Scripts de podman run: {len(podman_run_scripts)}{Colors.ENDC}")
    
    # Verificar si Podman está instalado
    podman_version, podman_code = run_command("podman --version", check=False)
    if podman_code == 0:
        print(f"{Colors.OKGREEN}✓ Podman está instalado: {podman_version}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠ Podman no está instalado{Colors.ENDC}")
    
    # Verificar si el alias de Docker a Podman está configurado
    alias_exists = False
    for rc_file in [os.path.expanduser("~/.bashrc"), os.path.expanduser("~/.zshrc")]:
        if os.path.exists(rc_file):
            with open(rc_file, "r") as f:
                if "alias docker=podman" in f.read():
                    alias_exists = True
                    break
    
    if alias_exists:
        print(f"{Colors.OKGREEN}✓ Alias 'docker=podman' configurado{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠ Alias 'docker=podman' no configurado{Colors.ENDC}")
    
    print(f"{Colors.BOLD}{Colors.OKGREEN}Migración completada. Consulte la guía docker_to_podman_migration_guide.md para más información.{Colors.ENDC}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Migrar de Docker a Podman")
    parser.add_argument("--output-dir", default="./podman_migration", help="Directorio de salida para los archivos migrados")
    parser.add_argument("--skip-tests", action="store_true", help="Omitir la ejecución de pruebas unitarias")
    parser.add_argument("--skip-install", action="store_true", help="Omitir la instalación de Podman")
    parser.add_argument("--skip-migration", action="store_true", help="Omitir la migración de Docker a Podman")
    parser.add_argument("--dockerfiles", nargs="+", help="Lista de Dockerfiles a convertir")
    parser.add_argument("--docker-compose", help="Archivo docker-compose.yml a migrar")
    parser.add_argument("--no-images", action="store_true", help="No migrar imágenes de Docker")
    parser.add_argument("--no-containers", action="store_true", help="No migrar contenedores de Docker")
    parser.add_argument("--no-alias", action="store_true", help="No crear alias de Docker a Podman")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verificar requisitos previos
    if not check_prerequisites():
        print(f"{Colors.FAIL}✗ No se cumplen los requisitos previos. Abortando.{Colors.ENDC}")
        return 1
    
    # Ejecutar pruebas unitarias
    if not args.skip_tests:
        if not run_tests():
            print(f"{Colors.WARNING}⚠ Algunas pruebas fallaron. Continuando de todos modos.{Colors.ENDC}")
    
    # Instalar Podman
    if not args.skip_install:
        if not install_podman():
            print(f"{Colors.FAIL}✗ Error al instalar Podman. Abortando.{Colors.ENDC}")
            return 1
        
        # Probar la instalación de Podman
        if not test_podman_installation():
            print(f"{Colors.WARNING}⚠ Error al verificar la instalación de Podman. Continuando de todos modos.{Colors.ENDC}")
    
    # Migrar de Docker a Podman
    if not args.skip_migration:
        if not migrate_docker_to_podman(
            args.output_dir,
            migrate_images=not args.no_images,
            migrate_containers=not args.no_containers,
            migrate_compose=args.docker_compose,
            create_alias=not args.no_alias
        ):
            print(f"{Colors.WARNING}⚠ Error durante la migración. Continuando de todos modos.{Colors.ENDC}")
    
    # Convertir Dockerfiles
    if args.dockerfiles:
        if not convert_dockerfiles(args.dockerfiles, args.output_dir):
            print(f"{Colors.WARNING}⚠ Error al convertir Dockerfiles. Continuando de todos modos.{Colors.ENDC}")
    
    # Mostrar resumen de la migración
    show_migration_summary(args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
