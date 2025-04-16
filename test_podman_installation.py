#!/usr/bin/env python3
"""
Script para verificar la instalación de Podman y comparar su funcionamiento con Docker.
Este script realiza pruebas básicas para asegurar que Podman está instalado correctamente
y puede realizar operaciones fundamentales con contenedores.
"""

import subprocess
import sys
import os
import json
from datetime import datetime

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

def run_command(command, check=True):
    """Ejecuta un comando y devuelve su salida."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
            shell=True
        )
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), e.returncode

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

def check_podman_info():
    """Obtiene información detallada de Podman."""
    print(f"{Colors.HEADER}Obteniendo información de Podman...{Colors.ENDC}")
    output, return_code = run_command("podman info --format json", check=False)
    
    if return_code == 0:
        try:
            info = json.loads(output)
            print(f"{Colors.OKGREEN}✓ Podman info ejecutado correctamente{Colors.ENDC}")
            print(f"{Colors.OKBLUE}Versión: {info.get('version', {}).get('Version', 'N/A')}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}Sistema operativo: {info.get('host', {}).get('os', 'N/A')}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}Arquitectura: {info.get('host', {}).get('arch', 'N/A')}{Colors.ENDC}")
            return True
        except json.JSONDecodeError:
            print(f"{Colors.FAIL}✗ Error al parsear la salida JSON de podman info{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ Error al ejecutar podman info{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return False

def test_basic_container():
    """Prueba la creación y ejecución de un contenedor básico."""
    print(f"{Colors.HEADER}Probando la creación de un contenedor básico...{Colors.ENDC}")
    
    # Intentar eliminar el contenedor si ya existe
    run_command("podman rm -f test-container", check=False)
    
    # Ejecutar un contenedor simple
    output, return_code = run_command(
        "podman run --name test-container -d alpine echo 'Hello from Podman!'",
        check=False
    )
    
    if return_code == 0:
        print(f"{Colors.OKGREEN}✓ Contenedor creado correctamente: {output[:12]}{Colors.ENDC}")
        
        # Verificar los logs del contenedor
        logs_output, logs_code = run_command("podman logs test-container", check=False)
        if logs_code == 0 and "Hello from Podman!" in logs_output:
            print(f"{Colors.OKGREEN}✓ Logs del contenedor verificados correctamente{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Error al verificar los logs del contenedor{Colors.ENDC}")
            print(f"{Colors.FAIL}Logs: {logs_output}{Colors.ENDC}")
        
        # Limpiar
        run_command("podman rm test-container", check=False)
        return True
    else:
        print(f"{Colors.FAIL}✗ Error al crear el contenedor{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {output}{Colors.ENDC}")
        return False

def test_volume_mount():
    """Prueba la creación y montaje de volúmenes."""
    print(f"{Colors.HEADER}Probando la creación y montaje de volúmenes...{Colors.ENDC}")
    
    # Crear un volumen
    volume_output, volume_code = run_command("podman volume create test-volume", check=False)
    
    if volume_code == 0:
        print(f"{Colors.OKGREEN}✓ Volumen creado correctamente: {volume_output}{Colors.ENDC}")
        
        # Crear un archivo en el volumen
        temp_dir = "/tmp/podman-test-" + datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(temp_dir, exist_ok=True)
        test_file = os.path.join(temp_dir, "test.txt")
        
        with open(test_file, "w") as f:
            f.write("Test file for Podman volume mount")
        
        # Montar el volumen y verificar el archivo
        mount_output, mount_code = run_command(
            f"podman run --rm -v {temp_dir}:/data alpine cat /data/test.txt",
            check=False
        )
        
        if mount_code == 0 and "Test file for Podman volume mount" in mount_output:
            print(f"{Colors.OKGREEN}✓ Montaje de volumen verificado correctamente{Colors.ENDC}")
            
            # Limpiar
            run_command("podman volume rm test-volume", check=False)
            os.remove(test_file)
            os.rmdir(temp_dir)
            return True
        else:
            print(f"{Colors.FAIL}✗ Error al verificar el montaje del volumen{Colors.ENDC}")
            print(f"{Colors.FAIL}Salida: {mount_output}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ Error al crear el volumen{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {volume_output}{Colors.ENDC}")
        return False

def test_network():
    """Prueba la creación y uso de redes."""
    print(f"{Colors.HEADER}Probando la creación y uso de redes...{Colors.ENDC}")
    
    # Crear una red
    network_output, network_code = run_command("podman network create test-network", check=False)
    
    if network_code == 0:
        print(f"{Colors.OKGREEN}✓ Red creada correctamente: {network_output}{Colors.ENDC}")
        
        # Ejecutar un contenedor en la red
        container_output, container_code = run_command(
            "podman run --rm --network test-network alpine ip addr show",
            check=False
        )
        
        if container_code == 0:
            print(f"{Colors.OKGREEN}✓ Contenedor ejecutado correctamente en la red{Colors.ENDC}")
            
            # Limpiar
            run_command("podman network rm test-network", check=False)
            return True
        else:
            print(f"{Colors.FAIL}✗ Error al ejecutar el contenedor en la red{Colors.ENDC}")
            print(f"{Colors.FAIL}Error: {container_output}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ Error al crear la red{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {network_output}{Colors.ENDC}")
        return False

def test_pod():
    """Prueba la creación y gestión de pods."""
    print(f"{Colors.HEADER}Probando la creación y gestión de pods...{Colors.ENDC}")
    
    # Crear un pod
    pod_output, pod_code = run_command("podman pod create --name test-pod", check=False)
    
    if pod_code == 0:
        print(f"{Colors.OKGREEN}✓ Pod creado correctamente: {pod_output}{Colors.ENDC}")
        
        # Ejecutar un contenedor en el pod
        container_output, container_code = run_command(
            "podman run --pod test-pod -d alpine sleep 300",
            check=False
        )
        
        if container_code == 0:
            print(f"{Colors.OKGREEN}✓ Contenedor ejecutado correctamente en el pod{Colors.ENDC}")
            
            # Verificar el estado del pod
            pod_ps_output, pod_ps_code = run_command("podman pod ps", check=False)
            
            if pod_ps_code == 0 and "test-pod" in pod_ps_output:
                print(f"{Colors.OKGREEN}✓ Estado del pod verificado correctamente{Colors.ENDC}")
                
                # Limpiar
                run_command("podman pod rm -f test-pod", check=False)
                return True
            else:
                print(f"{Colors.FAIL}✗ Error al verificar el estado del pod{Colors.ENDC}")
                return False
        else:
            print(f"{Colors.FAIL}✗ Error al ejecutar el contenedor en el pod{Colors.ENDC}")
            print(f"{Colors.FAIL}Error: {container_output}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.FAIL}✗ Error al crear el pod{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {pod_output}{Colors.ENDC}")
        return False

def compare_with_docker():
    """Compara la funcionalidad básica con Docker si está disponible."""
    print(f"{Colors.HEADER}Comparando con Docker (si está disponible)...{Colors.ENDC}")
    
    docker_output, docker_code = run_command("docker --version", check=False)
    
    if docker_code == 0:
        print(f"{Colors.OKGREEN}✓ Docker está instalado: {docker_output}{Colors.ENDC}")
        
        # Comparar tiempos de inicio de contenedor
        print(f"{Colors.OKBLUE}Comparando tiempos de inicio de contenedor...{Colors.ENDC}")
        
        # Docker
        docker_time_start = datetime.now()
        run_command("docker run --rm alpine echo 'test'", check=False)
        docker_time_end = datetime.now()
        docker_time = (docker_time_end - docker_time_start).total_seconds()
        
        # Podman
        podman_time_start = datetime.now()
        run_command("podman run --rm alpine echo 'test'", check=False)
        podman_time_end = datetime.now()
        podman_time = (podman_time_end - podman_time_start).total_seconds()
        
        print(f"{Colors.OKBLUE}Tiempo de Docker: {docker_time:.2f} segundos{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Tiempo de Podman: {podman_time:.2f} segundos{Colors.ENDC}")
        
        if podman_time < docker_time:
            print(f"{Colors.OKGREEN}✓ Podman es más rápido que Docker en este caso{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠ Docker es más rápido que Podman en este caso{Colors.ENDC}")
        
        return True
    else:
        print(f"{Colors.WARNING}⚠ Docker no está instalado, omitiendo comparación{Colors.ENDC}")
        return False

def main():
    """Función principal que ejecuta todas las pruebas."""
    print(f"{Colors.BOLD}{Colors.HEADER}=== Prueba de Instalación de Podman ==={Colors.ENDC}")
    
    tests = [
        check_podman_installed,
        check_podman_info,
        test_basic_container,
        test_volume_mount,
        test_network,
        test_pod,
        compare_with_docker
    ]
    
    results = []
    
    for test in tests:
        result = test()
        results.append(result)
    
    # Resumen
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== Resumen de Pruebas ==={Colors.ENDC}")
    
    all_passed = all(results[:6])  # Excluimos la comparación con Docker
    
    if all_passed:
        print(f"{Colors.BOLD}{Colors.OKGREEN}✓ Todas las pruebas de Podman pasaron correctamente{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.FAIL}✗ Algunas pruebas de Podman fallaron{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
