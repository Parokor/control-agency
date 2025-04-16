#!/usr/bin/env python3
"""
Script para convertir Dockerfiles a Containerfiles para Podman.
Aunque Podman puede usar Dockerfiles directamente, este script realiza
algunas optimizaciones y ajustes para aprovechar mejor las características de Podman.
"""

import argparse
import os
import re
import sys
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

def parse_dockerfile(dockerfile_path):
    """Parsea un Dockerfile y devuelve sus instrucciones."""
    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Eliminar comentarios y líneas vacías
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        # Unir líneas continuadas con \
        instructions = []
        current_instruction = ""
        
        for line in lines:
            if line.endswith('\\'):
                current_instruction += line[:-1] + " "
            else:
                current_instruction += line
                instructions.append(current_instruction.strip())
                current_instruction = ""
        
        # Añadir la última instrucción si existe
        if current_instruction:
            instructions.append(current_instruction.strip())
        
        return instructions
    except Exception as e:
        print(f"{Colors.FAIL}Error al parsear el Dockerfile: {str(e)}{Colors.ENDC}")
        return []

def optimize_for_podman(instructions):
    """Optimiza las instrucciones para Podman."""
    optimized = []
    
    for instruction in instructions:
        # Convertir instrucciones a mayúsculas para facilitar la comparación
        upper_instruction = instruction.upper()
        
        # Reemplazar MAINTAINER con LABEL
        if upper_instruction.startswith('MAINTAINER'):
            parts = instruction.split(' ', 1)
            if len(parts) > 1:
                optimized.append(f"LABEL maintainer=\"{parts[1]}\"")
                print(f"{Colors.WARNING}Reemplazado MAINTAINER con LABEL maintainer{Colors.ENDC}")
            else:
                optimized.append(instruction)
        
        # Optimizar HEALTHCHECK
        elif upper_instruction.startswith('HEALTHCHECK'):
            # Podman soporta HEALTHCHECK, pero con algunas diferencias
            optimized.append(instruction)
            print(f"{Colors.OKBLUE}HEALTHCHECK detectado: Podman soporta esta instrucción{Colors.ENDC}")
        
        # Optimizar USER
        elif upper_instruction.startswith('USER'):
            # Podman maneja los usuarios de manera diferente, pero la instrucción es compatible
            optimized.append(instruction)
            print(f"{Colors.OKBLUE}USER detectado: Podman maneja los usuarios de manera diferente{Colors.ENDC}")
        
        # Advertir sobre SHELL
        elif upper_instruction.startswith('SHELL'):
            optimized.append(instruction)
            print(f"{Colors.WARNING}SHELL detectado: Asegúrese de que el shell especificado esté disponible en Podman{Colors.ENDC}")
        
        # Advertir sobre ADD con URLs
        elif upper_instruction.startswith('ADD') and ('http://' in instruction or 'https://' in instruction):
            optimized.append(instruction)
            print(f"{Colors.WARNING}ADD con URL detectado: Considere usar RUN curl o RUN wget en su lugar{Colors.ENDC}")
        
        # Añadir la instrucción sin cambios
        else:
            optimized.append(instruction)
    
    return optimized

def add_podman_specific_instructions(instructions):
    """Añade instrucciones específicas de Podman."""
    result = []
    
    # Añadir comentario al principio
    result.append("# Containerfile optimizado para Podman")
    result.append("# Convertido automáticamente desde Dockerfile")
    result.append("")
    
    # Buscar la instrucción FROM
    from_index = -1
    for i, instruction in enumerate(instructions):
        if instruction.upper().startswith('FROM'):
            from_index = i
            break
    
    # Añadir instrucciones antes de FROM
    if from_index > 0:
        result.extend(instructions[:from_index])
    
    # Añadir FROM
    if from_index >= 0:
        result.append(instructions[from_index])
    
    # Añadir LABEL para Podman
    result.append("LABEL io.podman.annotations.init=\"true\"")
    
    # Añadir el resto de instrucciones
    if from_index >= 0:
        result.extend(instructions[from_index + 1:])
    else:
        result.extend(instructions)
    
    return result

def write_containerfile(instructions, output_path):
    """Escribe las instrucciones en un Containerfile."""
    try:
        with open(output_path, 'w') as f:
            for instruction in instructions:
                f.write(instruction + '\n')
        
        print(f"{Colors.OKGREEN}Containerfile escrito en {output_path}{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.FAIL}Error al escribir el Containerfile: {str(e)}{Colors.ENDC}")
        return False

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Convertir Dockerfile a Containerfile para Podman")
    parser.add_argument("dockerfile", help="Ruta al Dockerfile a convertir")
    parser.add_argument("--output", "-o", help="Ruta de salida para el Containerfile (por defecto: Containerfile)")
    parser.add_argument("--optimize", "-p", action="store_true", help="Optimizar para Podman")
    
    args = parser.parse_args()
    
    # Verificar que el Dockerfile existe
    if not os.path.exists(args.dockerfile):
        print(f"{Colors.FAIL}Error: El archivo {args.dockerfile} no existe{Colors.ENDC}")
        return 1
    
    # Determinar la ruta de salida
    output_path = args.output if args.output else "Containerfile"
    
    # Parsear el Dockerfile
    print(f"{Colors.HEADER}Parseando Dockerfile: {args.dockerfile}{Colors.ENDC}")
    instructions = parse_dockerfile(args.dockerfile)
    
    if not instructions:
        print(f"{Colors.FAIL}Error: No se encontraron instrucciones en el Dockerfile{Colors.ENDC}")
        return 1
    
    print(f"{Colors.OKGREEN}Se encontraron {len(instructions)} instrucciones en el Dockerfile{Colors.ENDC}")
    
    # Optimizar para Podman si se solicita
    if args.optimize:
        print(f"{Colors.HEADER}Optimizando instrucciones para Podman...{Colors.ENDC}")
        instructions = optimize_for_podman(instructions)
        
        print(f"{Colors.HEADER}Añadiendo instrucciones específicas de Podman...{Colors.ENDC}")
        instructions = add_podman_specific_instructions(instructions)
    
    # Escribir el Containerfile
    print(f"{Colors.HEADER}Escribiendo Containerfile: {output_path}{Colors.ENDC}")
    if write_containerfile(instructions, output_path):
        print(f"{Colors.OKGREEN}Conversión completada con éxito{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}Error al escribir el Containerfile{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
