#!/usr/bin/env python3
"""
Script principal para la gestión de recursos en la nube.
Este script proporciona una interfaz de línea de comandos para gestionar
recursos en diferentes plataformas en la nube.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Optional
from auth.token_manager import TokenManager
from cloud.platform_manager import get_platform_instance
from cloud.resource_scheduler import ResourceScheduler, Task, TaskPriority, TaskStatus, PerformancePredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "cloud_manager.log"))
    ]
)
logger = logging.getLogger(__name__)

# Crear directorio de logs si no existe
os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)

def setup_tokens(args):
    """
    Configura los tokens de autenticación.
    
    Args:
        args: Argumentos de la línea de comandos.
    """
    token_manager = TokenManager()
    
    if args.save_token:
        if not args.platform or not args.token:
            print("Error: Se requiere una plataforma (--platform) y un token (--token) para guardar un token.")
            return
        
        if token_manager.save_token(args.platform, args.token):
            print(f"Token para {args.platform} guardado correctamente.")
        else:
            print(f"Error al guardar el token para {args.platform}.")
    
    elif args.list_tokens:
        platforms = token_manager.list_platforms()
        if platforms:
            print("Plataformas con tokens guardados:")
            for platform in platforms:
                print(f"- {platform}")
        else:
            print("No hay tokens guardados.")
    
    elif args.verify_token:
        if not args.platform:
            print("Error: Se requiere una plataforma (--platform) para verificar un token.")
            return
        
        if token_manager.verify_token(args.platform):
            print(f"El token para {args.platform} es válido.")
        else:
            print(f"El token para {args.platform} no es válido o no existe.")
    
    elif args.delete_token:
        if not args.platform:
            print("Error: Se requiere una plataforma (--platform) para eliminar un token.")
            return
        
        if token_manager.delete_token(args.platform):
            print(f"Token para {args.platform} eliminado correctamente.")
        else:
            print(f"Error al eliminar el token para {args.platform}.")

def manage_platforms(args):
    """
    Gestiona las plataformas en la nube.
    
    Args:
        args: Argumentos de la línea de comandos.
    """
    token_manager = TokenManager()
    
    if args.test_platform:
        platform = get_platform_instance(args.test_platform, token_manager)
        if platform and platform.test_connection():
            print(f"Conexión exitosa con {args.test_platform}")
        else:
            print(f"Error al conectar con {args.test_platform}")
    
    elif args.list_instances:
        if not args.platform:
            print("Error: Se requiere una plataforma (--platform) para listar instancias.")
            return
        
        platform = get_platform_instance(args.platform, token_manager)
        if not platform:
            print(f"Error al obtener la instancia de plataforma para {args.platform}")
            return
        
        instances = platform.list_instances()
        if instances:
            print(f"Instancias disponibles en {args.platform}:")
            for i, instance in enumerate(instances, 1):
                print(f"{i}. ID: {instance.get('id')}, Nombre: {instance.get('name', 'N/A')}, Estado: {instance.get('status', 'N/A')}")
        else:
            print(f"No hay instancias disponibles en {args.platform} o no se pudieron obtener.")
    
    elif args.create_instance:
        if not args.platform:
            print("Error: Se requiere una plataforma (--platform) para crear una instancia.")
            return
        
        platform = get_platform_instance(args.platform, token_manager)
        if not platform:
            print(f"Error al obtener la instancia de plataforma para {args.platform}")
            return
        
        instance_id = platform.create_instance()
        if instance_id:
            print(f"Instancia creada correctamente en {args.platform}. ID: {instance_id}")
        else:
            print(f"Error al crear la instancia en {args.platform}")
    
    elif args.stop_instance:
        if not args.platform or not args.instance_id:
            print("Error: Se requiere una plataforma (--platform) y un ID de instancia (--instance-id) para detener una instancia.")
            return
        
        platform = get_platform_instance(args.platform, token_manager)
        if not platform:
            print(f"Error al obtener la instancia de plataforma para {args.platform}")
            return
        
        if platform.stop_instance(args.instance_id):
            print(f"Instancia {args.instance_id} detenida correctamente en {args.platform}")
        else:
            print(f"Error al detener la instancia {args.instance_id} en {args.platform}")
    
    elif args.instance_status:
        if not args.platform or not args.instance_id:
            print("Error: Se requiere una plataforma (--platform) y un ID de instancia (--instance-id) para obtener el estado de una instancia.")
            return
        
        platform = get_platform_instance(args.platform, token_manager)
        if not platform:
            print(f"Error al obtener la instancia de plataforma para {args.platform}")
            return
        
        status = platform.get_instance_status(args.instance_id)
        if status:
            print(f"Estado de la instancia {args.instance_id} en {args.platform}: {status}")
        else:
            print(f"Error al obtener el estado de la instancia {args.instance_id} en {args.platform}")
    
    elif args.anti_timeout:
        if not args.platform or not args.instance_id:
            print("Error: Se requiere una plataforma (--platform) y un ID de instancia (--instance-id) para implementar mecanismos anti-timeout.")
            return
        
        platform = get_platform_instance(args.platform, token_manager)
        if not platform:
            print(f"Error al obtener la instancia de plataforma para {args.platform}")
            return
        
        if platform.implement_anti_timeout(args.instance_id):
            print(f"Mecanismos anti-timeout implementados correctamente en la instancia {args.instance_id} en {args.platform}")
        else:
            print(f"Error al implementar mecanismos anti-timeout en la instancia {args.instance_id} en {args.platform}")

def manage_tasks(args):
    """
    Gestiona las tareas en el programador de recursos.
    
    Args:
        args: Argumentos de la línea de comandos.
    """
    scheduler = ResourceScheduler()
    
    if args.start_scheduler:
        scheduler.start()
        print("Programador de recursos iniciado.")
    
    elif args.stop_scheduler:
        scheduler.stop()
        print("Programador de recursos detenido.")
    
    elif args.scheduler_status:
        status = scheduler.get_scheduler_status()
        print("Estado del programador de recursos:")
        print(f"- En ejecución: {status['running']}")
        print(f"- Tareas concurrentes máximas: {status['max_concurrent_tasks']}")
        print(f"- Tareas pendientes: {status['pending_tasks']}")
        print(f"- Total de tareas: {status['total_tasks']}")
        print(f"- Plataformas: {', '.join(status['platforms'])}")
        print("- Instancias:")
        for platform, instances in status['instances'].items():
            print(f"  - {platform}: {len(instances)} instancias")
    
    elif args.submit_task:
        if not args.task_id:
            print("Error: Se requiere un ID de tarea (--task-id) para enviar una tarea.")
            return
        
        try:
            with open(args.submit_task, 'r') as f:
                code = f.read()
            
            platforms = args.platforms.split(',') if args.platforms else ["colab", "paperspace", "runpod"]
            
            task = Task(
                task_id=args.task_id,
                code=code,
                platform_preference=platforms,
                priority=TaskPriority[args.priority] if args.priority else TaskPriority.MEDIUM,
                timeout=args.timeout if args.timeout else 3600
            )
            
            if scheduler.submit_task(task):
                print(f"Tarea {args.task_id} enviada correctamente.")
            else:
                print(f"Error al enviar la tarea {args.task_id}.")
        except Exception as e:
            print(f"Error al enviar la tarea: {str(e)}")
    
    elif args.cancel_task:
        if scheduler.cancel_task(args.cancel_task):
            print(f"Tarea {args.cancel_task} cancelada correctamente.")
        else:
            print(f"Error al cancelar la tarea {args.cancel_task}.")
    
    elif args.get_task:
        task = scheduler.get_task(args.get_task)
        if task:
            print(f"Tarea {task.task_id}:")
            print(f"- Estado: {task.status.name}")
            print(f"- Plataforma: {task.platform}")
            print(f"- Instancia: {task.instance_id}")
            print(f"- Inicio: {time.ctime(task.start_time) if task.start_time else 'N/A'}")
            print(f"- Fin: {time.ctime(task.end_time) if task.end_time else 'N/A'}")
            if task.result:
                print(f"- Resultado: {task.result}")
            if task.error:
                print(f"- Error: {task.error}")
        else:
            print(f"No se encontró la tarea {args.get_task}.")
    
    elif args.list_tasks:
        status_filter = TaskStatus[args.task_status] if args.task_status else None
        tasks = scheduler.get_tasks(status_filter)
        if tasks:
            print(f"Tareas ({len(tasks)}):")
            for task in tasks:
                print(f"- {task.task_id}: {task.status.name}")
        else:
            print("No hay tareas.")
    
    elif args.platform_status:
        status = scheduler.get_platform_status(args.platform_status)
        print(f"Estado de la plataforma {args.platform_status}:")
        print(f"- Estado: {status['status']}")
        if status['status'] == "ERROR":
            print(f"- Mensaje: {status['message']}")
        else:
            print(f"- Instancias: {len(status['instances'])}")
            for i, instance in enumerate(status['instances'], 1):
                print(f"  {i}. ID: {instance.get('id')}, Estado: {instance.get('status')}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Gestión de recursos en la nube")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Subparser para la gestión de tokens
    token_parser = subparsers.add_parser("token", help="Gestión de tokens de autenticación")
    token_parser.add_argument("--save-token", action="store_true", help="Guardar un token")
    token_parser.add_argument("--list-tokens", action="store_true", help="Listar tokens guardados")
    token_parser.add_argument("--verify-token", action="store_true", help="Verificar un token")
    token_parser.add_argument("--delete-token", action="store_true", help="Eliminar un token")
    token_parser.add_argument("--platform", help="Plataforma para el token")
    token_parser.add_argument("--token", help="Token de autenticación")
    
    # Subparser para la gestión de plataformas
    platform_parser = subparsers.add_parser("platform", help="Gestión de plataformas en la nube")
    platform_parser.add_argument("--test-platform", help="Probar la conexión con una plataforma")
    platform_parser.add_argument("--list-instances", action="store_true", help="Listar instancias disponibles")
    platform_parser.add_argument("--create-instance", action="store_true", help="Crear una instancia")
    platform_parser.add_argument("--stop-instance", action="store_true", help="Detener una instancia")
    platform_parser.add_argument("--instance-status", action="store_true", help="Obtener el estado de una instancia")
    platform_parser.add_argument("--anti-timeout", action="store_true", help="Implementar mecanismos anti-timeout")
    platform_parser.add_argument("--platform", help="Plataforma para la operación")
    platform_parser.add_argument("--instance-id", help="ID de la instancia")
    
    # Subparser para la gestión de tareas
    task_parser = subparsers.add_parser("task", help="Gestión de tareas en el programador de recursos")
    task_parser.add_argument("--start-scheduler", action="store_true", help="Iniciar el programador de recursos")
    task_parser.add_argument("--stop-scheduler", action="store_true", help="Detener el programador de recursos")
    task_parser.add_argument("--scheduler-status", action="store_true", help="Obtener el estado del programador de recursos")
    task_parser.add_argument("--submit-task", help="Enviar una tarea (archivo de código)")
    task_parser.add_argument("--task-id", help="ID de la tarea")
    task_parser.add_argument("--priority", choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"], help="Prioridad de la tarea")
    task_parser.add_argument("--platforms", help="Plataformas preferidas (separadas por comas)")
    task_parser.add_argument("--timeout", type=int, help="Tiempo máximo de ejecución en segundos")
    task_parser.add_argument("--cancel-task", help="Cancelar una tarea (ID)")
    task_parser.add_argument("--get-task", help="Obtener una tarea (ID)")
    task_parser.add_argument("--list-tasks", action="store_true", help="Listar todas las tareas")
    task_parser.add_argument("--task-status", choices=["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"], help="Filtrar tareas por estado")
    task_parser.add_argument("--platform-status", help="Obtener el estado de una plataforma")
    
    args = parser.parse_args()
    
    if args.command == "token":
        setup_tokens(args)
    elif args.command == "platform":
        manage_platforms(args)
    elif args.command == "task":
        manage_tasks(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
