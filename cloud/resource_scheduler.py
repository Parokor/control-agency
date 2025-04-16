#!/usr/bin/env python3
"""
Módulo para la programación y gestión de recursos en la nube.
Este módulo proporciona clases para gestionar la asignación de recursos
en diferentes plataformas en la nube de forma inteligente.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from cloud.platform_manager import get_platform_instance, CloudPlatform
from auth.token_manager import TokenManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Enumeración para las prioridades de las tareas."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Enumeración para los estados de las tareas."""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4

class Task:
    """Clase para representar una tarea a ejecutar en la nube."""
    
    def __init__(self, task_id: str, code: str, platform_preference: List[str] = None, priority: TaskPriority = TaskPriority.MEDIUM, timeout: int = 3600, callback: Callable = None):
        """
        Inicializa una tarea.
        
        Args:
            task_id: ID único de la tarea.
            code: Código a ejecutar.
            platform_preference: Lista de plataformas preferidas en orden de preferencia.
            priority: Prioridad de la tarea.
            timeout: Tiempo máximo de ejecución en segundos.
            callback: Función a llamar cuando la tarea se complete.
        """
        self.task_id = task_id
        self.code = code
        self.platform_preference = platform_preference or ["colab", "paperspace", "runpod"]
        self.priority = priority
        self.timeout = timeout
        self.callback = callback
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.platform = None
        self.instance_id = None
        self.start_time = None
        self.end_time = None
    
    def __lt__(self, other):
        """
        Comparación para la cola de prioridad.
        
        Args:
            other: Otra tarea.
        
        Returns:
            bool: True si esta tarea tiene mayor prioridad que la otra.
        """
        if not isinstance(other, Task):
            return NotImplemented
        return self.priority.value > other.priority.value
    
    def to_dict(self) -> Dict:
        """
        Convierte la tarea a un diccionario.
        
        Returns:
            Dict: Diccionario con los datos de la tarea.
        """
        return {
            "task_id": self.task_id,
            "code": self.code,
            "platform_preference": self.platform_preference,
            "priority": self.priority.name,
            "timeout": self.timeout,
            "status": self.status.name,
            "result": self.result,
            "error": self.error,
            "platform": self.platform,
            "instance_id": self.instance_id,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """
        Crea una tarea a partir de un diccionario.
        
        Args:
            data: Diccionario con los datos de la tarea.
        
        Returns:
            Task: Tarea creada.
        """
        task = cls(
            task_id=data["task_id"],
            code=data["code"],
            platform_preference=data.get("platform_preference", ["colab", "paperspace", "runpod"]),
            priority=TaskPriority[data.get("priority", "MEDIUM")],
            timeout=data.get("timeout", 3600),
            callback=None  # No se puede serializar la función callback
        )
        task.status = TaskStatus[data.get("status", "PENDING")]
        task.result = data.get("result")
        task.error = data.get("error")
        task.platform = data.get("platform")
        task.instance_id = data.get("instance_id")
        task.start_time = data.get("start_time")
        task.end_time = data.get("end_time")
        return task

class ResourceScheduler:
    """Clase para la programación y gestión de recursos en la nube."""
    
    def __init__(self, token_manager: TokenManager = None, max_concurrent_tasks: int = 5, task_queue_file: str = None):
        """
        Inicializa el programador de recursos.
        
        Args:
            token_manager: Gestor de tokens.
            max_concurrent_tasks: Número máximo de tareas concurrentes.
            task_queue_file: Archivo para guardar la cola de tareas.
        """
        self.token_manager = token_manager or TokenManager()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue_file = task_queue_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "task_queue.json")
        
        # Crear directorio para el archivo de cola de tareas si no existe
        os.makedirs(os.path.dirname(self.task_queue_file), exist_ok=True)
        
        # Cola de tareas pendientes (ordenada por prioridad)
        self.task_queue = queue.PriorityQueue()
        
        # Diccionario de tareas por ID
        self.tasks = {}
        
        # Diccionario de plataformas
        self.platforms = {}
        
        # Diccionario de instancias por plataforma
        self.instances = {}
        
        # Semáforo para limitar el número de tareas concurrentes
        self.semaphore = threading.Semaphore(max_concurrent_tasks)
        
        # Evento para detener el programador
        self.stop_event = threading.Event()
        
        # Hilo para procesar la cola de tareas
        self.queue_processor_thread = None
        
        # Cargar tareas pendientes
        self._load_tasks()
    
    def _load_tasks(self):
        """Carga las tareas pendientes desde el archivo."""
        if not os.path.exists(self.task_queue_file):
            return
        
        try:
            with open(self.task_queue_file, 'r') as f:
                tasks_data = json.load(f)
            
            for task_data in tasks_data:
                task = Task.from_dict(task_data)
                if task.status == TaskStatus.PENDING:
                    self.tasks[task.task_id] = task
                    self.task_queue.put(task)
            
            logger.info(f"Cargadas {len(self.tasks)} tareas pendientes.")
        except Exception as e:
            logger.error(f"Error al cargar las tareas pendientes: {str(e)}")
    
    def _save_tasks(self):
        """Guarda las tareas pendientes en el archivo."""
        try:
            tasks_data = [task.to_dict() for task in self.tasks.values()]
            
            with open(self.task_queue_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            logger.info(f"Guardadas {len(tasks_data)} tareas.")
        except Exception as e:
            logger.error(f"Error al guardar las tareas: {str(e)}")
    
    def _get_platform(self, platform_name: str) -> Optional[CloudPlatform]:
        """
        Obtiene una instancia de plataforma en la nube.
        
        Args:
            platform_name: Nombre de la plataforma.
        
        Returns:
            CloudPlatform: Instancia de plataforma en la nube o None si hay un error.
        """
        if platform_name not in self.platforms:
            platform = get_platform_instance(platform_name, self.token_manager)
            if platform:
                self.platforms[platform_name] = platform
            else:
                logger.error(f"No se pudo obtener la instancia de plataforma para {platform_name}")
                return None
        
        return self.platforms[platform_name]
    
    def _get_available_instance(self, platform_name: str) -> Optional[str]:
        """
        Obtiene una instancia disponible en la plataforma especificada.
        
        Args:
            platform_name: Nombre de la plataforma.
        
        Returns:
            str: ID de la instancia disponible o None si no hay instancias disponibles.
        """
        platform = self._get_platform(platform_name)
        if not platform:
            return None
        
        # Verificar si hay instancias disponibles
        if platform_name in self.instances and self.instances[platform_name]:
            # Verificar el estado de las instancias
            for instance_id in list(self.instances[platform_name]):
                status = platform.get_instance_status(instance_id)
                if status and status.upper() in ["RUNNING", "READY", "AVAILABLE"]:
                    return instance_id
                else:
                    # Eliminar la instancia si no está disponible
                    self.instances[platform_name].remove(instance_id)
        
        # Crear una nueva instancia si no hay instancias disponibles
        instance_id = platform.create_instance()
        if instance_id:
            if platform_name not in self.instances:
                self.instances[platform_name] = []
            self.instances[platform_name].append(instance_id)
            
            # Implementar mecanismos anti-timeout
            platform.implement_anti_timeout(instance_id)
            
            return instance_id
        
        return None
    
    def _process_task(self, task: Task):
        """
        Procesa una tarea.
        
        Args:
            task: Tarea a procesar.
        """
        try:
            # Marcar la tarea como en ejecución
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            # Intentar ejecutar la tarea en las plataformas preferidas
            for platform_name in task.platform_preference:
                platform = self._get_platform(platform_name)
                if not platform:
                    continue
                
                # Obtener una instancia disponible
                instance_id = self._get_available_instance(platform_name)
                if not instance_id:
                    logger.warning(f"No se pudo obtener una instancia disponible en {platform_name}")
                    continue
                
                # Ejecutar la tarea
                task.platform = platform_name
                task.instance_id = instance_id
                
                result = platform.execute_code(instance_id, task.code)
                if result:
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.end_time = time.time()
                    logger.info(f"Tarea {task.task_id} completada en {platform_name}")
                    break
                else:
                    logger.warning(f"Error al ejecutar la tarea {task.task_id} en {platform_name}")
            
            # Si la tarea no se pudo completar en ninguna plataforma
            if task.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.FAILED
                task.error = "No se pudo ejecutar la tarea en ninguna plataforma"
                task.end_time = time.time()
                logger.error(f"Tarea {task.task_id} fallida: {task.error}")
            
            # Llamar al callback si existe
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error en el callback de la tarea {task.task_id}: {str(e)}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
            logger.error(f"Error al procesar la tarea {task.task_id}: {str(e)}")
        finally:
            # Guardar las tareas
            self._save_tasks()
            
            # Liberar el semáforo
            self.semaphore.release()
    
    def _process_queue(self):
        """Procesa la cola de tareas."""
        while not self.stop_event.is_set():
            try:
                # Obtener una tarea de la cola (con timeout para poder comprobar el evento de parada)
                try:
                    task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Adquirir el semáforo (limita el número de tareas concurrentes)
                self.semaphore.acquire()
                
                # Procesar la tarea en un hilo separado
                threading.Thread(target=self._process_task, args=(task,), daemon=True).start()
            except Exception as e:
                logger.error(f"Error al procesar la cola de tareas: {str(e)}")
    
    def start(self):
        """Inicia el programador de recursos."""
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            logger.warning("El programador de recursos ya está en ejecución.")
            return
        
        # Reiniciar el evento de parada
        self.stop_event.clear()
        
        # Iniciar el hilo para procesar la cola de tareas
        self.queue_processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_processor_thread.start()
        
        logger.info("Programador de recursos iniciado.")
    
    def stop(self):
        """Detiene el programador de recursos."""
        # Establecer el evento de parada
        self.stop_event.set()
        
        # Esperar a que el hilo termine
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            self.queue_processor_thread.join(timeout=5)
        
        # Guardar las tareas
        self._save_tasks()
        
        logger.info("Programador de recursos detenido.")
    
    def submit_task(self, task: Task) -> bool:
        """
        Envía una tarea para su ejecución.
        
        Args:
            task: Tarea a ejecutar.
        
        Returns:
            bool: True si la tarea se envió correctamente, False en caso contrario.
        """
        try:
            # Verificar si la tarea ya existe
            if task.task_id in self.tasks:
                logger.warning(f"La tarea {task.task_id} ya existe.")
                return False
            
            # Añadir la tarea al diccionario y a la cola
            self.tasks[task.task_id] = task
            self.task_queue.put(task)
            
            # Guardar las tareas
            self._save_tasks()
            
            logger.info(f"Tarea {task.task_id} enviada correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al enviar la tarea {task.task_id}: {str(e)}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea.
        
        Args:
            task_id: ID de la tarea a cancelar.
        
        Returns:
            bool: True si la tarea se canceló correctamente, False en caso contrario.
        """
        try:
            # Verificar si la tarea existe
            if task_id not in self.tasks:
                logger.warning(f"La tarea {task_id} no existe.")
                return False
            
            task = self.tasks[task_id]
            
            # Verificar si la tarea ya está completada o fallida
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.warning(f"La tarea {task_id} ya está {task.status.name.lower()}.")
                return False
            
            # Marcar la tarea como cancelada
            task.status = TaskStatus.CANCELLED
            task.end_time = time.time()
            
            # Guardar las tareas
            self._save_tasks()
            
            logger.info(f"Tarea {task_id} cancelada correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al cancelar la tarea {task_id}: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Obtiene una tarea por su ID.
        
        Args:
            task_id: ID de la tarea.
        
        Returns:
            Task: Tarea o None si no existe.
        """
        return self.tasks.get(task_id)
    
    def get_tasks(self, status: TaskStatus = None) -> List[Task]:
        """
        Obtiene todas las tareas o las tareas con un estado específico.
        
        Args:
            status: Estado de las tareas a obtener.
        
        Returns:
            List[Task]: Lista de tareas.
        """
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        else:
            return list(self.tasks.values())
    
    def get_platform_status(self, platform_name: str) -> Dict:
        """
        Obtiene el estado de una plataforma.
        
        Args:
            platform_name: Nombre de la plataforma.
        
        Returns:
            Dict: Estado de la plataforma.
        """
        platform = self._get_platform(platform_name)
        if not platform:
            return {"status": "ERROR", "message": f"No se pudo obtener la instancia de plataforma para {platform_name}"}
        
        # Verificar la conexión
        if not platform.test_connection():
            return {"status": "ERROR", "message": f"No se pudo conectar con {platform_name}"}
        
        # Obtener las instancias
        instances = platform.list_instances()
        
        return {
            "status": "OK",
            "platform": platform_name,
            "instances": instances or []
        }
    
    def get_scheduler_status(self) -> Dict:
        """
        Obtiene el estado del programador de recursos.
        
        Returns:
            Dict: Estado del programador de recursos.
        """
        return {
            "running": self.queue_processor_thread is not None and self.queue_processor_thread.is_alive(),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "pending_tasks": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "platforms": list(self.platforms.keys()),
            "instances": self.instances
        }

class PerformancePredictor:
    """Clase para predecir el rendimiento de las tareas en diferentes plataformas."""
    
    def __init__(self, history_file: str = None):
        """
        Inicializa el predictor de rendimiento.
        
        Args:
            history_file: Archivo para guardar el historial de rendimiento.
        """
        self.history_file = history_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "performance_history.json")
        
        # Crear directorio para el archivo de historial si no existe
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        # Historial de rendimiento por plataforma y tipo de tarea
        self.history = {}
        
        # Cargar historial
        self._load_history()
    
    def _load_history(self):
        """Carga el historial de rendimiento desde el archivo."""
        if not os.path.exists(self.history_file):
            return
        
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
            
            logger.info(f"Cargado historial de rendimiento con {len(self.history)} entradas.")
        except Exception as e:
            logger.error(f"Error al cargar el historial de rendimiento: {str(e)}")
    
    def _save_history(self):
        """Guarda el historial de rendimiento en el archivo."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"Guardado historial de rendimiento con {len(self.history)} entradas.")
        except Exception as e:
            logger.error(f"Error al guardar el historial de rendimiento: {str(e)}")
    
    def record_performance(self, task: Task):
        """
        Registra el rendimiento de una tarea.
        
        Args:
            task: Tarea completada.
        """
        if task.status != TaskStatus.COMPLETED or not task.start_time or not task.end_time:
            return
        
        # Calcular el tiempo de ejecución
        execution_time = task.end_time - task.start_time
        
        # Determinar el tipo de tarea (simplificado)
        task_type = "default"
        if "tensorflow" in task.code.lower() or "torch" in task.code.lower():
            task_type = "ml"
        elif "numpy" in task.code.lower() or "pandas" in task.code.lower():
            task_type = "data"
        elif "matplotlib" in task.code.lower() or "seaborn" in task.code.lower():
            task_type = "visualization"
        
        # Registrar el rendimiento
        if task.platform not in self.history:
            self.history[task.platform] = {}
        
        if task_type not in self.history[task.platform]:
            self.history[task.platform][task_type] = []
        
        self.history[task.platform][task_type].append(execution_time)
        
        # Limitar el historial a las últimas 100 entradas por tipo de tarea
        if len(self.history[task.platform][task_type]) > 100:
            self.history[task.platform][task_type] = self.history[task.platform][task_type][-100:]
        
        # Guardar el historial
        self._save_history()
    
    def predict_execution_time(self, task: Task, platform: str) -> Optional[float]:
        """
        Predice el tiempo de ejecución de una tarea en una plataforma.
        
        Args:
            task: Tarea a ejecutar.
            platform: Plataforma en la que se ejecutará la tarea.
        
        Returns:
            float: Tiempo de ejecución estimado en segundos o None si no hay datos suficientes.
        """
        # Determinar el tipo de tarea (simplificado)
        task_type = "default"
        if "tensorflow" in task.code.lower() or "torch" in task.code.lower():
            task_type = "ml"
        elif "numpy" in task.code.lower() or "pandas" in task.code.lower():
            task_type = "data"
        elif "matplotlib" in task.code.lower() or "seaborn" in task.code.lower():
            task_type = "visualization"
        
        # Verificar si hay datos para la plataforma y el tipo de tarea
        if platform not in self.history or task_type not in self.history[platform] or not self.history[platform][task_type]:
            return None
        
        # Calcular el tiempo medio de ejecución
        execution_times = self.history[platform][task_type]
        return sum(execution_times) / len(execution_times)
    
    def get_best_platform(self, task: Task, available_platforms: List[str] = None) -> Optional[str]:
        """
        Obtiene la mejor plataforma para ejecutar una tarea.
        
        Args:
            task: Tarea a ejecutar.
            available_platforms: Lista de plataformas disponibles.
        
        Returns:
            str: Nombre de la mejor plataforma o None si no hay datos suficientes.
        """
        if not available_platforms:
            available_platforms = ["colab", "paperspace", "runpod"]
        
        best_platform = None
        best_time = float('inf')
        
        for platform in available_platforms:
            predicted_time = self.predict_execution_time(task, platform)
            if predicted_time and predicted_time < best_time:
                best_platform = platform
                best_time = predicted_time
        
        return best_platform

def main():
    """Función principal para probar el programador de recursos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Programador de recursos en la nube")
    parser.add_argument("--start", action="store_true", help="Iniciar el programador de recursos")
    parser.add_argument("--stop", action="store_true", help="Detener el programador de recursos")
    parser.add_argument("--status", action="store_true", help="Obtener el estado del programador de recursos")
    parser.add_argument("--submit", help="Enviar una tarea (archivo de código)")
    parser.add_argument("--task-id", help="ID de la tarea")
    parser.add_argument("--priority", choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"], default="MEDIUM", help="Prioridad de la tarea")
    parser.add_argument("--platforms", help="Plataformas preferidas (separadas por comas)")
    parser.add_argument("--cancel", help="Cancelar una tarea (ID)")
    parser.add_argument("--get-task", help="Obtener una tarea (ID)")
    parser.add_argument("--list-tasks", action="store_true", help="Listar todas las tareas")
    parser.add_argument("--platform-status", help="Obtener el estado de una plataforma")
    
    args = parser.parse_args()
    
    # Crear instancia de ResourceScheduler
    scheduler = ResourceScheduler()
    
    # Iniciar el programador de recursos
    if args.start:
        scheduler.start()
        print("Programador de recursos iniciado.")
    
    # Detener el programador de recursos
    elif args.stop:
        scheduler.stop()
        print("Programador de recursos detenido.")
    
    # Obtener el estado del programador de recursos
    elif args.status:
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
    
    # Enviar una tarea
    elif args.submit:
        if not args.task_id:
            print("Error: Se requiere un ID de tarea (--task-id)")
            return
        
        try:
            with open(args.submit, 'r') as f:
                code = f.read()
            
            platforms = args.platforms.split(',') if args.platforms else ["colab", "paperspace", "runpod"]
            
            task = Task(
                task_id=args.task_id,
                code=code,
                platform_preference=platforms,
                priority=TaskPriority[args.priority]
            )
            
            if scheduler.submit_task(task):
                print(f"Tarea {args.task_id} enviada correctamente.")
            else:
                print(f"Error al enviar la tarea {args.task_id}.")
        except Exception as e:
            print(f"Error al enviar la tarea: {str(e)}")
    
    # Cancelar una tarea
    elif args.cancel:
        if scheduler.cancel_task(args.cancel):
            print(f"Tarea {args.cancel} cancelada correctamente.")
        else:
            print(f"Error al cancelar la tarea {args.cancel}.")
    
    # Obtener una tarea
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
    
    # Listar todas las tareas
    elif args.list_tasks:
        tasks = scheduler.get_tasks()
        if tasks:
            print(f"Tareas ({len(tasks)}):")
            for task in tasks:
                print(f"- {task.task_id}: {task.status.name}")
        else:
            print("No hay tareas.")
    
    # Obtener el estado de una plataforma
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
    
    # Mostrar ayuda si no se proporciona ninguna opción
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
