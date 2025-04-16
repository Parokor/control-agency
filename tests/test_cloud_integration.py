#!/usr/bin/env python3
"""
Pruebas unitarias para la integración con plataformas en la nube.
"""

import os
import sys
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth.token_manager import TokenManager
from auth.github_auth import GitHubAuth
from auth.cloud_auth import get_auth_instance, ColabAuth, PaperspaceAuth, RunPodAuth
from cloud.platform_manager import get_platform_instance, CloudPlatform, ColabPlatform, PaperspacePlatform, RunPodPlatform
from cloud.resource_scheduler import ResourceScheduler, Task, TaskPriority, TaskStatus, PerformancePredictor

class TestTokenManager(unittest.TestCase):
    """Pruebas para el gestor de tokens."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear un directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        self.token_manager = TokenManager(config_dir=self.test_dir, master_password="test_password")
    
    def test_save_and_load_token(self):
        """Prueba guardar y cargar un token."""
        # Guardar un token
        self.assertTrue(self.token_manager.save_token("test_platform", "test_token"))
        
        # Cargar el token
        token = self.token_manager.load_token("test_platform")
        self.assertEqual(token, "test_token")
    
    def test_delete_token(self):
        """Prueba eliminar un token."""
        # Guardar un token
        self.assertTrue(self.token_manager.save_token("test_platform", "test_token"))
        
        # Eliminar el token
        self.assertTrue(self.token_manager.delete_token("test_platform"))
        
        # Verificar que el token se ha eliminado
        self.assertIsNone(self.token_manager.load_token("test_platform"))
    
    def test_list_platforms(self):
        """Prueba listar plataformas con tokens guardados."""
        # Guardar varios tokens
        self.assertTrue(self.token_manager.save_token("platform1", "token1"))
        self.assertTrue(self.token_manager.save_token("platform2", "token2"))
        
        # Listar plataformas
        platforms = self.token_manager.list_platforms()
        self.assertIn("platform1", platforms)
        self.assertIn("platform2", platforms)

class TestGitHubAuth(unittest.TestCase):
    """Pruebas para la autenticación con GitHub."""
    
    @patch('auth.github_auth.requests.get')
    def test_test_connection(self, mock_get):
        """Prueba la conexión con GitHub."""
        # Configurar el mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"login": "test_user"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Crear instancia de GitHubAuth
        github_auth = GitHubAuth(token="test_token")
        
        # Probar la conexión
        self.assertTrue(github_auth.test_connection())
        
        # Verificar que se llamó a requests.get con los parámetros correctos
        mock_get.assert_called_once_with(
            "https://api.github.com/user",
            headers={'Authorization': 'token test_token', 'Accept': 'application/vnd.github.v3+json'}
        )
    
    @patch('auth.github_auth.requests.get')
    def test_get_repo_info(self, mock_get):
        """Prueba obtener información de un repositorio."""
        # Configurar el mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"full_name": "test_owner/test_repo"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Crear instancia de GitHubAuth
        github_auth = GitHubAuth(token="test_token")
        
        # Obtener información del repositorio
        repo_info = github_auth.get_repo_info("test_owner", "test_repo")
        
        # Verificar el resultado
        self.assertEqual(repo_info["full_name"], "test_owner/test_repo")
        
        # Verificar que se llamó a requests.get con los parámetros correctos
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/test_owner/test_repo",
            headers={'Authorization': 'token test_token', 'Accept': 'application/vnd.github.v3+json'}
        )

class TestCloudAuth(unittest.TestCase):
    """Pruebas para la autenticación con plataformas en la nube."""
    
    def test_get_auth_instance(self):
        """Prueba obtener instancias de autenticación para diferentes plataformas."""
        # Colab
        colab_auth = get_auth_instance("colab", token="test_token")
        self.assertIsInstance(colab_auth, ColabAuth)
        
        # Paperspace
        paperspace_auth = get_auth_instance("paperspace", token="test_token")
        self.assertIsInstance(paperspace_auth, PaperspaceAuth)
        
        # RunPod
        runpod_auth = get_auth_instance("runpod", token="test_token")
        self.assertIsInstance(runpod_auth, RunPodAuth)
        
        # Plataforma no soportada
        self.assertIsNone(get_auth_instance("unsupported_platform", token="test_token"))
    
    @patch('auth.cloud_auth.requests.get')
    def test_colab_auth_test_connection(self, mock_get):
        """Prueba la conexión con Google Colab."""
        # Configurar el mock
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Crear instancia de ColabAuth
        colab_auth = ColabAuth(token="test_token")
        
        # Probar la conexión
        self.assertTrue(colab_auth.test_connection())
        
        # Verificar que se llamó a requests.get con los parámetros correctos
        mock_get.assert_called_once_with(
            "https://colab.research.google.com/api/notebooks",
            headers={'Authorization': 'Bearer test_token', 'Content-Type': 'application/json'}
        )

class TestPlatformManager(unittest.TestCase):
    """Pruebas para el gestor de plataformas en la nube."""
    
    def test_get_platform_instance(self):
        """Prueba obtener instancias de plataforma para diferentes plataformas."""
        # Crear un token manager mock
        token_manager = MagicMock()
        token_manager.load_token.return_value = "test_token"
        
        # Colab
        colab_platform = get_platform_instance("colab", token_manager)
        self.assertIsInstance(colab_platform, ColabPlatform)
        
        # Paperspace
        paperspace_platform = get_platform_instance("paperspace", token_manager)
        self.assertIsInstance(paperspace_platform, PaperspacePlatform)
        
        # RunPod
        runpod_platform = get_platform_instance("runpod", token_manager)
        self.assertIsInstance(runpod_platform, RunPodPlatform)
        
        # Plataforma no soportada
        self.assertIsNone(get_platform_instance("unsupported_platform", token_manager))

class TestResourceScheduler(unittest.TestCase):
    """Pruebas para el programador de recursos."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear un directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        self.task_queue_file = os.path.join(self.test_dir, "task_queue.json")
        
        # Crear un token manager mock
        self.token_manager = MagicMock()
        self.token_manager.load_token.return_value = "test_token"
        
        # Crear un programador de recursos
        self.scheduler = ResourceScheduler(
            token_manager=self.token_manager,
            max_concurrent_tasks=2,
            task_queue_file=self.task_queue_file
        )
    
    def test_submit_and_get_task(self):
        """Prueba enviar y obtener una tarea."""
        # Crear una tarea
        task = Task(
            task_id="test_task",
            code="print('Hello, world!')",
            platform_preference=["colab", "paperspace", "runpod"],
            priority=TaskPriority.MEDIUM
        )
        
        # Enviar la tarea
        self.assertTrue(self.scheduler.submit_task(task))
        
        # Obtener la tarea
        retrieved_task = self.scheduler.get_task("test_task")
        self.assertIsNotNone(retrieved_task)
        self.assertEqual(retrieved_task.task_id, "test_task")
        self.assertEqual(retrieved_task.code, "print('Hello, world!')")
        self.assertEqual(retrieved_task.status, TaskStatus.PENDING)
    
    def test_cancel_task(self):
        """Prueba cancelar una tarea."""
        # Crear una tarea
        task = Task(
            task_id="test_task",
            code="print('Hello, world!')",
            platform_preference=["colab", "paperspace", "runpod"],
            priority=TaskPriority.MEDIUM
        )
        
        # Enviar la tarea
        self.assertTrue(self.scheduler.submit_task(task))
        
        # Cancelar la tarea
        self.assertTrue(self.scheduler.cancel_task("test_task"))
        
        # Verificar que la tarea se ha cancelado
        cancelled_task = self.scheduler.get_task("test_task")
        self.assertEqual(cancelled_task.status, TaskStatus.CANCELLED)
    
    def test_get_tasks(self):
        """Prueba obtener todas las tareas o las tareas con un estado específico."""
        # Crear varias tareas
        task1 = Task(
            task_id="task1",
            code="print('Task 1')",
            platform_preference=["colab"],
            priority=TaskPriority.LOW
        )
        
        task2 = Task(
            task_id="task2",
            code="print('Task 2')",
            platform_preference=["paperspace"],
            priority=TaskPriority.MEDIUM
        )
        
        task3 = Task(
            task_id="task3",
            code="print('Task 3')",
            platform_preference=["runpod"],
            priority=TaskPriority.HIGH
        )
        
        # Enviar las tareas
        self.assertTrue(self.scheduler.submit_task(task1))
        self.assertTrue(self.scheduler.submit_task(task2))
        self.assertTrue(self.scheduler.submit_task(task3))
        
        # Cancelar una tarea
        self.assertTrue(self.scheduler.cancel_task("task3"))
        
        # Obtener todas las tareas
        all_tasks = self.scheduler.get_tasks()
        self.assertEqual(len(all_tasks), 3)
        
        # Obtener las tareas pendientes
        pending_tasks = self.scheduler.get_tasks(TaskStatus.PENDING)
        self.assertEqual(len(pending_tasks), 2)
        
        # Obtener las tareas canceladas
        cancelled_tasks = self.scheduler.get_tasks(TaskStatus.CANCELLED)
        self.assertEqual(len(cancelled_tasks), 1)
        self.assertEqual(cancelled_tasks[0].task_id, "task3")

class TestPerformancePredictor(unittest.TestCase):
    """Pruebas para el predictor de rendimiento."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear un directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.test_dir, "performance_history.json")
        
        # Crear un predictor de rendimiento
        self.predictor = PerformancePredictor(history_file=self.history_file)
    
    def test_record_and_predict_performance(self):
        """Prueba registrar y predecir el rendimiento de una tarea."""
        # Crear una tarea completada
        task = Task(
            task_id="test_task",
            code="import tensorflow as tf\nprint('Hello, TensorFlow!')",
            platform_preference=["colab"],
            priority=TaskPriority.MEDIUM
        )
        task.status = TaskStatus.COMPLETED
        task.platform = "colab"
        task.start_time = 1000
        task.end_time = 1100
        
        # Registrar el rendimiento
        self.predictor.record_performance(task)
        
        # Predecir el tiempo de ejecución
        predicted_time = self.predictor.predict_execution_time(task, "colab")
        self.assertEqual(predicted_time, 100)
        
        # Obtener la mejor plataforma
        best_platform = self.predictor.get_best_platform(task, ["colab", "paperspace", "runpod"])
        self.assertEqual(best_platform, "colab")

if __name__ == '__main__':
    unittest.main()
