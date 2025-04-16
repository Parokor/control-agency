# Integración con Plataformas en la Nube

Este módulo proporciona una arquitectura basada en la nube con tokens de autenticación para vincular GitHub con diferentes plataformas en la nube como Google Colab, Paperspace Gradient y RunPod.

## Características

- **Autenticación con Tokens**: Gestión segura de tokens de autenticación para diferentes plataformas.
- **Integración con GitHub**: Autenticación con GitHub para acceder a repositorios, crear archivos, etc.
- **Integración con Plataformas en la Nube**: Soporte para Google Colab, Paperspace Gradient y RunPod.
- **Programación de Recursos**: Asignación inteligente de tareas a diferentes plataformas en la nube.
- **Predicción de Rendimiento**: Predicción del tiempo de ejecución de tareas en diferentes plataformas.
- **Mecanismos Anti-Timeout**: Implementación de mecanismos para prevenir timeouts en plataformas gratuitas.

## Estructura del Módulo

```
.
├── auth/                       # Módulo de autenticación
│   ├── __init__.py             # Inicialización del módulo
│   ├── github_auth.py          # Autenticación con GitHub
│   ├── cloud_auth.py           # Autenticación con plataformas en la nube
│   └── token_manager.py        # Gestión de tokens de autenticación
├── cloud/                      # Módulo de gestión de plataformas en la nube
│   ├── __init__.py             # Inicialización del módulo
│   ├── platform_manager.py     # Gestión de plataformas en la nube
│   └── resource_scheduler.py   # Programación de recursos en la nube
├── tests/                      # Pruebas unitarias
│   ├── __init__.py             # Inicialización del módulo
│   └── test_cloud_integration.py # Pruebas para la integración con plataformas en la nube
├── cloud_manager.py            # Script principal para la gestión de recursos en la nube
├── run_tests.py                # Script para ejecutar las pruebas unitarias
└── requirements.txt            # Requisitos de Python
```

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Parokor/control-agency.git
   cd control-agency
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Gestión de Tokens

```bash
# Guardar un token
python cloud_manager.py token --save-token --platform github --token YOUR_TOKEN

# Listar tokens guardados
python cloud_manager.py token --list-tokens

# Verificar un token
python cloud_manager.py token --verify-token --platform github

# Eliminar un token
python cloud_manager.py token --delete-token --platform github
```

### Gestión de Plataformas

```bash
# Probar la conexión con una plataforma
python cloud_manager.py platform --test-platform colab

# Listar instancias disponibles
python cloud_manager.py platform --list-instances --platform colab

# Crear una instancia
python cloud_manager.py platform --create-instance --platform colab

# Detener una instancia
python cloud_manager.py platform --stop-instance --platform colab --instance-id INSTANCE_ID

# Obtener el estado de una instancia
python cloud_manager.py platform --instance-status --platform colab --instance-id INSTANCE_ID

# Implementar mecanismos anti-timeout
python cloud_manager.py platform --anti-timeout --platform colab --instance-id INSTANCE_ID
```

### Gestión de Tareas

```bash
# Iniciar el programador de recursos
python cloud_manager.py task --start-scheduler

# Obtener el estado del programador de recursos
python cloud_manager.py task --scheduler-status

# Enviar una tarea
python cloud_manager.py task --submit-task path/to/code.py --task-id task1 --priority HIGH --platforms colab,paperspace

# Obtener una tarea
python cloud_manager.py task --get-task task1

# Listar todas las tareas
python cloud_manager.py task --list-tasks

# Listar tareas con un estado específico
python cloud_manager.py task --list-tasks --task-status PENDING

# Cancelar una tarea
python cloud_manager.py task --cancel-task task1

# Detener el programador de recursos
python cloud_manager.py task --stop-scheduler
```

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
# Ejecutar todas las pruebas
python run_tests.py

# Ejecutar un módulo de pruebas específico
python run_tests.py --module tests.test_cloud_integration

# Mostrar información detallada
python run_tests.py --verbose
```

## Mecanismos Anti-Timeout

El módulo implementa mecanismos anti-timeout para prevenir la desconexión en plataformas gratuitas como Google Colab. Estos mecanismos incluyen:

- **Actividad de Cómputo**: Generación de actividad de cómputo aleatoria.
- **Actividad de Memoria**: Generación de actividad de memoria aleatoria.
- **Actividad de Disco**: Generación de actividad de disco aleatoria.
- **Actividad de Pantalla**: Generación de actividad de pantalla aleatoria (solo para Google Colab).

## Predicción de Rendimiento

El módulo incluye un predictor de rendimiento que registra el tiempo de ejecución de las tareas en diferentes plataformas y predice el tiempo de ejecución de nuevas tareas. Esto permite asignar las tareas a las plataformas más adecuadas.

## Contribuir

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Haz push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
