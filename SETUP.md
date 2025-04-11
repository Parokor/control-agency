# Control Agency: Guía de Configuración Paso a Paso

Esta guía te llevará a través del proceso completo de configuración del sistema Control Agency, desde la instalación de requisitos hasta la puesta en marcha de todos los componentes.

## Índice

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalación del Sistema Base](#2-instalación-del-sistema-base)
3. [Configuración del Backend](#3-configuración-del-backend)
4. [Configuración de la Base de Datos](#4-configuración-de-la-base-de-datos)
5. [Configuración de los Contenedores](#5-configuración-de-los-contenedores)
6. [Integración y Verificación](#6-integración-y-verificación)
7. [Solución de Problemas](#7-solución-de-problemas)

## 1. Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

### Software Necesario

- **Git**: Para clonar el repositorio
- **Python 3.8+**: Para el backend
- **Node.js 16+**: Para el frontend
- **Docker**: Para los contenedores especializados

### Verificación de Requisitos

Ejecuta estos comandos para verificar que tienes todo lo necesario:

```bash
# Verificar Git
git --version
# Debería mostrar: git version 2.x.x

# Verificar Python
python --version
# Debería mostrar: Python 3.8.x o superior

# Verificar Node.js
node --version
# Debería mostrar: v16.x.x o superior

# Verificar Docker
docker --version
# Debería mostrar: Docker version 20.x.x o superior
```

## 2. Instalación del Sistema Base

### Paso 1: Clonar el Repositorio

```bash
# Clonar el repositorio
git clone https://github.com/Parokor/control-agency.git

# Entrar al directorio del proyecto
cd control-agency
```

### Paso 2: Crear la Estructura de Directorios

El sistema necesita una estructura específica de directorios. Vamos a crearla:

```bash
# Crear directorios necesarios
mkdir -p backend/app
mkdir -p backend/tests
mkdir -p frontend/src
mkdir -p frontend/public
mkdir -p containers/chat_container
mkdir -p containers/dev_container
mkdir -p containers/media_container
mkdir -p scripts
```

## 3. Configuración del Backend

El backend es el núcleo del sistema Control Agency, gestionando la orquestación de recursos y la comunicación entre componentes.

### Paso 1: Crear y Configurar el Directorio del Backend

```bash
# Crear el directorio del backend si no existe
mkdir -p backend

# Navegar al directorio del backend
cd backend

# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### Paso 2: Instalar Dependencias del Backend

```bash
# Asegúrate de estar en el directorio backend con el entorno virtual activado

# Crear archivo de requisitos
cat > requirements.txt << EOL
fastapi==0.95.2
uvicorn==0.22.0
pydantic==1.10.8
python-dotenv==1.0.0
supabase==1.0.3
requests==2.31.0
numpy==1.24.3
scikit-learn==1.2.2
EOL

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 3: Crear la Aplicación FastAPI Básica

```bash
# Crear el directorio app si no existe
mkdir -p app

# Crear el archivo principal de la aplicación
cat > app/main.py << EOL
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear la aplicación FastAPI
app = FastAPI(title="Control Agency API")

# Ruta de verificación de salud
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Iniciar el servidor con: uvicorn app.main:app --reload
EOL

# Crear archivo __init__.py para el paquete app
touch app/__init__.py
```

### Paso 4: Configurar Variables de Entorno

```bash
# Crear archivo .env para variables de entorno
cat > .env << EOL
# Configuración de Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Configuración de seguridad
JWT_SECRET=your_jwt_secret

# Configuración de API externas
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
EOL

echo "Recuerda reemplazar los valores de ejemplo en el archivo .env con tus propias claves."
```

### Paso 5: Ejecutar el Backend en Modo Desarrollo

```bash
# Asegúrate de estar en el directorio backend con el entorno virtual activado

# Iniciar el servidor en modo desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Ahora deberías poder acceder a la API en http://localhost:8080 y ver la documentación automática en http://localhost:8080/docs.

## 4. Configuración de la Base de Datos

Control Agency utiliza Supabase como base de datos, que ofrece un generoso plan gratuito.

### Paso 1: Crear una Cuenta en Supabase

1. Ve a [Supabase](https://supabase.com) y haz clic en "Start your project"
2. Regístrate con tu cuenta de GitHub o correo electrónico
3. Verifica tu dirección de correo si es necesario

### Paso 2: Crear un Nuevo Proyecto

1. Desde el dashboard de Supabase, haz clic en "New Project"
2. Selecciona una organización (crea una si es necesario)
3. Ingresa un nombre para tu proyecto (ej. "control-agency")
4. Establece una contraseña segura para la base de datos
5. Elige la región más cercana a tus usuarios
6. Haz clic en "Create new project"

### Paso 3: Obtener Credenciales de API

1. Ve al dashboard del proyecto
2. En la barra lateral izquierda, haz clic en "Project Settings"
3. Haz clic en "API" en el submenú
4. Encontrarás tu URL de API y clave anon (clave de API pública)
5. Copia estos valores para usarlos en la configuración del backend

### Paso 4: Actualizar el Archivo .env

```bash
# Navega al directorio backend
cd backend

# Edita el archivo .env con tus credenciales reales de Supabase
# Reemplaza your_supabase_url y your_supabase_key con tus valores reales
```

### Paso 5: Crear Script de Inicialización de Base de Datos

```bash
# Crear el directorio scripts si no existe
mkdir -p scripts

# Navegar al directorio scripts
cd ../../scripts

# Crear script de inicialización
cat > init_database.py << EOL
import os
import sys
import argparse
from supabase import create_client

def init_database(url, key):
    # Crear cliente de Supabase
    supabase = create_client(url, key)

    # Crear tablas necesarias
    print("Creando tablas en Supabase...")

    # Tabla de usuarios
    supabase.table("users").create({
        "id": "uuid references auth.users(id)",
        "email": "text",
        "created_at": "timestamp with time zone default now()"
    })

    # Tabla de proyectos
    supabase.table("projects").create({
        "id": "uuid default uuid_generate_v4() primary key",
        "name": "text",
        "description": "text",
        "user_id": "uuid references users(id)",
        "created_at": "timestamp with time zone default now()"
    })

    # Tabla de recursos
    supabase.table("resources").create({
        "id": "uuid default uuid_generate_v4() primary key",
        "type": "text",
        "status": "text",
        "project_id": "uuid references projects(id)",
        "created_at": "timestamp with time zone default now()"
    })

    print("Inicialización de la base de datos completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicializar base de datos de Control Agency")
    parser.add_argument("--url", required=True, help="URL de Supabase")
    parser.add_argument("--key", required=True, help="Clave API de Supabase")

    args = parser.parse_args()
    init_database(args.url, args.key)
EOL

# Ejecutar el script con tus credenciales
python init_database.py --url "TU_URL_DE_SUPABASE" --key "TU_CLAVE_DE_SUPABASE"
```

## 5. Configuración de los Contenedores

Control Agency utiliza tres contenedores especializados para diferentes funcionalidades.

### Paso 1: Configurar el Contenedor de Chat

```bash
# Crear el directorio para los contenedores si no existe
mkdir -p containers/chat_container

# Navegar al directorio del contenedor de chat
cd containers/chat_container

# Crear Dockerfile
cat > Dockerfile << EOL
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Crear archivo de requisitos
cat > requirements.txt << EOL
fastapi==0.95.2
uvicorn==0.22.0
pydantic==1.10.8
python-dotenv==1.0.0
requests==2.31.0
EOL

# Crear archivo principal
cat > main.py << EOL
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests

# Cargar variables de entorno
load_dotenv()

# Obtener claves de API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Crear la aplicación FastAPI
app = FastAPI(title="Chat Container")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "container": "chat"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Implementación básica usando Groq API
        # En una implementación real, se usaría la API de Groq o OpenRouter
        return {"response": f"Echo: {request.message}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOL
```

### Paso 2: Configurar el Contenedor de Desarrollo

```bash
# Crear el directorio para el contenedor de desarrollo si no existe
mkdir -p containers/dev_container

# Navegar al directorio del contenedor de desarrollo
cd ../../containers/dev_container

# Crear Dockerfile
cat > Dockerfile << EOL
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para repositorios
RUN mkdir -p /app/repositories

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Crear archivo de requisitos
cat > requirements.txt << EOL
fastapi==0.95.2
uvicorn==0.22.0
pydantic==1.10.8
python-dotenv==1.0.0
gitpython==3.1.31
EOL

# Crear archivo principal
cat > main.py << EOL
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import git
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear la aplicación FastAPI
app = FastAPI(title="Development Container")

class RepoRequest(BaseModel):
    url: str
    name: str = None

class RepoResponse(BaseModel):
    name: str
    path: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "container": "development"}

@app.post("/repos", response_model=RepoResponse)
async def create_repo(request: RepoRequest):
    try:
        name = request.name or request.url.split("/")[-1].replace(".git", "")
        path = f"/app/repositories/{name}"

        # Clonar repositorio
        git.Repo.clone_from(request.url, path)

        return {"name": name, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOL
```

### Paso 3: Configurar el Contenedor de Medios

```bash
# Crear el directorio para el contenedor de medios si no existe
mkdir -p containers/media_container

# Navegar al directorio del contenedor de medios
cd ../media_container

# Crear Dockerfile
cat > Dockerfile << EOL
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y python3 python3-pip git

# Clonar ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Instalar requisitos de ComfyUI
WORKDIR /app/ComfyUI
RUN pip3 install -r requirements.txt

# Crear directorio para salida
RUN mkdir -p /app/output

# Exponer puerto
EXPOSE 8188

# Comando para iniciar ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--output-directory", "/app/output"]
EOL
```

### Paso 4: Construir y Ejecutar los Contenedores

```bash
# Navegar al directorio raíz del proyecto
cd ../../

# Crear archivo docker-compose.yml
cat > docker-compose.yml << EOL
version: '3'

services:
  chat-container:
    build: ./containers/chat_container
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped

  dev-container:
    build: ./containers/dev_container
    ports:
      - "8001:8000"
    env_file:
      - .env
    volumes:
      - dev-container-data:/app/repositories
    restart: unless-stopped

  media-container:
    build: ./containers/media_container
    ports:
      - "8188:8188"
    volumes:
      - media-container-data:/app/output
    restart: unless-stopped

volumes:
  dev-container-data:
  media-container-data:
EOL

# Construir y ejecutar los contenedores
docker-compose up -d
```

## 6. Integración y Verificación

Ahora que todos los componentes están configurados, vamos a integrarlos y verificar que funcionan correctamente.

### Paso 1: Verificar el Backend

```bash
# Verificar que el backend está funcionando
curl http://localhost:8080/health
# Deberías ver: {"status":"healthy","version":"0.1.0"}
```

### Paso 2: Verificar los Contenedores

```bash
# Verificar que los contenedores están funcionando
docker ps
# Deberías ver los tres contenedores ejecutándose

# Verificar el contenedor de chat
curl http://localhost:8000/health
# Deberías ver: {"status":"healthy","container":"chat"}

# Verificar el contenedor de desarrollo
curl http://localhost:8001/health
# Deberías ver: {"status":"healthy","container":"development"}

# El contenedor de medios (ComfyUI) se puede verificar accediendo a:
# http://localhost:8188 en tu navegador
```

### Paso 3: Verificar la Base de Datos

```bash
# Navegar al directorio backend
cd backend

# Activar el entorno virtual si no está activado
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Ejecutar script de verificación de base de datos
python -c "import os; from dotenv import load_dotenv; import supabase; load_dotenv(); client = supabase.create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')); print(client.table('users').select('*').execute())"
# Deberías ver la estructura de la tabla de usuarios
```

## 7. Solución de Problemas

Si encuentras problemas durante la configuración, aquí hay algunas soluciones comunes:

### Problemas con el Backend

**Problema**: El servidor no inicia
**Solución**: Verifica que estás en el directorio correcto y que el entorno virtual está activado.

```bash
cd backend
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**Problema**: Error de módulo no encontrado
**Solución**: Instala la dependencia faltante.

```bash
pip install nombre_del_modulo
```

### Problemas con Docker

**Problema**: Los contenedores no inician
**Solución**: Verifica que Docker está instalado y en ejecución.

```bash
# Verificar el estado de Docker
systemctl status docker

# Iniciar Docker si no está en ejecución
sudo systemctl start docker
```

**Problema**: Error de permisos al ejecutar Docker
**Solución**: Añade tu usuario al grupo Docker.

```bash
sudo usermod -aG docker $USER
# Cierra sesión y vuelve a iniciar sesión para que los cambios surtan efecto
```

### Problemas con Supabase

**Problema**: Error de conexión a Supabase
**Solución**: Verifica tus credenciales y conexión a internet.

```bash
# Verifica que las credenciales en .env son correctas
cat backend/.env

# Prueba la conexión a Supabase
curl https://tu-proyecto.supabase.co/rest/v1/
```

## Conclusión

¡Felicidades! Has configurado con éxito el sistema Control Agency. Ahora puedes:

1. Usar el backend para orquestar recursos en múltiples plataformas
2. Interactuar con el sistema a través de los contenedores especializados
3. Almacenar y recuperar datos de la base de datos Supabase

Para obtener más información sobre cómo usar el sistema, consulta la [documentación principal](README.md).

Si tienes preguntas o necesitas ayuda adicional, no dudes en [crear un issue](https://github.com/Parokor/control-agency/issues) en el repositorio.
