# Guía de Migración de Docker a Podman

## Introducción

Esta guía proporciona información detallada sobre cómo migrar de Docker a Podman en el proyecto Control Agency. La migración es necesaria debido a que Docker ya no recibe soporte oficial en Red Hat Enterprise Linux 8 (RHEL 8) y porque Podman ofrece ventajas significativas en términos de seguridad y arquitectura.

## Diferencias Clave entre Docker y Podman

### 1. Arquitectura

**Docker:**
- Utiliza un modelo cliente-servidor
- Requiere un daemon central (dockerd) que se ejecuta en segundo plano
- El cliente Docker se comunica con el daemon a través de una API REST

**Podman:**
- Utiliza un modelo fork-exec (sin daemon)
- Cada contenedor se ejecuta como un proceso hijo de Podman
- No requiere un proceso en segundo plano

### 2. Seguridad

**Docker:**
- El daemon Docker requiere privilegios de root
- Los contenedores pueden potencialmente acceder al kernel con privilegios de root
- Representa un riesgo de seguridad si un contenedor está mal configurado

**Podman:**
- No requiere privilegios de root para ejecutar contenedores
- Utiliza espacios de nombres de usuario para aislar los contenedores
- Cada contenedor solo tiene los derechos del usuario que lo ejecuta
- Registra los cambios en el sistema auditd

### 3. Soporte de Pods

**Docker:**
- No tiene soporte nativo para pods
- Requiere Docker Compose para gestionar múltiples contenedores juntos

**Podman:**
- Soporta nativamente el concepto de pods de Kubernetes
- Permite agrupar varios contenedores en un espacio de nombres común
- Facilita la transición a Kubernetes

### 4. Compatibilidad

**Docker:**
- Estándar de facto en la industria
- Amplia documentación y comunidad
- Compatible con la mayoría de herramientas y plataformas

**Podman:**
- Compatible con imágenes y comandos de Docker
- Puede utilizar registros de contenedores como Docker Hub
- Puede generar archivos YAML para Kubernetes

### 5. Rendimiento

**Podman:**
- Generalmente más rápido al iniciar contenedores (no requiere daemon)
- Requiere menos espacio de almacenamiento
- Más eficiente en términos de recursos

## Pasos para la Migración

### 1. Instalación de Podman

```bash
# En sistemas basados en Red Hat (RHEL, CentOS, Fedora)
sudo dnf install podman

# En Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y podman

# En macOS (usando Homebrew)
brew install podman
```

### 2. Verificación de la Instalación

```bash
podman --version
podman info
```

### 3. Migración de Imágenes

```bash
# Listar imágenes Docker existentes
docker images

# Guardar imágenes Docker como archivos tar
docker save -o <nombre_imagen>.tar <nombre_imagen>:<tag>

# Cargar imágenes en Podman
podman load -i <nombre_imagen>.tar

# Verificar que las imágenes se han cargado correctamente
podman images
```

### 4. Actualización de Scripts y Configuraciones

#### Cambio de Nombres de Archivos

- Renombrar `Dockerfile` a `Containerfile` (aunque Podman también reconoce `Dockerfile`)
- Renombrar `docker-compose.yml` a `podman-compose.yml` (si se utiliza podman-compose)

#### Actualización de Comandos

La mayoría de los comandos de Docker tienen equivalentes directos en Podman:

| Docker | Podman |
|--------|--------|
| `docker build` | `podman build` |
| `docker run` | `podman run` |
| `docker ps` | `podman ps` |
| `docker images` | `podman images` |
| `docker pull` | `podman pull` |
| `docker push` | `podman push` |
| `docker exec` | `podman exec` |
| `docker rm` | `podman rm` |
| `docker rmi` | `podman rmi` |

Para facilitar la transición, se puede crear un alias:

```bash
alias docker=podman
```

### 5. Migración de Docker Compose a Podman

Podman ofrece varias opciones para reemplazar Docker Compose:

1. **Podman Compose**: Una implementación de Docker Compose para Podman
   ```bash
   pip install podman-compose
   podman-compose up -d
   ```

2. **Podman Pods**: Crear pods para agrupar contenedores relacionados
   ```bash
   podman pod create --name mi-aplicacion
   podman run --pod mi-aplicacion -d imagen1
   podman run --pod mi-aplicacion -d imagen2
   ```

3. **Generar archivos Kubernetes**: Convertir configuraciones de Docker Compose a Kubernetes
   ```bash
   podman generate kube mi-contenedor > mi-contenedor.yaml
   ```

### 6. Integración con Systemd

Podman se integra bien con systemd, lo que permite gestionar contenedores como servicios del sistema:

```bash
# Generar un archivo de servicio systemd para un contenedor
podman generate systemd --name mi-contenedor > mi-contenedor.service

# Instalar el servicio
mv mi-contenedor.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now mi-contenedor
```

## Consideraciones Importantes

### Almacenamiento

Podman almacena imágenes y contenedores en ubicaciones diferentes a Docker:

- **Docker**: `/var/lib/docker/`
- **Podman**: 
  - Root: `/var/lib/containers/`
  - Non-root: `$HOME/.local/share/containers/`

### Redes

Podman respeta las reglas del cortafuegos existentes, mientras que Docker las sobrescribe. Esto puede requerir ajustes en la configuración de red.

### Volúmenes

La gestión de volúmenes en Podman es similar a Docker, pero con algunas diferencias:

```bash
# Crear un volumen
podman volume create mi-volumen

# Usar un volumen
podman run -v mi-volumen:/ruta/en/contenedor imagen
```

## Pruebas y Verificación

Después de migrar a Podman, es importante realizar pruebas exhaustivas:

1. Verificar que todos los contenedores se inician correctamente
2. Comprobar la conectividad entre contenedores
3. Validar el acceso a volúmenes y datos persistentes
4. Probar los scripts y automatizaciones actualizados
5. Verificar la integración con otras herramientas y sistemas

## Solución de Problemas Comunes

### Error: No se puede conectar a un host remoto

Podman no tiene un daemon en red, lo que dificulta la gestión de contenedores en hosts remotos. Soluciones:

1. Usar SSH para conectarse al host remoto
2. Configurar el servicio Podman API (podman system service)

### Error: Problemas de permisos

Si encuentra problemas de permisos al ejecutar contenedores sin privilegios:

1. Verificar la configuración de espacios de nombres de usuario
2. Comprobar los permisos de los archivos y directorios montados
3. Utilizar la opción `--userns=keep-id` para mantener el ID de usuario

## Conclusión

La migración de Docker a Podman ofrece beneficios significativos en términos de seguridad y arquitectura. Aunque puede haber algunos desafíos durante la transición, la compatibilidad entre ambas herramientas facilita el proceso. Con una planificación adecuada y siguiendo los pasos de esta guía, la migración puede realizarse de manera eficiente y con un impacto mínimo en las operaciones existentes.

## Referencias

- [Documentación oficial de Podman](https://podman.io/docs)
- [Guía de migración de Docker a Podman (Red Hat)](https://www.redhat.com/sysadmin/replace-docker-podman-macos)
- [Podman vs. Docker: Diferencias clave (IONOS)](https://www.ionos.com/es-us/digitalguide/servidores/know-how/podman-vs-docker/)
