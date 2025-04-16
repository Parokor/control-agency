# Migración de Docker a Podman

Este repositorio contiene herramientas y scripts para migrar de Docker a Podman en el proyecto Control Agency. La migración es necesaria debido a que Docker ya no recibe soporte oficial en Red Hat Enterprise Linux 8 (RHEL 8) y porque Podman ofrece ventajas significativas en términos de seguridad y arquitectura.

## Contenido

- [Guía de Migración](docker_to_podman_migration_guide.md): Documentación detallada sobre las diferencias entre Docker y Podman y los pasos para la migración.
- [Scripts de Migración](#scripts-de-migración): Conjunto de scripts para automatizar el proceso de migración.
- [Pruebas](#pruebas): Scripts para verificar que la migración se ha realizado correctamente.
- [Uso](#uso): Instrucciones para utilizar las herramientas de migración.

## Scripts de Migración

### Script Principal

- `migrate_to_podman.py`: Coordina todo el proceso de migración, desde la instalación de Podman hasta la migración de imágenes, contenedores y archivos de configuración.

### Scripts Individuales

- `install_podman.sh`: Instala Podman en diferentes sistemas operativos.
- `test_podman_installation.py`: Verifica que Podman está instalado correctamente y puede realizar operaciones básicas.
- `migrate_docker_to_podman.py`: Migra imágenes y contenedores de Docker a Podman.
- `convert_dockerfile_to_containerfile.py`: Convierte Dockerfiles a Containerfiles para Podman.

## Pruebas

- `run_migration_tests.py`: Ejecuta pruebas unitarias para verificar que todos los componentes de la migración funcionan correctamente.

## Uso

### Migración Completa

Para realizar una migración completa de Docker a Podman, ejecute:

```bash
python3 migrate_to_podman.py
```

Esto realizará las siguientes acciones:
1. Verificar los requisitos previos
2. Ejecutar pruebas unitarias
3. Instalar Podman
4. Probar la instalación de Podman
5. Migrar imágenes y contenedores de Docker a Podman
6. Crear un alias de Docker a Podman

### Opciones Avanzadas

```bash
python3 migrate_to_podman.py --help
```

Opciones disponibles:
- `--output-dir`: Directorio de salida para los archivos migrados (por defecto: ./podman_migration)
- `--skip-tests`: Omitir la ejecución de pruebas unitarias
- `--skip-install`: Omitir la instalación de Podman
- `--skip-migration`: Omitir la migración de Docker a Podman
- `--dockerfiles`: Lista de Dockerfiles a convertir
- `--docker-compose`: Archivo docker-compose.yml a migrar
- `--no-images`: No migrar imágenes de Docker
- `--no-containers`: No migrar contenedores de Docker
- `--no-alias`: No crear alias de Docker a Podman

### Ejemplos

#### Migrar solo imágenes y contenedores

```bash
python3 migrate_to_podman.py --skip-tests --skip-install
```

#### Convertir Dockerfiles específicos

```bash
python3 migrate_to_podman.py --skip-tests --skip-install --skip-migration --dockerfiles Dockerfile Dockerfile.dev
```

#### Migrar un archivo docker-compose.yml

```bash
python3 migrate_to_podman.py --skip-tests --skip-install --no-images --no-containers --docker-compose docker-compose.yml
```

## Diferencias Clave entre Docker y Podman

| Característica | Docker | Podman |
|----------------|--------|--------|
| Arquitectura | Cliente-servidor con daemon | Sin daemon (fork-exec) |
| Seguridad | Requiere privilegios de root | No requiere privilegios de root |
| Soporte de Pods | No (requiere Docker Compose) | Sí (nativo) |
| Compatibilidad | Estándar de facto | Compatible con Docker |
| Rendimiento | Más lento al iniciar | Más rápido al iniciar |

Para más detalles, consulte la [Guía de Migración](docker_to_podman_migration_guide.md).

## Requisitos

- Python 3.6 o superior
- pip
- Docker (opcional, solo para migrar imágenes y contenedores existentes)

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.
