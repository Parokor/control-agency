#!/bin/bash

# Script para instalar Podman en diferentes sistemas operativos
# Este script detecta automáticamente el sistema operativo y utiliza
# el gestor de paquetes apropiado para instalar Podman.

# Colores para la salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes de error y salir
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Función para imprimir mensajes informativos
info() {
    echo -e "${BLUE}Info: $1${NC}"
}

# Función para imprimir mensajes de éxito
success() {
    echo -e "${GREEN}Success: $1${NC}"
}

# Función para imprimir mensajes de advertencia
warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Función para verificar si un comando está disponible
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Función para verificar si Podman ya está instalado
check_podman_installed() {
    if command_exists podman; then
        PODMAN_VERSION=$(podman --version | awk '{print $3}')
        success "Podman ya está instalado (versión $PODMAN_VERSION)"
        return 0
    else
        return 1
    fi
}

# Función para instalar Podman en sistemas basados en Debian/Ubuntu
install_podman_debian() {
    info "Instalando Podman en sistema basado en Debian/Ubuntu..."
    
    # Actualizar repositorios
    sudo apt-get update || error_exit "No se pudo actualizar los repositorios"
    
    # Instalar dependencias
    sudo apt-get install -y curl gnupg2 software-properties-common || error_exit "No se pudieron instalar las dependencias"
    
    # Añadir repositorio de Podman (para versiones más recientes)
    source /etc/os-release
    if [[ "$VERSION_ID" == "20.04" || "$VERSION_ID" == "22.04" ]]; then
        info "Añadiendo repositorio de Podman para Ubuntu $VERSION_ID..."
        . /etc/os-release
        echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
        curl -L "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key" | sudo apt-key add -
        sudo apt-get update
    fi
    
    # Instalar Podman
    sudo apt-get install -y podman || error_exit "No se pudo instalar Podman"
    
    success "Podman instalado correctamente"
}

# Función para instalar Podman en sistemas basados en Red Hat (RHEL, CentOS, Fedora)
install_podman_redhat() {
    info "Instalando Podman en sistema basado en Red Hat..."
    
    # Verificar si dnf o yum está disponible
    if command_exists dnf; then
        PKG_MGR="dnf"
    elif command_exists yum; then
        PKG_MGR="yum"
    else
        error_exit "No se encontró un gestor de paquetes compatible (dnf o yum)"
    fi
    
    # Instalar Podman
    sudo $PKG_MGR install -y podman || error_exit "No se pudo instalar Podman"
    
    success "Podman instalado correctamente"
}

# Función para instalar Podman en sistemas basados en Arch Linux
install_podman_arch() {
    info "Instalando Podman en sistema basado en Arch Linux..."
    
    # Actualizar repositorios
    sudo pacman -Sy || error_exit "No se pudo actualizar los repositorios"
    
    # Instalar Podman
    sudo pacman -S --noconfirm podman || error_exit "No se pudo instalar Podman"
    
    success "Podman instalado correctamente"
}

# Función para instalar Podman en macOS usando Homebrew
install_podman_macos() {
    info "Instalando Podman en macOS..."
    
    # Verificar si Homebrew está instalado
    if ! command_exists brew; then
        warning "Homebrew no está instalado. Instalando Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || error_exit "No se pudo instalar Homebrew"
    fi
    
    # Instalar Podman
    brew install podman || error_exit "No se pudo instalar Podman"
    
    # Inicializar la máquina virtual de Podman
    podman machine init || warning "No se pudo inicializar la máquina virtual de Podman"
    podman machine start || warning "No se pudo iniciar la máquina virtual de Podman"
    
    success "Podman instalado correctamente"
}

# Función para configurar Podman después de la instalación
configure_podman() {
    info "Configurando Podman..."
    
    # Verificar si Podman está instalado
    if ! command_exists podman; then
        error_exit "Podman no está instalado"
    fi
    
    # Crear directorio de configuración si no existe
    PODMAN_CONF_DIR="$HOME/.config/containers"
    mkdir -p "$PODMAN_CONF_DIR"
    
    # Configurar registros
    cat > "$PODMAN_CONF_DIR/registries.conf" << EOL
[registries.search]
registries = ['docker.io', 'quay.io', 'registry.fedoraproject.org', 'registry.access.redhat.com']

[registries.insecure]
registries = []

[registries.block]
registries = []
EOL
    
    # Configurar política de firmas
    cat > "$PODMAN_CONF_DIR/policy.json" << EOL
{
    "default": [
        {
            "type": "insecureAcceptAnything"
        }
    ],
    "transports": {
        "docker": {}
    }
}
EOL
    
    success "Podman configurado correctamente"
}

# Función para crear un alias de Docker a Podman
create_docker_alias() {
    info "Creando alias de Docker a Podman..."
    
    # Verificar si el alias ya existe
    if grep -q "alias docker=podman" "$HOME/.bashrc" 2>/dev/null || grep -q "alias docker=podman" "$HOME/.zshrc" 2>/dev/null; then
        warning "El alias 'docker=podman' ya existe"
    else
        # Determinar el shell actual
        if [[ "$SHELL" == *"zsh"* ]]; then
            echo "alias docker=podman" >> "$HOME/.zshrc"
            source "$HOME/.zshrc" 2>/dev/null || true
        else
            echo "alias docker=podman" >> "$HOME/.bashrc"
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
        success "Alias 'docker=podman' creado correctamente"
    fi
}

# Función principal
main() {
    echo "=== Instalación de Podman ==="
    
    # Verificar si Podman ya está instalado
    if check_podman_installed; then
        read -p "Podman ya está instalado. ¿Desea continuar con la configuración? (s/n): " CONTINUE
        if [[ "$CONTINUE" != "s" && "$CONTINUE" != "S" ]]; then
            info "Instalación cancelada"
            exit 0
        fi
    fi
    
    # Detectar el sistema operativo
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if [[ -f /etc/debian_version ]]; then
            # Debian/Ubuntu
            install_podman_debian
        elif [[ -f /etc/redhat-release ]]; then
            # Red Hat/CentOS/Fedora
            install_podman_redhat
        elif [[ -f /etc/arch-release ]]; then
            # Arch Linux
            install_podman_arch
        else
            error_exit "Sistema operativo Linux no soportado"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        install_podman_macos
    else
        error_exit "Sistema operativo no soportado"
    fi
    
    # Configurar Podman
    configure_podman
    
    # Preguntar si se desea crear un alias de Docker a Podman
    read -p "¿Desea crear un alias de 'docker' a 'podman'? (s/n): " CREATE_ALIAS
    if [[ "$CREATE_ALIAS" == "s" || "$CREATE_ALIAS" == "S" ]]; then
        create_docker_alias
    fi
    
    # Verificar la instalación
    podman --version || error_exit "No se pudo verificar la instalación de Podman"
    
    success "Instalación y configuración de Podman completada"
    info "Para probar la instalación, ejecute: python3 test_podman_installation.py"
}

# Ejecutar la función principal
main
