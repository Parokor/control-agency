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
