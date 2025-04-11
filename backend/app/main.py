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
