from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import json
import os
import asyncio
from typing import List, Dict, Any

app = FastAPI()

# Groq API integration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# OpenRouter integration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    def add_message(self, client_id: str, message: Dict[str, Any]):
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        self.conversation_history[client_id].append(message)
    
    def get_history(self, client_id: str) -> List[Dict[str, Any]]:
        return self.conversation_history.get(client_id, [])

manager = ConnectionManager()

# Root endpoint
@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat Container - Dolphin 3.0 R1</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                #messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
                #input { width: 100%; padding: 10px; box-sizing: border-box; }
                .user-message { background-color: #e6f7ff; padding: 8px; border-radius: 5px; margin-bottom: 10px; }
                .bot-message { background-color: #f0f0f0; padding: 8px; border-radius: 5px; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>Chat Container - Dolphin 3.0 R1</h1>
            <div id="messages"></div>
            <input id="input" type="text" placeholder="Type your message here...">
            
            <script>
                const clientId = Date.now().toString();
                const ws = new WebSocket(`ws://${window.location.host}/ws/chat/${clientId}`);
                const messages = document.getElementById('messages');
                const input = document.getElementById('input');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    const message = data.choices[0].message.content;
                    const div = document.createElement('div');
                    div.className = 'bot-message';
                    div.textContent = message;
                    messages.appendChild(div);
                    messages.scrollTop = messages.scrollHeight;
                };
                
                input.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        const message = input.value;
                        if (message) {
                            const div = document.createElement('div');
                            div.className = 'user-message';
                            div.textContent = message;
                            messages.appendChild(div);
                            
                            ws.send(JSON.stringify({
                                messages: [
                                    { role: "user", content: message }
                                ]
                            }));
                            
                            input.value = '';
                            messages.scrollTop = messages.scrollHeight;
                        }
                    }
                });
            </script>
        </body>
    </html>
    """)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chat-container"}

# Chat endpoint
@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Add conversation history
            history = manager.get_history(client_id)
            if history:
                message_data["messages"] = history + message_data["messages"]
            
            # Process with Dolphin 3.0 R1
            response = await process_with_llm(message_data)
            
            # Store messages in history
            for msg in message_data["messages"]:
                manager.add_message(client_id, msg)
            
            # Store assistant response
            if "choices" in response and response["choices"]:
                manager.add_message(client_id, response["choices"][0]["message"])
            
            await websocket.send_text(json.dumps(response))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client {client_id} disconnected")

async def process_with_llm(message_data):
    """Process message with Dolphin 3.0 R1 via Groq Cloud LPU"""
    if not GROQ_API_KEY:
        return await process_with_openrouter(message_data)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "dolphin-3.0-r1",
        "messages": message_data["messages"],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to OpenRouter
                return await process_with_openrouter(message_data)
    except Exception as e:
        print(f"Error with Groq API: {e}")
        # Fallback to OpenRouter
        return await process_with_openrouter(message_data)

async def process_with_openrouter(message_data):
    """Fallback to OpenRouter if Groq is unavailable"""
    if not OPENROUTER_API_KEY:
        return {"choices": [{"message": {"role": "assistant", "content": "API keys not configured. Please set up GROQ_API_KEY or OPENROUTER_API_KEY."}}]}
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "dolphin-3.0-r1",
        "messages": message_data["messages"],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant", 
                            "content": f"Error: {response.status_code} - {response.text}"
                        }
                    }]
                }
    except Exception as e:
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"Error: {str(e)}"
                }
            }]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
