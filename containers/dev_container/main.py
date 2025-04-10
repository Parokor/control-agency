from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import subprocess
import os
import git
import json
import shutil
from typing import List, Optional, Dict, Any

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Repository storage
REPO_DIR = "./repositories"
os.makedirs(REPO_DIR, exist_ok=True)

# Models
class RepositoryRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    setup_command: Optional[str] = None

class FileContent(BaseModel):
    content: str

class CommandRequest(BaseModel):
    command: str
    cwd: Optional[str] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dev-container"}

@app.post("/api/repositories/clone")
async def clone_repository(request: RepositoryRequest, background_tasks: BackgroundTasks):
    """Clone a GitHub repository and set up the environment"""
    repo_name = request.repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    try:
        if os.path.exists(repo_path):
            # Pull latest changes if repo exists
            repo = git.Repo(repo_path)
            repo.git.checkout(request.branch)
            repo.remotes.origin.pull()
        else:
            # Clone new repository
            os.makedirs(os.path.dirname(repo_path), exist_ok=True)
            git.Repo.clone_from(request.repo_url, repo_path, branch=request.branch)
        
        # Run setup command if provided
        if request.setup_command:
            background_tasks.add_task(run_setup_command, repo_path, request.setup_command)
        
        return {"status": "success", "repo_path": repo_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repositories")
async def list_repositories():
    """List all cloned repositories"""
    repos = []
    
    if os.path.exists(REPO_DIR):
        for repo_name in os.listdir(REPO_DIR):
            repo_path = os.path.join(REPO_DIR, repo_name)
            if os.path.isdir(repo_path) and os.path.exists(os.path.join(repo_path, ".git")):
                try:
                    repo = git.Repo(repo_path)
                    repos.append({
                        "name": repo_name,
                        "path": repo_path,
                        "active_branch": repo.active_branch.name,
                        "remotes": [{"name": remote.name, "url": remote.url} for remote in repo.remotes]
                    })
                except Exception as e:
                    repos.append({
                        "name": repo_name,
                        "path": repo_path,
                        "error": str(e)
                    })
    
    return {"repositories": repos}

@app.get("/api/repositories/{repo_name}/files")
async def list_files(repo_name: str, path: str = ""):
    """List files in a repository"""
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
    
    target_path = os.path.join(repo_path, path).rstrip("/")
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Path {path} not found in repository {repo_name}")
    
    if os.path.isfile(target_path):
        # Return file content
        try:
            with open(target_path, "r") as f:
                content = f.read()
            
            return {
                "type": "file",
                "name": os.path.basename(target_path),
                "path": path,
                "content": content
            }
        except UnicodeDecodeError:
            return {
                "type": "file",
                "name": os.path.basename(target_path),
                "path": path,
                "content": "Binary file not displayed",
                "is_binary": True
            }
    
    # List directory contents
    items = []
    
    for item in os.listdir(target_path):
        item_path = os.path.join(target_path, item)
        item_type = "directory" if os.path.isdir(item_path) else "file"
        
        items.append({
            "name": item,
            "type": item_type,
            "path": os.path.join(path, item) if path else item
        })
    
    return {
        "type": "directory",
        "name": os.path.basename(target_path) or repo_name,
        "path": path,
        "items": items
    }

@app.put("/api/repositories/{repo_name}/files")
async def update_file(repo_name: str, path: str, file_content: FileContent):
    """Update a file in a repository"""
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
    
    file_path = os.path.join(repo_path, path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, "w") as f:
            f.write(file_content.content)
        
        return {"status": "success", "message": f"File {path} updated"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/repositories/{repo_name}/files")
async def delete_file(repo_name: str, path: str):
    """Delete a file or directory in a repository"""
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
    
    target_path = os.path.join(repo_path, path)
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Path {path} not found in repository {repo_name}")
    
    try:
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)
        
        return {"status": "success", "message": f"{path} deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories/{repo_name}/commit")
async def commit_changes(repo_name: str, message: str):
    """Commit changes in a repository"""
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
    
    try:
        repo = git.Repo(repo_path)
        
        # Add all changes
        repo.git.add(A=True)
        
        # Commit
        repo.index.commit(message)
        
        return {"status": "success", "message": f"Changes committed with message: {message}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories/{repo_name}/push")
async def push_changes(repo_name: str, branch: str = None):
    """Push changes to remote repository"""
    repo_path = os.path.join(REPO_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
    
    try:
        repo = git.Repo(repo_path)
        
        # Get branch
        if branch is None:
            branch = repo.active_branch.name
        
        # Push
        origin = repo.remote(name="origin")
        origin.push(branch)
        
        return {"status": "success", "message": f"Changes pushed to {branch}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute")
async def execute_command(command_request: CommandRequest):
    """Execute a command"""
    try:
        cwd = command_request.cwd or os.getcwd()
        
        # Execute command
        process = subprocess.Popen(
            command_request.command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        return {
            "status": "success",
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_setup_command(repo_path: str, command: str):
    """Run setup command in repository directory"""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        # Log results
        with open(os.path.join(repo_path, "setup_log.txt"), "w") as f:
            f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
    
    except Exception as e:
        # Log error
        with open(os.path.join(repo_path, "setup_error.txt"), "w") as f:
            f.write(f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
