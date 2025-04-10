import os
import sys
import asyncio
import json
from typing import Dict, Any, List, Optional

class OpenManusIntegration:
    """Integration with OpenManus AI Agent"""
    
    def __init__(self, config_path: str = "config/openmanus_config.toml"):
        self.config_path = config_path
        self.openmanus_path = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize OpenManus integration"""
        # Check if OpenManus is installed
        if os.path.exists("OpenManus"):
            self.openmanus_path = os.path.abspath("OpenManus")
        else:
            # Clone OpenManus repository
            print("Cloning OpenManus repository...")
            os.system("git clone https://github.com/mannaandpoem/OpenManus.git")
            self.openmanus_path = os.path.abspath("OpenManus")
        
        # Create virtual environment if it doesn't exist
        if not os.path.exists(os.path.join(self.openmanus_path, ".venv")):
            print("Creating virtual environment...")
            os.system(f"cd {self.openmanus_path} && python -m venv .venv")
        
        # Install dependencies
        print("Installing OpenManus dependencies...")
        if sys.platform == "win32":
            os.system(f"cd {self.openmanus_path} && .venv\\Scripts\\pip install -r requirements.txt")
        else:
            os.system(f"cd {self.openmanus_path} && .venv/bin/pip install -r requirements.txt")
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join(self.openmanus_path, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create config file if it doesn't exist
        if not os.path.exists(os.path.join(config_dir, "config.toml")):
            self._create_default_config(os.path.join(config_dir, "config.toml"))
        
        self.initialized = True
        return True
    
    def _create_default_config(self, config_path: str):
        """Create default OpenManus configuration"""
        config = """
# Global LLM configuration
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = ""  # Will be set from environment variable
max_tokens = 4096
temperature = 0.0

# Vision model configuration
[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = ""  # Will be set from environment variable

# Agent configuration
[agent]
max_iterations = 10
verbose = true

# Tool configurations
[tools]
enable_browser = true
enable_file_operations = true
enable_code_execution = true
"""
        with open(config_path, "w") as f:
            f.write(config)
    
    async def execute_task(self, task: str, resources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using OpenManus"""
        if not self.initialized:
            await self.initialize()
        
        # Create task file
        task_file = os.path.join(self.openmanus_path, "task.json")
        with open(task_file, "w") as f:
            json.dump({
                "task": task,
                "resources": resources or {}
            }, f)
        
        # Execute task
        if sys.platform == "win32":
            cmd = f"cd {self.openmanus_path} && .venv\\Scripts\\python run_task.py --task-file task.json --output-file result.json"
        else:
            cmd = f"cd {self.openmanus_path} && .venv/bin/python run_task.py --task-file task.json --output-file result.json"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }
        
        # Read result file
        result_file = os.path.join(self.openmanus_path, "result.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            return {
                "status": "success",
                "result": result
            }
        else:
            return {
                "status": "error",
                "error": "Result file not found",
                "output": stdout.decode()
            }
    
    async def run_mcp(self, task: str) -> Dict[str, Any]:
        """Run Multi-Context Processor for complex reasoning"""
        if not self.initialized:
            await self.initialize()
        
        # Create task file
        task_file = os.path.join(self.openmanus_path, "mcp_task.json")
        with open(task_file, "w") as f:
            json.dump({
                "task": task
            }, f)
        
        # Execute MCP
        if sys.platform == "win32":
            cmd = f"cd {self.openmanus_path} && .venv\\Scripts\\python run_mcp.py --task-file mcp_task.json --output-file mcp_result.json"
        else:
            cmd = f"cd {self.openmanus_path} && .venv/bin/python run_mcp.py --task-file mcp_task.json --output-file mcp_result.json"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }
        
        # Read result file
        result_file = os.path.join(self.openmanus_path, "mcp_result.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            return {
                "status": "success",
                "result": result
            }
        else:
            return {
                "status": "error",
                "error": "Result file not found",
                "output": stdout.decode()
            }
    
    async def run_flow(self, task: str, agents: List[str]) -> Dict[str, Any]:
        """Run multi-agent flow for parallel processing"""
        if not self.initialized:
            await self.initialize()
        
        # Create task file
        task_file = os.path.join(self.openmanus_path, "flow_task.json")
        with open(task_file, "w") as f:
            json.dump({
                "task": task,
                "agents": agents
            }, f)
        
        # Execute flow
        if sys.platform == "win32":
            cmd = f"cd {self.openmanus_path} && .venv\\Scripts\\python run_flow.py --task-file flow_task.json --output-file flow_result.json"
        else:
            cmd = f"cd {self.openmanus_path} && .venv/bin/python run_flow.py --task-file flow_task.json --output-file flow_result.json"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }
        
        # Read result file
        result_file = os.path.join(self.openmanus_path, "flow_result.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            return {
                "status": "success",
                "result": result
            }
        else:
            return {
                "status": "error",
                "error": "Result file not found",
                "output": stdout.decode()
            }

# Example usage
async def main():
    openmanus = OpenManusIntegration()
    await openmanus.initialize()
    
    result = await openmanus.execute_task("Analyze the performance of different GPU types for AI training")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
