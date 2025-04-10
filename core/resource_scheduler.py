import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import uuid

class ResourceType(Enum):
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"
    QUANTUM = "quantum"

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceScheduler:
    def __init__(self):
        self.resources = {}  # Information about available resources
        self.tasks = {}      # Tasks in the system
        self.task_queue = [] # Queue of pending tasks
        self.running_tasks = {}  # Tasks currently running
        self.resource_stats = {}  # Resource usage statistics
        self.lock = asyncio.Lock()
        self.running = False
    
    async def register_resource(self, resource_id: str, platform: str, 
                              resource_type: ResourceType, capabilities: Dict[str, Any]) -> bool:
        """Registers a new resource in the system"""
        async with self.lock:
            if resource_id in self.resources:
                # Update existing resource
                self.resources[resource_id].update({
                    "platform": platform,
                    "type": resource_type,
                    "capabilities": capabilities,
                    "last_updated": time.time()
                })
            else:
                # Create new resource
                self.resources[resource_id] = {
                    "id": resource_id,
                    "platform": platform,
                    "type": resource_type,
                    "capabilities": capabilities,
                    "status": "available",
                    "created_at": time.time(),
                    "last_updated": time.time(),
                    "usage_stats": {
                        "total_tasks": 0,
                        "total_runtime": 0,
                        "successful_tasks": 0,
                        "failed_tasks": 0
                    }
                }
            
            # Initialize statistics if they don't exist
            if resource_id not in self.resource_stats:
                self.resource_stats[resource_id] = {
                    "hourly": [0] * 24,  # Usage by hour of day
                    "daily": [0] * 7,    # Usage by day of week
                    "availability": [],  # Availability history
                    "performance": []    # Performance history
                }
            
            return True
    
    async def update_resource_status(self, resource_id: str, status: str, 
                                   metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Updates status and metrics of a resource"""
        async with self.lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources[resource_id]
            old_status = resource["status"]
            resource["status"] = status
            resource["last_updated"] = time.time()
            
            if metrics:
                if "metrics" not in resource:
                    resource["metrics"] = {}
                resource["metrics"].update(metrics)
            
            # Record availability change for analysis
            if old_status != status:
                now = datetime.now()
                self.resource_stats[resource_id]["availability"].append({
                    "timestamp": time.time(),
                    "old_status": old_status,
                    "new_status": status,
                    "hour": now.hour,
                    "day": now.weekday()
                })
                
                # Keep history manageable
                if len(self.resource_stats[resource_id]["availability"]) > 1000:
                    self.resource_stats[resource_id]["availability"] = \
                        self.resource_stats[resource_id]["availability"][-1000:]
            
            return True
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], 
                       requirements: Optional[Dict[str, Any]] = None, 
                       priority: int = 0) -> str:
        """Submits a new task to the system"""
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "requirements": requirements or {},
            "priority": priority,
            "status": TaskStatus.PENDING.value,
            "created_at": time.time(),
            "updated_at": time.time(),
            "attempts": 0,
            "max_attempts": 3,
            "assigned_resource": None,
            "results": None,
            "error": None
        }
        
        async with self.lock:
            self.tasks[task_id] = task
            
            # Insert in queue by priority
            # Higher priority at the beginning
            index = 0
            for i, queued_task_id in enumerate(self.task_queue):
                queued_task = self.tasks[queued_task_id]
                if queued_task["priority"] < priority:
                    break
                index = i + 1
            
            self.task_queue.insert(index, task_id)
            self.tasks[task_id]["status"] = TaskStatus.QUEUED.value
        
        # Start scheduler if not running
        if not self.running:
            asyncio.create_task(self._scheduler_loop())
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Gets status of a task"""
        async with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            return {
                "id": task["id"],
                "status": task["status"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"],
                "attempts": task["attempts"],
                "assigned_resource": task["assigned_resource"],
                "results": task["results"] if task["status"] == TaskStatus.COMPLETED.value else None,
                "error": task["error"] if task["status"] == TaskStatus.FAILED.value else None
            }
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        self.running = True
        
        while self.running:
            try:
                await self._process_queue()
                await asyncio.sleep(1)  # Avoid excessive CPU usage
            except Exception as e:
                print(f"Error in scheduler: {e}")
                await asyncio.sleep(5)  # Wait longer on error
        
        print("Scheduler stopped")
    
    async def _process_queue(self):
        """Processes tasks in the queue"""
        async with self.lock:
            # Nothing to process
            if not self.task_queue:
                return
            
            # Get available resources
            available_resources = []
            for resource_id, resource in self.resources.items():
                if resource["status"] == "available":
                    available_resources.append(resource)
            
            # No available resources
            if not available_resources:
                return
            
            # Try to assign tasks to resources
            tasks_to_remove = []
            
            for i, task_id in enumerate(self.task_queue):
                task = self.tasks[task_id]
                
                # Find suitable resource for task
                selected_resource = await self._select_resource_for_task(task, available_resources)
                
                if selected_resource:
                    # Mark task as running
                    task["status"] = TaskStatus.RUNNING.value
                    task["updated_at"] = time.time()
                    task["assigned_resource"] = selected_resource["id"]
                    task["attempts"] += 1
                    
                    # Mark resource as busy
                    selected_resource["status"] = "busy"
                    
                    # Move task to running
                    self.running_tasks[task_id] = task
                    tasks_to_remove.append(task_id)
                    
                    # Update statistics
                    selected_resource["usage_stats"]["total_tasks"] += 1
                    
                    # Start execution in background
                    asyncio.create_task(self._execute_task(task_id, selected_resource["id"]))
                    
                    # Remove resource from available
                    available_resources.remove(selected_resource)
                    
                    # Exit if no more resources
                    if not available_resources:
                        break
            
            # Remove assigned tasks from queue
            for task_id in tasks_to_remove:
                self.task_queue.remove(task_id)
    
    async def _select_resource_for_task(self, task: Dict[str, Any], 
                                      available_resources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Selects the most suitable resource for a task"""
        requirements = task["requirements"]
        
        # Filter by resource type
        resource_type = requirements.get("resource_type")
        if resource_type:
            resources = [r for r in available_resources if r["type"] == resource_type]
        else:
            resources = available_resources
        
        if not resources:
            return None
        
        # Filter by memory requirements
        min_memory = requirements.get("min_memory_gb")
        if min_memory:
            resources = [r for r in resources if r["capabilities"].get("memory_gb", 0) >= min_memory]
        
        if not resources:
            return None
        
        # Filter by preferred platform
        preferred_platform = requirements.get("preferred_platform")
        if preferred_platform:
            preferred_resources = [r for r in resources if r["platform"] == preferred_platform]
            if preferred_resources:
                resources = preferred_resources
        
        if not resources:
            return None
        
        # Return first resource for now
        # In a real implementation, use more sophisticated selection algorithms
        return resources[0]
    
    async def _execute_task(self, task_id: str, resource_id: str):
        """Executes a task on a specific resource"""
        try:
            # Here would be the actual execution logic using platform adapters
            task = self.tasks[task_id]
            resource = self.resources[resource_id]
            
            start_time = time.time()
            
            # Simulation of execution
            print(f"Executing task {task_id} on resource {resource_id}")
            await asyncio.sleep(3)  # Simulate execution time
            
            # In a real implementation, invoke the platform adapter:
            # result = await self.platform_adapters[resource["platform"]].execute_task(task, resource)
            
            # Simulate successful result
            result = {"status": "success", "data": "Simulated result"}
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            async with self.lock:
                # Update task
                task["status"] = TaskStatus.COMPLETED.value
                task["updated_at"] = time.time()
                task["results"] = result
                task["execution_time"] = execution_time
                
                # Update resource statistics
                resource["status"] = "available"
                resource["usage_stats"]["total_runtime"] += execution_time
                resource["usage_stats"]["successful_tasks"] += 1
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                # Record performance
                now = datetime.now()
                self.resource_stats[resource_id]["performance"].append({
                    "timestamp": time.time(),
                    "task_type": task["type"],
                    "execution_time": execution_time,
                    "success": True,
                    "hour": now.hour,
                    "day": now.weekday()
                })
                
                # Update hourly and daily statistics
                self.resource_stats[resource_id]["hourly"][now.hour] += 1
                self.resource_stats[resource_id]["daily"][now.weekday()] += 1
                
                # Keep history manageable
                if len(self.resource_stats[resource_id]["performance"]) > 1000:
                    self.resource_stats[resource_id]["performance"] = \
                        self.resource_stats[resource_id]["performance"][-1000:]
        
        except Exception as e:
            # Handle execution error
            async with self.lock:
                task = self.tasks.get(task_id)
                resource = self.resources.get(resource_id)
                
                if task:
                    task["status"] = TaskStatus.FAILED.value
                    task["updated_at"] = time.time()
                    task["error"] = str(e)
                    
                    # Requeue if attempts remain
                    if task["attempts"] < task["max_attempts"]:
                        self.task_queue.append(task_id)
                    
                    # Remove from running tasks
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
                
                if resource:
                    resource["status"] = "available"
                    resource["usage_stats"]["failed_tasks"] += 1
            
            print(f"Error executing task {task_id}: {e}")
