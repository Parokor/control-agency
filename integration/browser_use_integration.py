import os
import sys
import asyncio
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

class BrowserUseIntegration:
    """Integration with Browser-Use for web automation"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.initialized = False
    
    async def initialize(self):
        """Initialize Browser-Use integration"""
        # Check if Browser-Use is installed
        try:
            import browser_use
            self.initialized = True
            return True
        except ImportError:
            # Install Browser-Use
            print("Installing Browser-Use...")
            os.system("pip install browser-use")
            
            # Install Playwright
            print("Installing Playwright...")
            os.system("playwright install chromium")
            
            try:
                import browser_use
                self.initialized = True
                return True
            except ImportError:
                print("Failed to install Browser-Use")
                return False
    
    async def execute_web_task(self, task: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a web task using Browser-Use"""
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {
                    "status": "error",
                    "error": "Failed to initialize Browser-Use"
                }
        
        # Load environment variables for API keys
        load_dotenv()
        
        # Import here to ensure it's installed
        from browser_use import Agent
        from langchain_openai import ChatOpenAI
        
        try:
            # Create agent
            agent = Agent(
                task=task,
                llm=ChatOpenAI(model=self.model),
                **(options or {})
            )
            
            # Execute task
            result = await agent.run()
            
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def search_and_analyze(self, query: str, analysis_prompt: str) -> Dict[str, Any]:
        """Search for information and analyze the results"""
        search_task = f"Search for information about: {query}"
        search_result = await self.execute_web_task(search_task)
        
        if search_result["status"] == "error":
            return search_result
        
        # Now analyze the search results
        analysis_task = f"{analysis_prompt}\n\nUse this information: {search_result['result']}"
        return await self.execute_web_task(analysis_task)
    
    async def compare_websites(self, urls: List[str], comparison_criteria: str) -> Dict[str, Any]:
        """Compare multiple websites based on specific criteria"""
        task = f"Visit these websites: {', '.join(urls)}\n\nCompare them based on the following criteria: {comparison_criteria}"
        return await self.execute_web_task(task)
    
    async def fill_form(self, url: str, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Fill out a form on a website"""
        form_instructions = "\n".join([f"- Fill '{field}' with '{value}'" for field, value in form_data.items()])
        task = f"Go to {url} and fill out the form with the following information:\n{form_instructions}\n\nSubmit the form and report the result."
        return await self.execute_web_task(task)
    
    async def monitor_website(self, url: str, check_frequency: int = 60, max_checks: int = 10) -> Dict[str, Any]:
        """Monitor a website for changes over time"""
        task = f"Visit {url} and record the current state. Check back every {check_frequency} seconds for a total of {max_checks} checks. Report any changes observed."
        
        options = {
            "max_iterations": max_checks,
            "wait_time_between_iterations": check_frequency
        }
        
        return await self.execute_web_task(task, options)

# Example usage
async def main():
    browser_agent = BrowserUseIntegration()
    await browser_agent.initialize()
    
    result = await browser_agent.execute_web_task("Compare the price of gpt-4o and DeepSeek-V3")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
