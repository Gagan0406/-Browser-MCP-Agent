import asyncio
import os
import streamlit as st
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

# Optional imports with graceful fallback
try:
    from mcp_agent.app import MCPApp
    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
    from mcp_agent.workflows.llm.augmented_llm import RequestParams
    MCP_AVAILABLE = True
except Exception:
    MCPApp = Agent = GoogleAugmentedLLM = RequestParams = None
    MCP_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    FAISS_AVAILABLE = True
except Exception:
    faiss = EMB_MODEL = None
    FAISS_AVAILABLE = False

# Configuration
class Config:
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    EXECUTION_TIMEOUT = 300
    MAX_MEMORY_ITEMS = 200
    MAX_TOOL_LOGS = 100
    SCREENSHOT_DIR = "/tmp"
    MAX_WAIT_SECONDS = 10
    LLM_PROVIDER = "groq"  # "groq" or "google"
    GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768"
    GROQ_MAX_TOKENS = 4096
    GROQ_TEMPERATURE = 0.7

class ActionType(Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    EXTRACT_TEXT = "extract_text"
    SCREENSHOT = "screenshot"
    TYPE = "type"
    WAIT = "wait"
    THINK = "think"
    SCRAPE = "scrape"
    SCROLL = "scroll"
    FILL = "fill"

@dataclass
class ExecutionResult:
    step: Dict[str, Any]
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0

def load_secrets() -> bool:
    """Load API secrets from configuration file."""
    secrets_file = "mcp_agent.secrets.yaml"
    api_key_loaded = False
    
    if not os.path.exists(secrets_file):
        # Check environment variables
        if os.getenv("GROQ_API_KEY"):
            api_key_loaded = True
        elif os.getenv("GEMINI_API_KEY"):
            api_key_loaded = True
        return api_key_loaded
    
    try:
        with open(secrets_file, "r") as file:
            secrets = yaml.safe_load(file)
            
            # Load Groq API key
            if secrets and "groq" in secrets and "api_key" in secrets["groq"]:
                api_key = secrets["groq"]["api_key"]
                if api_key and api_key != "YOUR_GROQ_API_KEY":
                    os.environ["GROQ_API_KEY"] = api_key
                    api_key_loaded = True
            
            # Load Gemini API key (fallback)
            if secrets and "google" in secrets and "api_key" in secrets["google"]:
                api_key = secrets["google"]["api_key"]
                if api_key and api_key != "YOUR_GEMINI_API_KEY":
                    os.environ["GEMINI_API_KEY"] = api_key
                    api_key_loaded = True
                    
    except Exception as e:
        st.error(f"Error loading secrets: {e}")
    
    return api_key_loaded

class VectorMemory:
    """Enhanced memory system with optional FAISS indexing."""
    
    def __init__(self):
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.index = None
        self.emb_model = EMB_MODEL if FAISS_AVAILABLE else None
        
    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        """Add text to memory with optional metadata."""
        if not text or not text.strip():
            return
            
        # Limit memory size
        if len(self.texts) >= Config.MAX_MEMORY_ITEMS:
            self.texts.pop(0)
            self.metadatas.pop(0)
            self.ids.pop(0)
            if self.index is not None and FAISS_AVAILABLE:
                self._rebuild_index()
        
        if FAISS_AVAILABLE and self.emb_model is not None:
            try:
                vec = self.emb_model.encode([text])[0].astype('float32')
                if self.index is None:
                    d = len(vec)
                    self.index = faiss.IndexFlatL2(d)
                self.index.add(vec.reshape(1, -1))
            except Exception:
                pass
        
        self.texts.append(text)
        self.metadatas.append(meta or {})
        self.ids.append(str(uuid4()))
    
    def _rebuild_index(self):
        """Rebuild FAISS index from existing texts."""
        if not FAISS_AVAILABLE or self.emb_model is None:
            return
        try:
            if self.texts:
                vecs = self.emb_model.encode(self.texts)
                d = vecs.shape[1]
                self.index = faiss.IndexFlatL2(d)
                self.index.add(vecs.astype('float32'))
        except Exception:
            self.index = None
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        if not self.texts:
            return []
        
        results = []
        k = min(k, len(self.texts))
        
        if FAISS_AVAILABLE and self.index is not None and self.emb_model is not None:
            try:
                qvec = self.emb_model.encode([query])[0].astype('float32')
                D, I = self.index.search(qvec.reshape(1, -1), k)
                for idx in I[0]:
                    if 0 <= idx < len(self.texts):
                        results.append({
                            "text": self.texts[idx],
                            "meta": self.metadatas[idx],
                            "id": self.ids[idx]
                        })
                return results
            except Exception:
                pass
        
        # Fallback: return most recent items
        for i in range(min(k, len(self.texts))):
            idx = len(self.texts) - 1 - i
            results.append({
                "text": self.texts[idx],
                "meta": self.metadatas[idx],
                "id": self.ids[idx]
            })
        return results
    
    def get_all(self) -> List[str]:
        """Get all memory texts."""
        return self.texts.copy()
    
    def clear(self):
        """Clear all memory."""
        self.texts.clear()
        self.metadatas.clear()
        self.ids.clear()
        self.index = None

class AgentWrapper:
    """Wrapper for MCP agents with fallback simulation."""
    
    def __init__(self, name: str, instruction: str, server_names: List[str] = None):
        self.name = name
        self.instruction = instruction
        self.server_names = server_names or []
        self.agent = None
        self.llm = None
        self.groq_client = None
        self.initialized = False
        self.simulation_mode = not MCP_AVAILABLE
        
        # Initialize Groq client if available
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            except Exception as e:
                st.warning(f"Groq client initialization failed: {e}")
    
    async def initialize(self):
        """Initialize the agent."""
        if not MCP_AVAILABLE:
            self.simulation_mode = True
            self.initialized = True
            return
        
        try:
            self.agent = Agent(
                name=self.name,
                instruction=self.instruction,
                server_names=self.server_names
            )
            await self.agent.initialize()
            self.initialized = True
        except Exception as e:
            st.warning(f"Agent {self.name} initialization failed: {e}. Using simulation mode.")
            self.simulation_mode = True
            self.initialized = True
    
    async def attach_llm(self, llm_cls):
        """Attach LLM to the agent."""
        if not self.agent or not GoogleAugmentedLLM:
            return
        try:
            self.llm = await self.agent.attach_llm(llm_cls)
        except Exception as e:
            st.warning(f"LLM attachment failed for {self.name}: {e}")
    
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with the given arguments."""
        if self.simulation_mode or not self.agent:
            await asyncio.sleep(0.2)
            return {
                "simulated": True,
                "tool": tool_name,
                "args": tool_args,
                "message": f"Simulated execution of {tool_name}"
            }
        
        try:
            # MCP Agent uses 'arguments' instead of 'args'
            return await self.agent.call_tool(name=tool_name, arguments=tool_args)
        except Exception as e:
            return {"error": str(e), "tool": tool_name, "args": tool_args}
    
    async def generate(self, message: str, request_params=None) -> str:
        """Generate a response using the LLM."""
        # Try Groq first if available
        if Config.LLM_PROVIDER == "groq" and self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": self.instruction},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=Config.GROQ_MAX_TOKENS,
                    temperature=Config.GROQ_TEMPERATURE
                )
                return response.choices[0].message.content
            except Exception as e:
                st.warning(f"Groq generation failed: {e}, falling back...")
        
        # Fallback to Google LLM
        if self.llm:
            try:
                return await self.llm.generate_str(
                    message=message,
                    request_params=request_params
                )
            except Exception as e:
                st.warning(f"LLM generation failed: {e}")
        
        # Final fallback response
        return f"[Simulated response from {self.name}] Received request: {message[:100]}..."

class Supervisor:
    """Orchestrates multi-agent workflow execution."""
    
    def __init__(
        self,
        browser_agent: AgentWrapper,
        analysis_agent: AgentWrapper,
        memory: VectorMemory,
        ui_hooks: Dict[str, Any]
    ):
        self.browser = browser_agent
        self.analysis = analysis_agent
        self.memory = memory
        self.ui_hooks = ui_hooks
        self.exec_queue: List[Dict[str, Any]] = []
        self.total_steps_executed = 0
    
    def _log(self, level: str, message: str):
        """Add a log entry."""
        timestamp = time.strftime("%H:%M:%S")
        self.ui_hooks["tool_logs"].append({
            "level": level,
            "message": message,
            "timestamp": timestamp
        })
        # Keep only recent logs
        if len(self.ui_hooks["tool_logs"]) > Config.MAX_TOOL_LOGS:
            self.ui_hooks["tool_logs"] = self.ui_hooks["tool_logs"][-Config.MAX_TOOL_LOGS:]
    
    async def plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create an execution plan for the given goal."""
        self._log("info", f"Creating plan for: {goal[:80]}...")
        
        # Retrieve relevant context from memory
        context = self.memory.retrieve(goal, k=5)
        context_str = "\n".join([f"- {c['text'][:150]}" for c in context]) if context else "No prior context"
        
        prompt = f"""You are a browser automation planner. Create a detailed execution plan as a JSON object.

Goal: {goal}

Recent context from memory:
{context_str}

Return a JSON object with this exact structure:
{{
    "steps": [
        {{"action": "navigate", "args": {{"url": "https://example.com"}} }},
        {{"action": "wait", "args": {{"seconds": 2}} }},
        {{"action": "extract_text", "args": {{"selector": "body"}} }},
        {{"action": "think", "args": {{"text": "analyze the extracted content"}} }}
    ]
}}

Available actions:
- navigate: Navigate to a URL - args: {{"url": "..."}}
- click: Click an element - args: {{"selector": "..."}}
- type: Type text into an input - args: {{"selector": "...", "text": "..."}}
- fill: Fill a form field - args: {{"selector": "...", "value": "..."}}
- extract_text: Extract text from page - args: {{"selector": "body"}}
- screenshot: Take a screenshot - args: {{"selector": null, "fullPage": false}}
- scroll: Scroll the page - args: {{"direction": "down", "amount": 500}}
- wait: Wait for seconds - args: {{"seconds": 2}}
- think: Analyze and reason - args: {{"text": "what to think about"}}

CRITICAL: Return ONLY valid JSON with the exact structure shown. No markdown, no explanations, just pure JSON."""

        try:
            plan_text = await self.analysis.generate(prompt)
            
            # Clean and parse JSON
            plan_text = plan_text.strip()
            
            # Remove markdown code blocks
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0]
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0]
            
            plan_text = plan_text.strip()
            
            # Try to parse JSON
            plan_json = json.loads(plan_text)
            steps = plan_json.get("steps", [])
            
            # Validate steps
            if not isinstance(steps, list):
                raise ValueError("Steps must be a list")
            
            if not steps:
                raise ValueError("No steps in plan")
            
            # Validate each step
            valid_steps = []
            for step in steps:
                if isinstance(step, dict) and "action" in step:
                    if "args" not in step:
                        step["args"] = {}
                    valid_steps.append(step)
            
            if not valid_steps:
                raise ValueError("No valid steps found")
            
            self._log("info", f"✓ Created plan with {len(valid_steps)} steps")
            self.exec_queue = valid_steps.copy()
            return valid_steps
            
        except json.JSONDecodeError as e:
            self._log("warning", f"JSON parsing failed: {e}. Creating fallback plan.")
            steps = self._create_fallback_plan(goal)
            self.exec_queue = steps.copy()
            return steps
        except Exception as e:
            self._log("warning", f"Plan creation failed: {e}. Creating fallback plan.")
            steps = self._create_fallback_plan(goal)
            self.exec_queue = steps.copy()
            return steps
    
    def _create_fallback_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create a simple fallback plan when JSON parsing fails."""
        steps = []
        goal_lower = goal.lower()
        
        # Check if goal contains a URL
        if "http://" in goal or "https://" in goal:
            # Extract URL
            words = goal.split()
            url = next((w for w in words if w.startswith("http")), "https://example.com")
            
            steps.extend([
                {"action": "navigate", "args": {"url": url}},
                {"action": "wait", "args": {"seconds": 3}},
                {"action": "extract_text", "args": {"selector": "body"}},
                {"action": "think", "args": {"text": f"Analyze the content from {url} for: {goal}"}}
            ])
        elif "search" in goal_lower:
            search_term = goal.replace("search", "").replace("for", "").strip()
            steps.extend([
                {"action": "navigate", "args": {"url": "https://www.google.com"}},
                {"action": "wait", "args": {"seconds": 2}},
                {"action": "type", "args": {"selector": "textarea[name='q']", "text": search_term}},
                {"action": "think", "args": {"text": f"Perform search for: {search_term}"}}
            ])
        else:
            steps.append({"action": "think", "args": {"text": goal}})
        
        return steps
    
    async def _execute_step(self, step: Dict[str, Any]) -> ExecutionResult:
        """Execute a single step with retries."""
        # Validate step is a dictionary
        if not isinstance(step, dict):
            self._log("error", f"Invalid step format: {type(step)}")
            return ExecutionResult(
                step={"action": "invalid", "args": {}},
                status="error",
                error=f"Invalid step format: expected dict, got {type(step).__name__}"
            )
        
        action = step.get("action", "")
        args = step.get("args", {})
        start_time = time.time()
        
        if not action:
            self._log("error", "No action specified in step")
            return ExecutionResult(
                step=step,
                status="error",
                error="No action specified"
            )
        
        self._log("info", f"→ Executing: {action}")
        
        # Execute with retries
        last_error = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                result = await self._execute_action(action, args)
                duration = time.time() - start_time
                
                # Store significant results in memory
                if result and str(result).strip():
                    result_str = str(result)
                    if len(result_str) > 100:  # Only store substantial results
                        self.memory.add(result_str[:2000], meta={
                            "source": "execution",
                            "action": action,
                            "timestamp": time.time()
                        })
                
                self._log("info", f"✓ {action} completed ({duration:.2f}s)")
                self.total_steps_executed += 1
                return ExecutionResult(
                    step=step, 
                    status="success", 
                    output=result,
                    duration=duration
                )
                
            except Exception as e:
                last_error = str(e)
                self._log("warning", f"Attempt {attempt + 1}/{Config.MAX_RETRIES} failed: {last_error}")
                
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
        
        # All retries failed
        duration = time.time() - start_time
        self._log("error", f"✗ {action} failed after {Config.MAX_RETRIES} attempts")
        return ExecutionResult(
            step=step, 
            status="error", 
            error=last_error or "Unknown error",
            duration=duration
        )
    
    async def _execute_action(self, action: str, args: Dict[str, Any]) -> Any:
        """Execute a specific action."""
        action_lower = action.lower()
        
        if action_lower == "navigate":
            url = args.get("url", "")
            if not url:
                raise ValueError("URL is required for navigate action")
            return await self.browser.call_tool("playwright.navigate", {"url": url})
        
        elif action_lower == "click":
            selector = args.get("selector", "")
            if not selector:
                raise ValueError("Selector is required for click action")
            return await self.browser.call_tool("playwright.click", {"selector": selector})
        
        elif action_lower in ("extract_text", "scrape"):
            selector = args.get("selector", "body")
            return await self.browser.call_tool("playwright.extract_text", {"selector": selector})
        
        elif action_lower == "screenshot":
            result = await self.browser.call_tool("playwright.screenshot", {
                "selector": args.get("selector"),
                "fullPage": args.get("fullPage", False)
            })
            
            # Save screenshot if data is returned
            if isinstance(result, dict) and "data" in result:
                try:
                    path = f"{Config.SCREENSHOT_DIR}/screenshot_{int(time.time())}.png"
                    with open(path, "wb") as f:
                        f.write(result["data"])
                    self.memory.add(f"Screenshot saved: {path}", meta={"type": "screenshot"})
                    return {"screenshot_saved": path, "path": path}
                except Exception as e:
                    return {"screenshot_error": str(e)}
            return result
        
        elif action_lower in ("type", "fill"):
            selector = args.get("selector", "")
            text = args.get("text") or args.get("value", "")
            if not selector or not text:
                raise ValueError("Selector and text are required")
            return await self.browser.call_tool("playwright.type", {
                "selector": selector,
                "text": text
            })
        
        elif action_lower == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 500)
            # Simulate scroll
            return await self.browser.call_tool("playwright.evaluate", {
                "script": f"window.scrollBy(0, {amount if direction == 'down' else -amount})"
            })
        
        elif action_lower == "wait":
            seconds = min(args.get("seconds", 1), Config.MAX_WAIT_SECONDS)
            await asyncio.sleep(seconds)
            return {"waited": seconds, "message": f"Waited for {seconds} seconds"}
        
        elif action_lower == "think":
            think_text = args.get("text", "")
            if not think_text:
                return {"analysis": "No thinking task specified"}
            
            # Retrieve relevant context
            context = self.memory.retrieve(think_text, k=5)
            context_str = "\n".join([f"- {c['text'][:200]}" for c in context]) if context else "No relevant context"
            
            prompt = f"""Analyze the following task and provide insights:

Task: {think_text}

Relevant context from previous actions:
{context_str}

Provide a clear, concise analysis with any insights or recommendations."""

            analysis = await self.analysis.generate(
                prompt,
                request_params=RequestParams(use_history=True, maxTokens=1000) if RequestParams else None
            )
            
            # Store the analysis
            self.memory.add(f"Analysis: {analysis[:500]}", meta={"type": "analysis"})
            
            return {"analysis": analysis, "task": think_text}
        
        else:
            # Try generic tool call
            return await self.browser.call_tool(action, args)
    
    async def execute(self, steps: List[Dict[str, Any]], adaptive: bool = True) -> List[ExecutionResult]:
        """Execute a list of steps sequentially."""
        results = []
        remaining_steps = steps.copy()
        
        # Validate steps
        valid_steps = []
        for i, step in enumerate(remaining_steps):
            if isinstance(step, dict) and "action" in step:
                valid_steps.append(step)
            else:
                self._log("warning", f"Skipping invalid step {i+1}: {step}")
                results.append(ExecutionResult(
                    step={"action": "invalid", "args": {}},
                    status="error",
                    error=f"Invalid step format at index {i}"
                ))
        
        remaining_steps = valid_steps
        self._log("info", f"Starting execution of {len(remaining_steps)} valid steps")
        
        while remaining_steps:
            step = remaining_steps.pop(0)
            result = await self._execute_step(step)
            results.append(result)
            
            # Handle adaptive planning after think actions
            if adaptive and step.get("action") == "think" and result.status == "success":
                new_steps = await self._handle_adaptive_planning(result, remaining_steps)
                if new_steps:
                    remaining_steps = new_steps + remaining_steps
        
        self._log("info", f"✓ Execution complete: {len(results)} steps, {self.total_steps_executed} total executed")
        return results
    
    async def _handle_adaptive_planning(
        self, 
        think_result: ExecutionResult,
        remaining_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate follow-up steps based on think action results."""
        try:
            # FIX: Handle both dict and string output types
            output = think_result.output
            if isinstance(output, dict):
                analysis = output.get("analysis", "")
            elif isinstance(output, str):
                analysis = output
            else:
                analysis = str(output) if output else ""
            
            if len(analysis) < 50:  # Skip if analysis is too short
                return []
            
            prompt = f"""Based on this analysis, determine if we need follow-up actions:

Analysis:
{analysis[:800]}

Current remaining steps: {len(remaining_steps)}

If follow-up actions are needed, return JSON with "steps" array.
If no follow-up is needed, return {{"steps": []}}

Available actions: navigate, click, extract_text, screenshot, type, wait, think

Return ONLY valid JSON, no markdown."""

            follow_text = await self.analysis.generate(
                prompt,
                request_params=RequestParams(use_history=True, maxTokens=1500) if RequestParams else None
            )
            
            # Parse follow-up steps
            if "```json" in follow_text:
                follow_text = follow_text.split("```json")[1].split("```")[0]
            elif "```" in follow_text:
                follow_text = follow_text.split("```")[1].split("```")[0]
            
            follow_json = json.loads(follow_text.strip())
            follow_steps = follow_json.get("steps", [])
            
            if follow_steps:
                self._log("info", f"→ Added {len(follow_steps)} adaptive steps")
                return follow_steps
                
        except Exception as e:
            self._log("warning", f"Adaptive planning skipped: {e}")
        
        return []

class AgentManager:
    """Main manager for the agent system."""
    
    def __init__(self):
        self.initialized = False
        self.mcp_app = None
        self.mcp_context = None
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.browser_agent = None
        self.analysis_agent = None
        self.memory = VectorMemory()
        self.supervisor = None
        self.ui_hooks: Dict[str, Any] = {
            "tool_logs": [],
            "execution_stats": {"total_runs": 0, "successful_steps": 0, "failed_steps": 0}
        }
    
    def start_background_loop(self):
        """Start the asyncio event loop in a background thread."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        timeout = 10
        start = time.time()
        while self.loop is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        if self.loop is None:
            raise RuntimeError("Failed to start event loop")
    
    async def setup_agents(self):
        """Initialize all agents and the supervisor."""
        if self.initialized:
            return
        
        # Initialize MCP app if available
        if MCP_AVAILABLE and MCPApp is not None:
            try:
                self.mcp_app = MCPApp(name="mcp_browser_agent")
                self.mcp_context = self.mcp_app.run()
                await self.mcp_context.__aenter__()
            except Exception as e:
                st.warning(f"MCP initialization: {e}")
        
        # Create agents
        self.browser_agent = AgentWrapper(
            name="browser",
            instruction="You are a browser automation agent. Execute web interactions precisely and efficiently.",
            server_names=["playwright"]
        )
        
        self.analysis_agent = AgentWrapper(
            name="analysis",
            instruction="You are an intelligent analysis agent. Plan tasks systematically and provide insightful analysis."
        )
        
        # Initialize agents
        await self.browser_agent.initialize()
        await self.analysis_agent.initialize()
        
        if GoogleAugmentedLLM is not None:
            await self.browser_agent.attach_llm(GoogleAugmentedLLM)
            await self.analysis_agent.attach_llm(GoogleAugmentedLLM)
        
        # Create supervisor
        self.supervisor = Supervisor(
            self.browser_agent,
            self.analysis_agent,
            self.memory,
            self.ui_hooks
        )
        
        self.initialized = True
    
    def run_coroutine(self, coro):
        """Run a coroutine in the background event loop."""
        if self.loop is None:
            self.start_background_loop()
        
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=Config.EXECUTION_TIMEOUT)
    
    def run_goal(self, goal: str, adaptive: bool = True) -> Dict[str, Any]:
        """Execute a goal end-to-end."""
        return self.run_coroutine(self._run_goal_async(goal, adaptive))
    
    async def _run_goal_async(self, goal: str, adaptive: bool) -> Dict[str, Any]:
        """Async implementation of goal execution."""
        await self.setup_agents()
        
        start_time = time.time()
        
        # Create plan
        steps = await self.supervisor.plan(goal)
        
        # Execute plan
        results = await self.supervisor.execute(steps, adaptive=adaptive)
        
        # Update stats
        self.ui_hooks["execution_stats"]["total_runs"] += 1
        for r in results:
            if r.status == "success":
                self.ui_hooks["execution_stats"]["successful_steps"] += 1
            else:
                self.ui_hooks["execution_stats"]["failed_steps"] += 1
        
        total_time = time.time() - start_time
        
        # Format results
        return {
            "goal": goal,
            "plan": steps,
            "results": [
                {
                    "step": r.step,
                    "status": r.status,
                    "output": r.output,
                    "error": r.error,
                    "duration": r.duration
                }
                for r in results
            ],
            "execution_time": total_time,
            "total_steps": len(results),
            "successful_steps": sum(1 for r in results if r.status == "success"),
            "failed_steps": sum(1 for r in results if r.status == "error")
        }
    
    def get_memory_summary(self) -> str:
        """Get a summary of current memory."""
        texts = self.memory.get_all()
        if not texts:
            return "Memory is empty"
        return f"{len(texts)} items in memory (most recent: {texts[-1][:100]}...)"
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            if self.mcp_context:
                async def cleanup_context():
                    await self.mcp_context.__aexit__(None, None, None)
                
                if self.loop:
                    asyncio.run_coroutine_threadsafe(cleanup_context(), self.loop)
            
            if self.executor:
                self.executor.shutdown(wait=False)
                
        except Exception:
            pass

# Streamlit App
st.set_page_config(
    page_title="MCP Browser Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 10px;
        background-color: #d4edda;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        padding: 10px;
        background-color: #f8d7da;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load secrets
secrets_loaded = load_secrets()

# Check LLM availability
llm_status = []
if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
    llm_status.append("Groq")
if os.getenv("GEMINI_API_KEY"):
    llm_status.append("Gemini")

# Initialize session state
if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()

mgr: AgentManager = st.session_state.agent_manager

# Main UI
st.title("🤖 MCP Browser Agent")
st.caption("Intelligent browser automation with multi-agent orchestration")

# Status indicators
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    if MCP_AVAILABLE:
        st.success("✓ MCP Available")
    else:
        st.info("ℹ️ MCP: Simulation Mode")
with col_status2:
    if llm_status:
        st.success(f"✓ LLM: {', '.join(llm_status)}")
    else:
        st.warning("⚠️ No LLM API Key")
with col_status3:
    if secrets_loaded:
        st.success("✓ Secrets Loaded")
    else:
        st.info("ℹ️ Using Env Vars")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # LLM Provider Selection
    st.subheader("🤖 LLM Provider")
    available_providers = []
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        available_providers.append("Groq")
    if os.getenv("GEMINI_API_KEY"):
        available_providers.append("Google")
    
    if available_providers:
        selected_provider = st.selectbox(
            "Select Provider",
            available_providers,
            index=0
        )
        Config.LLM_PROVIDER = selected_provider.lower()
        
        if selected_provider == "Groq":
            Config.GROQ_MODEL = st.selectbox(
                "Model",
                ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
                index=0
            )
    else:
        st.warning("No LLM provider available")
    
    st.markdown("---")
    
    goal = st.text_area(
        "Enter Goal",
        placeholder="Examples:\n• Navigate to https://news.ycombinator.com and extract top 5 stories\n• Search Google for 'Python asyncio tutorial'\n• Go to example.com and take a screenshot",
        height=150,
        key="goal_input"
    )
    
    adaptive = st.checkbox("Enable Adaptive Planning", value=True, 
                          help="Allow the agent to create follow-up steps dynamically")
    
    col1, col2 = st.columns(2)
    with col1:
        run_button = st.button("▶️ Run Goal", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear All", use_container_width=True)
    
    if clear_button:
        if "last_run" in st.session_state:
            del st.session_state["last_run"]
        mgr.memory.clear()
        mgr.ui_hooks["tool_logs"].clear()
        mgr.ui_hooks["execution_stats"] = {"total_runs": 0, "successful_steps": 0, "failed_steps": 0}
        st.success("✓ Cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Statistics")
    stats = mgr.ui_hooks["execution_stats"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Runs", stats["total_runs"])
    with col_b:
        st.metric("Success", stats["successful_steps"])
    with col_c:
        st.metric("Failed", stats["failed_steps"])
    
    # Memory info
    st.markdown("---")
    st.subheader("🧠 Memory")
    memory_summary = mgr.get_memory_summary()
    st.text(memory_summary)
    
    if st.button("Clear Memory", use_container_width=True):
        mgr.memory.clear()
        st.success("Memory cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # Execution logs
    st.subheader("📋 Logs")
    log_container = st.container()
    with log_container:
        if mgr.ui_hooks["tool_logs"]:
            for log in mgr.ui_hooks["tool_logs"][-15:]:
                level = log["level"]
                emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "📝")
                timestamp = log.get("timestamp", "")
                st.text(f"{emoji} [{timestamp}] {log['message']}")
        else:
            st.info("No logs yet")

# Main content
if run_button and goal and goal.strip():
    with st.spinner("🔄 Planning and executing..."):
        try:
            output = mgr.run_goal(goal.strip(), adaptive=adaptive)
            st.session_state["last_run"] = output
            st.success(f"✅ Execution complete! ({output['execution_time']:.2f}s)")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Execution error: {str(e)}")
            st.exception(e)

# Display results
if "last_run" in st.session_state:
    output = st.session_state["last_run"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", output.get("total_steps", 0))
    with col2:
        st.metric("Successful", output.get("successful_steps", 0))
    with col3:
        st.metric("Failed", output.get("failed_steps", 0))
    with col4:
        st.metric("Time", f"{output.get('execution_time', 0):.2f}s")
    
    st.markdown("---")
    
    # Goal
    st.subheader("🎯 Goal")
    st.info(output.get("goal", ""))
    
    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📋 Plan", "📊 Results", "💾 Export"])
    
    with tab1:
        st.subheader("Execution Plan")
        if output.get("plan"):
            for idx, step in enumerate(output["plan"], 1):
                action = step.get('action', 'unknown')
                with st.expander(f"Step {idx}: {action}", expanded=False):
                    st.json(step)
        else:
            st.warning("No plan generated")
    
    with tab2:
        st.subheader("Execution Results")
        if output.get("results"):
            for idx, result in enumerate(output["results"], 1):
                status = result.get("status", "unknown")
                status_emoji = {
                    "success": "✅",
                    "error": "❌",
                    "pending": "⏳"
                }.get(status, "❓")
                
                action = result['step'].get('action', 'unknown')
                duration = result.get('duration', 0)
                
                with st.expander(
                    f"{status_emoji} Step {idx}: {action} ({duration:.2f}s)", 
                    expanded=(status == "error")
                ):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.write("**Status:**")
                        st.write("**Action:**")
                        st.write("**Duration:**")
                    with col_b:
                        st.write(status)
                        st.write(action)
                        st.write(f"{duration:.2f}s")
                    
                    if result.get("output"):
                        st.write("**Output:**")
                        output_data = result["output"]
                        
                        # Handle different output types
                        if isinstance(output_data, dict):
                            if "analysis" in output_data:
                                st.markdown(output_data["analysis"])
                            elif "screenshot_saved" in output_data:
                                st.success(f"Screenshot: {output_data['screenshot_saved']}")
                            else:
                                st.json(output_data)
                        else:
                            output_str = str(output_data)
                            if len(output_str) > 3000:
                                st.text_area("Output (truncated)", output_str[:3000] + "\n... (truncated)", height=200)
                            else:
                                st.code(output_str, language="text")
                    
                    if result.get("error"):
                        st.error(f"**Error:** {result['error']}")
        else:
            st.warning("No results to display")
    
    with tab3:
        st.subheader("Export Results")
        
        # Helper function to make data JSON-serializable
        def make_serializable(obj):
            """Convert objects to JSON-serializable format."""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            else:
                return str(obj)
        
        # Prepare export data
        export_data = {
            "goal": output.get("goal"),
            "execution_time": output.get("execution_time"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "plan": make_serializable(output.get("plan", [])),
            "results": make_serializable(output.get("results", [])),
            "stats": {
                "total_steps": output.get("total_steps"),
                "successful_steps": output.get("successful_steps"),
                "failed_steps": output.get("failed_steps")
            }
        }
        
        try:
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name=f"agent_results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
            
            st.text_area("JSON Preview", json_str[:2000] + ("..." if len(json_str) > 2000 else ""), height=300)
        except Exception as e:
            st.error(f"Export error: {e}")
            st.text("Raw data preview:")
            st.write(export_data)

else:
    # Welcome message
    st.markdown("""
    ## 👋 Welcome to MCP Browser Agent
    
    This intelligent agent can automate browser tasks using natural language goals.
    
    ### 🚀 Quick Start:
    1. Enter your goal in the sidebar
    2. Choose whether to enable adaptive planning
    3. Click "Run Goal" and watch it execute
    
    ### 📝 Example Goals:
    - Navigate to https://example.com and extract the main heading
    - Go to news.ycombinator.com and get the top 5 story titles
    - Search Google for "Python tutorial" and click the first result
    - Take a screenshot of https://github.com
    - Navigate to a form, fill it out, and submit
    
    ### ✨ Features:
    - **Multi-step Planning**: Automatically breaks down complex goals
    - **Adaptive Execution**: Dynamically adjusts plan based on results
    - **Memory System**: Remembers context across steps
    - **Error Recovery**: Automatic retries with smart fallbacks
    - **Rich Logging**: Real-time execution logs
    
    **Ready to get started?** Enter a goal in the sidebar! 👈
    """)

# Footer
st.markdown("---")
st.caption("MCP Browser Agent | Powered by Multi-Agent Orchestration")