import asyncio
import os
import streamlit as st
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


# Load secrets from YAML file
def load_secrets():
    secrets_file = "mcp_agent.secrets.yaml"
    if os.path.exists(secrets_file):
        try:
            with open(secrets_file, "r") as file:
                secrets = yaml.safe_load(file)
                if secrets and "google" in secrets and "api_key" in secrets["google"]:
                    api_key = secrets["google"]["api_key"]
                    if api_key and api_key != "YOUR_GEMINI_API_KEY":
                        os.environ["GEMINI_API_KEY"] = api_key
                        return True
        except Exception as e:
            st.error(f"Error loading secrets: {e}")
    return False


# Global agent management
class AgentManager:
    def __init__(self):
        self.initialized = False
        self.mcp_app = None
        self.mcp_context = None
        self.mcp_agent_app = None
        self.browser_agent = None
        self.llm = None
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def start_background_loop(self):
        """Start event loop in background thread"""

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.1)

    async def setup_agent(self):
        """Setup the MCP agent asynchronously"""
        if not self.initialized:
            try:
                # Create MCP app and context
                self.mcp_app = MCPApp(name="streamlit_mcp_agent")
                self.mcp_context = self.mcp_app.run()
                self.mcp_agent_app = await self.mcp_context.__aenter__()

                # Create and initialize browser agent
                self.browser_agent = Agent(
                    name="browser",
                    instruction="""You are a helpful web browsing assistant that can interact with websites using playwright.
                        - Navigate to websites and perform browser actions (click, scroll, type)
                        - Extract information from web pages 
                        - Take screenshots of page elements when useful
                        - Provide concise summaries of web content using markdown
                        - Follow multi-step browsing sequences to complete tasks
                        
                    Respond back with a status update on completing the commands.""",
                    server_names=["playwright"],
                )

                # Initialize agent and attach LLM
                await self.browser_agent.initialize()
                self.llm = await self.browser_agent.attach_llm(GoogleAugmentedLLM)

                # Log available tools
                tools = await self.browser_agent.list_tools()
                logger = self.mcp_agent_app.logger
                logger.info("Tools available:", data=tools)

                self.initialized = True
                return None
            except Exception as e:
                return f"Error during initialization: {str(e)}"
        return None

    async def run_command(self, message):
        """Execute a command through the agent"""
        try:
            # Ensure agent is initialized
            error = await self.setup_agent()
            if error:
                return error

            if not message or not message.strip():
                return "Please enter a command for the agent to execute."

            # Generate response
            result = await self.llm.generate_str(
                message=message,
                request_params=RequestParams(use_history=True, maxTokens=10000),
            )
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def run_command_sync(self, message):
        """Synchronous wrapper for run_command"""
        if not self.loop:
            self.start_background_loop()

        future = asyncio.run_coroutine_threadsafe(self.run_command(message), self.loop)
        return future.result(timeout=120)  # 2 minute timeout

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.mcp_context:
                # Schedule cleanup in the background thread
                if self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.mcp_context.__aexit__(None, None, None), self.loop
                    )
        except Exception:
            pass


# Load secrets at startup
load_secrets()

# Page config
st.set_page_config(page_title="Browser MCP Agent", page_icon="🌐", layout="wide")

# Initialize agent manager
if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()

# Title and description
st.markdown("<h1 class='main-header'>🌐 Browser MCP Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "Interact with a powerful web browsing agent that can navigate and interact with websites"
)

# Setup sidebar with example commands
with st.sidebar:
    st.markdown("### Example Commands")

    st.markdown("**Navigation**")
    st.markdown("- Go to github.com/Shubhamsaboo/awesome-llm-apps")

    st.markdown("**Interactions**")
    st.markdown("- click on mcp_ai_agents")
    st.markdown("- Scroll down to view more content")

    st.markdown("**Multi-step Tasks**")
    st.markdown(
        "- Navigate to github.com/Shubhamsaboo/awesome-llm-apps, scroll down, and report details"
    )
    st.markdown("- Scroll down and summarize the github readme")

    st.markdown("---")
    st.caption("Note: The agent uses Playwright to control a real browser.")

# Query input
query = st.text_area(
    "Your Command",
    placeholder="Ask the agent to navigate to websites and interact with them",
)

# Initialize session state for results
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# Check API key
def check_api_key():
    if not os.getenv("GEMINI_API_KEY"):
        if not load_secrets():
            st.error(
                "⚠️ Gemini API key not found. Please check your mcp_agent.secrets.yaml file."
            )
            return False
    return True


# Run button and processing logic
if st.button(
    "🚀 Run Command",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.processing,
):
    if not check_api_key():
        st.stop()

    if query and query.strip():
        st.session_state.processing = True

        with st.spinner("Processing your request..."):
            try:
                result = st.session_state.agent_manager.run_command_sync(query)
                st.session_state.last_result = result
            except Exception as e:
                st.session_state.last_result = f"Error during execution: {str(e)}"
            finally:
                st.session_state.processing = False
                st.rerun()
    else:
        st.warning("Please enter a command before clicking Run.")

# Display results
if st.session_state.last_result:
    st.markdown("### Response")
    st.markdown(st.session_state.last_result)

# Display help text for first-time users
if not st.session_state.last_result:
    st.markdown(
        """<div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h4>How to use this app:</h4>
        <ol>
            <li>Enter your Google Gemini API key in your mcp_agent.secrets.yaml file</li>
            <li>Type a command for the agent to navigate and interact with websites</li>
            <li>Click 'Run Command' to see results</li>
        </ol>
        <p><strong>Capabilities:</strong></p>
        <ul>
            <li>Navigate to websites using Playwright</li>
            <li>Click on elements, scroll, and type text</li>
            <li>Take screenshots of specific elements</li>
            <li>Extract information from web pages</li>
            <li>Perform multi-step browsing tasks</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.write(
    "Built with Streamlit, Playwright, and [MCP-Agent](https://www.github.com/lastmile-ai/mcp-agent) Framework ❤️"
)

# Cleanup on app termination
import atexit

atexit.register(lambda: st.session_state.get("agent_manager", AgentManager()).cleanup())