"""
Advanced Gradio UI for Computer-Use Agent

This is a Gradio interface for the Computer-Use Agent
with an advanced UI for model selection and configuration.

Supported Agent Loops and Models:
- AgentLoop.OPENAI: Uses OpenAI Operator CUA model
  • computer-use-preview

- AgentLoop.ANTHROPIC: Uses Anthropic Computer-Use models
  • claude-3-5-sonnet-20240620
  • claude-3-7-sonnet-20250219

- AgentLoop.OMNI (experimental): Uses OmniParser for element pixel-detection
  • claude-3-5-sonnet-20240620
  • claude-3-7-sonnet-20250219
  • gpt-4.5-preview
  • gpt-4o
  • gpt-4

Requirements:
    - Mac with Apple Silicon (M1/M2/M3/M4)
    - macOS 14 (Sonoma) or newer
    - Python 3.11+
    - Lume CLI installed (https://github.com/trycua/cua)
    - OpenAI or Anthropic API key
"""

import os
import asyncio
import logging
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple, Union
import gradio as gr
from gradio.components.chatbot import MetadataDict
from typing import cast

# Import from agent package
from agent.core.types import AgentResponse
from agent.core.callbacks import DefaultCallbackHandler
from computer import Computer

from agent import ComputerAgent, AgentLoop, LLM, LLMProvider

# Global variables
global_agent = None
global_computer = None
SETTINGS_FILE = Path(".gradio_settings.json")

# We'll use asyncio.run() instead of a persistent event loop


# --- Settings Load/Save Functions ---
def load_settings() -> Dict[str, Any]:
    """Loads settings from the JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                # Basic validation (can be expanded)
                if isinstance(settings, dict):
                    print(f"DEBUG - Loaded settings from {SETTINGS_FILE}")
                    return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings from {SETTINGS_FILE}: {e}")
    return {}


def save_settings(settings: Dict[str, Any]):
    """Saves settings to the JSON file."""
    # Ensure sensitive keys are not saved
    settings.pop("provider_api_key", None)
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"DEBUG - Saved settings to {SETTINGS_FILE}")
    except IOError as e:
        print(f"Warning: Could not save settings to {SETTINGS_FILE}: {e}")


# --- End Settings Load/Save ---


# Custom Screenshot Handler for Gradio chat
class GradioChatScreenshotHandler(DefaultCallbackHandler):
    """Custom handler that adds screenshots to the Gradio chatbot and updates annotated image."""

    def __init__(self, chatbot_history: List[gr.ChatMessage]):
        """Initialize with reference to chat history and annotated image component.

        Args:
            chatbot_history: Reference to the Gradio chatbot history list
            annotated_image: Reference to the annotated image component
        """
        self.chatbot_history = chatbot_history
        print("GradioChatScreenshotHandler initialized")

    async def on_screenshot(
        self,
        screenshot_base64: str,
        action_type: str = "",
        parsed_screen: Optional[dict] = None,
    ) -> None:
        """Add screenshot to chatbot when a screenshot is taken and update the annotated image.

        Args:
            screenshot_base64: Base64 encoded screenshot
            action_type: Type of action that triggered the screenshot

        Returns:
            Original screenshot (does not modify it)
        """
        # Create a markdown image element for the screenshot
        image_markdown = (
            f"![Screenshot after {action_type}](data:image/png;base64,{screenshot_base64})"
        )

        # Simply append the screenshot as a new message
        if self.chatbot_history is not None:
            self.chatbot_history.append(
                gr.ChatMessage(
                    role="assistant",
                    content=image_markdown,
                    metadata={"title": f"🖥️ Screenshot - {action_type}", "status": "done"},
                )
            )


# Detect if current device is MacOS
is_mac = platform.system().lower() == "darwin"

# Detect if lume is available (host device is macOS)
is_lume_available = is_mac or (os.environ.get("PYLUME_HOST", "localhost") != "localhost")

print("PYLUME_HOST: ", os.environ.get("PYLUME_HOST", "localhost"))
print("is_mac: ", is_mac)
print("Lume available: ", is_lume_available)

# Map model names to specific provider model names
MODEL_MAPPINGS = {
    "openai": {
        # Default to operator CUA model
        "default": "computer-use-preview",
        "OpenAI: Computer-Use Preview": "computer-use-preview",
        # Map standard OpenAI model names to CUA-specific model names
        "gpt-4-turbo": "computer-use-preview",
        "gpt-4o": "computer-use-preview",
        "gpt-4": "computer-use-preview",
        "gpt-4.5-preview": "computer-use-preview",
        "gpt-4o-mini": "gpt-4o-mini",
    },
    "anthropic": {
        # Default to newest model
        "default": "claude-3-7-sonnet-20250219",
        # New Claude 4 models
        "Anthropic: Claude 4 Opus (20250514)": "claude-opus-4-20250514",
        "Anthropic: Claude 4 Sonnet (20250514)": "claude-sonnet-4-20250514",
        "claude-opus-4-20250514": "claude-opus-4-20250514",
        "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
        # Specific Claude models for CUA
        "Anthropic: Claude 3.7 Sonnet (20250219)": "claude-3-5-sonnet-20240620",
        "Anthropic: Claude 3.5 Sonnet (20240620)": "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        # Map standard model names to CUA-specific model names
        "claude-3-opus": "claude-3-7-sonnet-20250219",
        "claude-3-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    },
    "omni": {
        # OMNI works with any of these models
        "default": "gpt-4o",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4": "gpt-4",
        "gpt-4.5-preview": "gpt-4.5-preview",
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
    },
    "uitars": {
        # UI-TARS models using MLXVLM provider
        "default": "mlx-community/UI-TARS-1.5-7B-4bit" if is_mac else "tgi",
        "mlx-community/UI-TARS-1.5-7B-4bit": "mlx-community/UI-TARS-1.5-7B-4bit",
        "mlx-community/UI-TARS-1.5-7B-6bit": "mlx-community/UI-TARS-1.5-7B-6bit",
    },
    "ollama": {
        # For Ollama models, we keep the original name
        "default": "llama3",  # A common default model
        # Don't map other models - we'll use the original name
    },
    "oaicompat": {
        # Default for OpenAI-compatible providers like VLLM
        "default": "Qwen2.5-VL-7B-Instruct",
    },
    "azure": {
        # Azure OpenAI Service models
        "default": "gpt-4o",
        "Azure: GPT-4o": "gpt-4o",
        "Azure: GPT-4": "gpt-4",
        "Azure: GPT-35-Turbo": "gpt-35-turbo",
    },
}


def get_provider_and_model(model_name: str, loop_provider: str) -> tuple:
    """
    Determine the provider and actual model name to use based on the input.

    Args:
        model_name: The requested model name
        loop_provider: The requested agent loop provider

    Returns:
        tuple: (provider, model_name_to_use, agent_loop)
    """
    # Get the agent loop
    loop_provider_map = {
        "OPENAI": AgentLoop.OPENAI,
        "ANTHROPIC": AgentLoop.ANTHROPIC,
        "OMNI": AgentLoop.OMNI,
        "OMNI-OLLAMA": AgentLoop.OMNI,  # Special case for Ollama models with OMNI parser
        "UITARS": AgentLoop.UITARS,  # UI-TARS implementation
        "AZURE": AgentLoop.OMNI,  # Azure uses OAICOMPAT via OMNI loop
    }
    agent_loop = loop_provider_map.get(loop_provider, AgentLoop.OPENAI)

    if loop_provider == "AZURE":
        provider = LLMProvider.OAICOMPAT
        model_name_to_use = MODEL_MAPPINGS["azure"].get(
            model_name, MODEL_MAPPINGS["azure"]["default"]
        )
        agent_loop = AgentLoop.OMNI
        return provider, model_name_to_use, agent_loop

    # Set up the provider and model based on the loop and model_name
    if agent_loop == AgentLoop.OPENAI:
        provider = LLMProvider.OPENAI
        model_name_to_use = MODEL_MAPPINGS["openai"].get(
            model_name, MODEL_MAPPINGS["openai"]["default"]
        )
    elif agent_loop == AgentLoop.ANTHROPIC:
        provider = LLMProvider.ANTHROPIC
        model_name_to_use = MODEL_MAPPINGS["anthropic"].get(
            model_name, MODEL_MAPPINGS["anthropic"]["default"]
        )
    elif agent_loop == AgentLoop.OMNI:
        # Determine provider and clean model name based on the full string from UI
        cleaned_model_name = model_name  # Default to using the name as-is (for custom)

        if model_name == "Custom model (OpenAI compatible API)":
            # Actual model name comes from custom_model_value via model_to_use.
            # Assume OAICOMPAT for custom models unless overridden by URL/key later?
            # get_provider_and_model determines the *initial* provider/model.
            # The custom URL/key in process_response ultimately dictates the OAICOMPAT setup.
            provider = LLMProvider.OAICOMPAT
            # We set cleaned_model_name below outside the checks based on model_to_use
            cleaned_model_name = ""  # Placeholder, will be set by custom value later
        elif model_name.startswith("OMNI: Ollama "):
            provider = LLMProvider.OLLAMA
            # Extract the part after "OMNI: Ollama "
            cleaned_model_name = model_name.split("OMNI: Ollama ", 1)[1]
        elif model_name.startswith("OMNI: Claude "):
            provider = LLMProvider.ANTHROPIC

            model_name = model_name.replace("OMNI: ", "Anthropic: ")
            cleaned_model_name = MODEL_MAPPINGS["anthropic"].get(
                model_name, MODEL_MAPPINGS["anthropic"]["default"]
            )
        elif model_name.startswith("OMNI: OpenAI "):
            provider = LLMProvider.OPENAI
            # Extract the model part, e.g., "GPT-4o mini"
            model_key_part = model_name.replace("OMNI: OpenAI ", "")
            # Normalize the extracted part: "gpt4omini"
            model_key_part_norm = model_key_part.lower().replace("-", "").replace(" ", "")

            cleaned_model_name = MODEL_MAPPINGS["omni"]["default"]  # Default if not found
            # Find the canonical name in the main OMNI map for OpenAI models
            for key_omni, val_omni in MODEL_MAPPINGS["omni"].items():
                # Normalize the omni map key: "gpt-4o-mini" -> "gpt4omini"
                key_omni_norm = key_omni.lower().replace("-", "").replace(" ", "")
                # Check if the normalized omni key matches the normalized extracted part
                if key_omni_norm == model_key_part_norm:
                    cleaned_model_name = (
                        val_omni  # Use the value from the OMNI map (e.g., gpt-4o-mini)
                    )
                    break
            # Note: No fallback needed here as we explicitly check against omni keys

        else:  # Handles unexpected formats or the raw custom name if "Custom model (OpenAI compatible API)" selected
            # Should only happen if user selected "Custom model (OpenAI compatible API)"
            # Or if a model name format isn't caught above
            provider = LLMProvider.OAICOMPAT
            cleaned_model_name = (
                model_name.strip() if model_name.strip() else MODEL_MAPPINGS["oaicompat"]["default"]
            )

        # Assign the determined model name
        model_name_to_use = cleaned_model_name
        # agent_loop remains AgentLoop.OMNI
    elif agent_loop == AgentLoop.UITARS:
        # For UITARS, use MLXVLM for mlx-community models, OAICOMPAT for custom
        if model_name == "Custom model (OpenAI compatible API)":
            provider = LLMProvider.OAICOMPAT
            model_name_to_use = "tgi"
        else:
            provider = LLMProvider.MLXVLM
            # Get the model name from the mappings or use as-is if not found
            model_name_to_use = MODEL_MAPPINGS["uitars"].get(
                model_name, model_name if model_name else MODEL_MAPPINGS["uitars"]["default"]
            )
    else:
        # Default to OpenAI if unrecognized loop
        provider = LLMProvider.OPENAI
        model_name_to_use = MODEL_MAPPINGS["openai"]["default"]
        agent_loop = AgentLoop.OPENAI

    print(
        f"Mapping {model_name} and {loop_provider} to {provider}, {model_name_to_use}, {agent_loop}"
    )

    return provider, model_name_to_use, agent_loop


def get_ollama_models() -> List[str]:
    """Get available models from Ollama if installed."""
    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:  # No models or just header
                return []

            models = []
            # Skip header line
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(f"OMNI: Ollama {model_name}")
            return models
        return []
    except Exception as e:
        logging.error(f"Error getting Ollama models: {e}")
        return []


def create_computer_instance(
    verbosity: int = logging.INFO,
    os_type: str = "macos",
    provider_type: str = "lume",
    name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Computer:
    """Create or get the global Computer instance."""
    global global_computer

    if global_computer is None:
        global_computer = Computer(
            verbosity=verbosity,
            os_type=os_type,
            provider_type=provider_type,
            name=name if name else "",
            api_key=api_key,
        )

    return global_computer


def create_agent(
    provider: LLMProvider,
    agent_loop: AgentLoop,
    model_name: str,
    api_key: Optional[str] = None,
    save_trajectory: bool = True,
    only_n_most_recent_images: int = 3,
    verbosity: int = logging.INFO,
    use_oaicompat: bool = False,
    provider_base_url: Optional[str] = None,
    computer_os: str = "macos",
    computer_provider: str = "lume",
    computer_name: Optional[str] = None,
    computer_api_key: Optional[str] = None,
) -> ComputerAgent:
    """Create or update the global agent with the specified parameters."""
    global global_agent

    # Create the computer if not already done
    computer = create_computer_instance(
        verbosity=verbosity,
        os_type=computer_os,
        provider_type=computer_provider,
        name=computer_name,
        api_key=computer_api_key,
    )

    # Get API key from environment if not provided
    if api_key is None:
        if provider == LLMProvider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == LLMProvider.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Use provided provider_base_url if available, otherwise use default
    default_base_url = "http://localhost:1234/v1" if use_oaicompat else None
    custom_base_url = provider_base_url or default_base_url

    if use_oaicompat:
        # Special handling for OAICOMPAT - use OAICOMPAT provider with custom base URL
        print(f"DEBUG - Creating OAICOMPAT agent with model: {model_name}, URL: {custom_base_url}")
        llm = LLM(
            provider=LLMProvider.OAICOMPAT,  # Set to OAICOMPAT instead of using original provider
            name=model_name,
            provider_base_url=custom_base_url,
        )
        print(f"DEBUG - LLM provider is now: {llm.provider}, base URL: {llm.provider_base_url}")
        # Note: Don't pass use_oaicompat to the agent, as it doesn't accept this parameter
    elif provider == LLMProvider.OAICOMPAT:
        # This path is unlikely to be taken with our current approach
        llm = LLM(provider=provider, name=model_name, provider_base_url=custom_base_url)
    else:
        # For other providers, just use standard parameters
        llm = LLM(provider=provider, name=model_name)

    # Create or update the agent
    global_agent = ComputerAgent(
        computer=computer,
        loop=agent_loop,
        model=llm,
        api_key=api_key,
        save_trajectory=save_trajectory,
        only_n_most_recent_images=only_n_most_recent_images,
        verbosity=verbosity,
    )

    return global_agent


def create_gradio_ui(
    provider_name: str = "openai",
    model_name: str = "gpt-4o",
) -> gr.Blocks:
    """Create a Gradio UI for the Computer-Use Agent.

    Args:
        provider_name: The provider to use (e.g., "openai", "anthropic")
        model_name: The model to use (e.g., "gpt-4o", "claude-3-7-sonnet")

    Returns:
        A Gradio Blocks application
    """
    # --- Load Settings ---
    saved_settings = load_settings()
    # --- End Load Settings ---

    # Check for API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    cua_api_key = os.environ.get("CUA_API_KEY", "")

    # Always show models regardless of API key availability
    openai_models = ["OpenAI: Computer-Use Preview"]
    anthropic_models = [
        "Anthropic: Claude 4 Opus (20250514)",
        "Anthropic: Claude 4 Sonnet (20250514)",
        "Anthropic: Claude 3.7 Sonnet (20250219)",
        "Anthropic: Claude 3.5 Sonnet (20240620)",
    ]
    omni_models = [
        "OMNI: OpenAI GPT-4o",
        "OMNI: OpenAI GPT-4o mini",
        "OMNI: OpenAI GPT-4.5-preview",
        "OMNI: Claude 4 Opus (20250514)",
        "OMNI: Claude 4 Sonnet (20250514)",
        "OMNI: Claude 3.7 Sonnet (20250219)",
        "OMNI: Claude 3.5 Sonnet (20240620)",
    ]

    # Check if API keys are available
    has_openai_key = bool(openai_api_key)
    has_anthropic_key = bool(anthropic_api_key)
    has_cua_key = bool(cua_api_key)

    print("has_openai_key", has_openai_key)
    print("has_anthropic_key", has_anthropic_key)
    print("has_cua_key", has_cua_key)

    # Get Ollama models for OMNI
    ollama_models = get_ollama_models()
    if ollama_models:
        omni_models += ollama_models

    # Detect if current device is MacOS
    is_mac = platform.system().lower() == "darwin"

    # Format model choices
    provider_to_models = {
        "OPENAI": openai_models,
        "ANTHROPIC": anthropic_models,
        "OMNI": omni_models
        + [
            "Custom model (OpenAI compatible API)",
            "Custom model (ollama)",
        ],  # Add custom model options
        "UITARS": (
            [
                "mlx-community/UI-TARS-1.5-7B-4bit",
                "mlx-community/UI-TARS-1.5-7B-6bit",
            ]
            if is_mac
            else []
        )
        + ["Custom model (OpenAI compatible API)"],  # UI-TARS options with MLX models
        "AZURE": [],  # Azure uses custom deployment name
    }

    # --- Apply Saved Settings (override defaults if available) ---
    initial_loop = saved_settings.get("agent_loop", "OMNI")
    # Ensure the saved model is actually available in the choices for the loaded loop
    available_models_for_loop = provider_to_models.get(initial_loop, [])
    saved_model_choice = saved_settings.get("model_choice")
    if saved_model_choice and saved_model_choice in available_models_for_loop:
        initial_model = saved_model_choice
    else:
        # If saved model isn't valid for the loop, reset to default for that loop
        if initial_loop == "OPENAI":
            initial_model = (
                "OpenAI: Computer-Use Preview" if openai_models else "No models available"
            )
        elif initial_loop == "ANTHROPIC":
            initial_model = anthropic_models[0] if anthropic_models else "No models available"
        elif initial_loop == "AZURE":
            initial_model = ""
        else:  # OMNI
            initial_model = (
                omni_models[0] if omni_models else "Custom model (OpenAI compatible API)"
            )
            if "Custom model (OpenAI compatible API)" in available_models_for_loop:
                initial_model = "Custom model (OpenAI compatible API)"  # Default to custom if available and no other default fits

    initial_custom_model = saved_settings.get("custom_model", "Qwen2.5-VL-7B-Instruct")
    initial_provider_base_url = saved_settings.get("provider_base_url", "http://localhost:1234/v1")
    initial_azure_endpoint = saved_settings.get("azure_endpoint", "")
    initial_azure_deployment = saved_settings.get("azure_deployment", "")
    initial_save_trajectory = saved_settings.get("save_trajectory", True)
    initial_recent_images = saved_settings.get("recent_images", 3)
    # --- End Apply Saved Settings ---

    # Example prompts
    example_messages = [
        "Create a Python virtual environment, install pandas and matplotlib, then plot stock data",
        "Open a PDF in Preview, add annotations, and save it as a compressed version",
        "Open Safari, search for 'macOS automation tools', and save the first three results as bookmarks",
        "Configure SSH keys and set up a connection to a remote server",
    ]

    # Function to generate Python code based on configuration and tasks
    def generate_python_code(
        agent_loop_choice,
        provider,
        model_name,
        tasks,
        provider_url,
        recent_images=3,
        save_trajectory=True,
        computer_os="macos",
        computer_provider="lume",
        container_name="",
        cua_cloud_api_key="",
    ):
        """Generate Python code for the current configuration and tasks.

        Args:
            agent_loop_choice: The agent loop type (e.g., UITARS, OPENAI, ANTHROPIC, OMNI)
            provider: The provider type (e.g., OPENAI, ANTHROPIC, OLLAMA, OAICOMPAT, MLXVLM)
            model_name: The model name
            tasks: List of tasks to execute
            provider_url: The provider base URL for OAICOMPAT providers
            recent_images: Number of recent images to keep in context
            save_trajectory: Whether to save the agent trajectory
            computer_os: Operating system type for the computer
            computer_provider: Provider type for the computer
            container_name: Optional VM name
            cua_cloud_api_key: Optional CUA Cloud API key

        Returns:
            Formatted Python code as a string
        """
        # Format the tasks as a Python list
        tasks_str = ""
        for task in tasks:
            if task and task.strip():
                tasks_str += f'            "{task}",\n'

        # Create the Python code template with computer configuration
        computer_args = []
        if computer_os != "macos":
            computer_args.append(f'os_type="{computer_os}"')
        if computer_provider != "lume":
            computer_args.append(f'provider_type="{computer_provider}"')
        if container_name:
            computer_args.append(f'name="{container_name}"')
        if cua_cloud_api_key:
            computer_args.append(f'api_key="{cua_cloud_api_key}"')

        computer_args_str = ", ".join(computer_args)
        if computer_args_str:
            computer_args_str = f"({computer_args_str})"
        else:
            computer_args_str = "()"

        code = f"""import asyncio
from computer import Computer
from agent import ComputerAgent, LLM, AgentLoop, LLMProvider

async def main():
    async with Computer{computer_args_str} as macos_computer:
        agent = ComputerAgent(
            computer=macos_computer,
            loop=AgentLoop.{agent_loop_choice},
            only_n_most_recent_images={recent_images},
            save_trajectory={save_trajectory},"""

        # Add the model configuration based on provider and agent loop
        if agent_loop_choice == "OPENAI":
            # For OPENAI loop, always use OPENAI provider with computer-use-preview
            code += f"""
            model=LLM(
                provider=LLMProvider.OPENAI, 
                name="computer-use-preview"
            )"""
        elif agent_loop_choice == "ANTHROPIC":
            # For ANTHROPIC loop, always use ANTHROPIC provider
            code += f"""
            model=LLM(
                provider=LLMProvider.ANTHROPIC, 
                name="{model_name}"
            )"""
        elif agent_loop_choice == "UITARS":
            # For UITARS, use MLXVLM for mlx-community models, OAICOMPAT for others
            if provider == LLMProvider.MLXVLM:
                code += f"""
            model=LLM(
                provider=LLMProvider.MLXVLM, 
                name="{model_name}"
            )"""
            else:  # OAICOMPAT
                code += f"""
            model=LLM(
                provider=LLMProvider.OAICOMPAT, 
                name="{model_name}",
                provider_base_url="{provider_url}"
            )"""
        elif agent_loop_choice == "OMNI":
            # For OMNI, provider can be OPENAI, ANTHROPIC, OLLAMA, or OAICOMPAT
            if provider == LLMProvider.OAICOMPAT:
                code += f"""
            model=LLM(
                provider=LLMProvider.OAICOMPAT, 
                name="{model_name}",
                provider_base_url="{provider_url}"
            )"""
            else:  # OPENAI, ANTHROPIC, OLLAMA
                code += f"""
            model=LLM(
                provider=LLMProvider.{provider.name}, 
                name="{model_name}"
            )"""
        else:
            # Default case - just use the provided provider and model
            code += f"""
            model=LLM(
                provider=LLMProvider.{provider.name}, 
                name="{model_name}"
            )"""

        code += """
        )
        """

        # Add tasks section if there are tasks
        if tasks_str:
            code += f"""
        # Prompts for the computer-use agent
        tasks = [
{tasks_str.rstrip()}
        ]

        for task in tasks:
            print(f"Executing task: {{task}}")
            async for result in agent.run(task):
                print(result)"""
        else:
            # If no tasks, just add a placeholder for a single task
            code += f"""
        # Execute a single task
        task = "Search for information about CUA on GitHub"
        print(f"Executing task: {{task}}")
        async for result in agent.run(task):
            print(result)"""

        # Add the main block
        code += """

if __name__ == "__main__":
    asyncio.run(main())"""

        return code

    # Create the Gradio interface with advanced UI
    with gr.Blocks(title="Computer-Use Agent") as demo:
        with gr.Row():
            # Left column for settings
            with gr.Column(scale=1):
                # Logo with theme-aware styling
                gr.HTML(
                    """
                    <style>
                    .light-logo, .dark-logo {
                        display: block;
                        margin: 0 auto;
                        width: 80px;
                    }
                    /* Hide dark logo in light mode */
                    .dark-logo {
                        display: none;
                    }
                    /* In dark mode, hide light logo and show dark logo */
                    .dark .light-logo {
                        display: none;
                    }
                    .dark .dark-logo {
                        display: block;
                    }
                    </style>
                    <div style="display: flex; justify-content: center; margin-bottom: 0.5em">
                        <img class="light-logo" alt="CUA Logo" 
                             src="https://github.com/trycua/cua/blob/main/img/logo_black.png?raw=true" />
                        <img class="dark-logo" alt="CUA Logo" 
                             src="https://github.com/trycua/cua/blob/main/img/logo_white.png?raw=true" />
                    </div>
                    """
                )

                # Add accordion for Python code
                with gr.Accordion("Python Code", open=False):
                    code_display = gr.Code(
                        language="python",
                        value=generate_python_code(
                            initial_loop,
                            LLMProvider.OPENAI,
                            "gpt-4o",
                            [],
                            "https://openrouter.ai/api/v1",
                            3,  # recent_images default
                            True,  # save_trajectory default
                            "macos",
                            "lume",
                            "",
                            "",
                        ),
                        interactive=False,
                    )

                with gr.Accordion("Computer Configuration", open=True):
                    # Computer configuration options
                    computer_os = gr.Radio(
                        choices=["macos", "linux", "windows"],
                        label="Operating System",
                        value="macos",
                        info="Select the operating system for the computer",
                    )

                    is_windows = platform.system().lower() == "windows"
                    is_mac = platform.system().lower() == "darwin"

                    providers = ["cloud"]
                    if is_lume_available:
                        providers += ["lume"]
                    if is_windows:
                        providers += ["winsandbox"]

                    computer_provider = gr.Radio(
                        choices=providers,
                        label="Provider",
                        value="lume" if is_mac else "cloud",
                        info="Select the computer provider",
                    )

                    container_name = gr.Textbox(
                        label="Container Name",
                        placeholder="Enter container name (optional)",
                        value="",
                        info="Optional name for the container",
                    )

                    cua_cloud_api_key = gr.Textbox(
                        label="CUA Cloud API Key",
                        placeholder="Enter your CUA Cloud API key",
                        value="",
                        type="password",
                        info="Required for cloud provider",
                        visible=(not has_cua_key),
                    )

                with gr.Accordion("Agent Configuration", open=True):
                    # Configuration options
                    agent_loop = gr.Dropdown(
                        choices=["OPENAI", "ANTHROPIC", "OMNI", "UITARS", "AZURE"],
                        label="Agent Loop",
                        value=initial_loop,
                        info="Select the agent loop provider",
                    )

                    # Create separate model selection dropdowns for each provider type
                    # This avoids the Gradio bug with updating choices
                    with gr.Group() as model_selection_group:
                        # OpenAI models dropdown
                        openai_model_choice = gr.Dropdown(
                            choices=openai_models,
                            label="OpenAI Model",
                            value=openai_models[0] if openai_models else "No models available",
                            info="Select OpenAI model",
                            interactive=True,
                            visible=(initial_loop == "OPENAI"),
                        )

                        # Anthropic models dropdown
                        anthropic_model_choice = gr.Dropdown(
                            choices=anthropic_models,
                            label="Anthropic Model",
                            value=(
                                anthropic_models[0] if anthropic_models else "No models available"
                            ),
                            info="Select Anthropic model",
                            interactive=True,
                            visible=(initial_loop == "ANTHROPIC"),
                        )

                        # OMNI models dropdown
                        omni_model_choice = gr.Dropdown(
                            choices=omni_models
                            + ["Custom model (OpenAI compatible API)", "Custom model (ollama)"],
                            label="OMNI Model",
                            value=(
                                omni_models[0]
                                if omni_models
                                else "Custom model (OpenAI compatible API)"
                            ),
                            info="Select OMNI model or choose a custom model option",
                            interactive=True,
                            visible=(initial_loop == "OMNI"),
                        )

                        # UITARS models dropdown
                        uitars_model_choice = gr.Dropdown(
                            choices=provider_to_models.get("UITARS", ["No models available"]),
                            label="UITARS Model",
                            value=(
                                provider_to_models.get("UITARS", ["No models available"])[0]
                                if provider_to_models.get("UITARS")
                                else "No models available"
                            ),
                            info="Select UITARS model",
                            interactive=True,
                            visible=(initial_loop == "UITARS"),
                        )

                        # Hidden field to store the selected model (for compatibility with existing code)
                        model_choice = gr.Textbox(visible=False)

                    # Add API key inputs for OpenAI and Anthropic
                    with gr.Group(
                        visible=not has_openai_key
                        and (initial_loop == "OPENAI" or initial_loop == "OMNI")
                    ) as openai_key_group:
                        openai_api_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="Enter your OpenAI API key",
                            value="",
                            interactive=True,
                            type="password",
                            info="Required for OpenAI models",
                        )

                    with gr.Group(
                        visible=not has_anthropic_key
                        and (initial_loop == "ANTHROPIC" or initial_loop == "OMNI")
                    ) as anthropic_key_group:
                        anthropic_api_key_input = gr.Textbox(
                            label="Anthropic API Key",
                            placeholder="Enter your Anthropic API key",
                            value="",
                            interactive=True,
                            type="password",
                            info="Required for Anthropic models",
                        )

                    # Function to set OpenAI API key environment variable
                    def set_openai_api_key(key):
                        if key and key.strip():
                            os.environ["OPENAI_API_KEY"] = key.strip()
                            print(f"DEBUG - Set OpenAI API key environment variable")
                        return key

                    # Function to set Anthropic API key environment variable
                    def set_anthropic_api_key(key):
                        if key and key.strip():
                            os.environ["ANTHROPIC_API_KEY"] = key.strip()
                            print(f"DEBUG - Set Anthropic API key environment variable")
                        return key

                    def set_azure_api_key(key):
                        if key and key.strip():
                            os.environ["AZURE_OPENAI_API_KEY"] = key.strip()
                            print(f"DEBUG - Set Azure API key environment variable")
                        return key

                    # Add change event handlers for API key inputs
                    openai_api_key_input.change(
                        fn=set_openai_api_key,
                        inputs=[openai_api_key_input],
                        outputs=[openai_api_key_input],
                        queue=False,
                    )

                    anthropic_api_key_input.change(
                        fn=set_anthropic_api_key,
                        inputs=[anthropic_api_key_input],
                        outputs=[anthropic_api_key_input],
                        queue=False,
                    )

                    azure_api_key_input.change(
                        fn=set_azure_api_key,
                        inputs=[azure_api_key_input],
                        outputs=[azure_api_key_input],
                        queue=False,
                    )

                    # Combined function to update UI based on selections
                    def update_ui(
                        loop=None,
                        openai_model=None,
                        anthropic_model=None,
                        omni_model=None,
                        uitars_model=None,
                    ):
                        # Default values if not provided
                        loop = loop or agent_loop.value

                        # Determine which model value to use for custom model checks
                        model_value = None
                        if loop == "OPENAI" and openai_model:
                            model_value = openai_model
                        elif loop == "ANTHROPIC" and anthropic_model:
                            model_value = anthropic_model
                        elif loop == "OMNI" and omni_model:
                            model_value = omni_model
                        elif loop == "UITARS" and uitars_model:
                            model_value = uitars_model
                        azure_visible = loop == "AZURE"

                        # Show/hide appropriate model dropdown based on loop selection
                        openai_visible = loop == "OPENAI"
                        anthropic_visible = loop == "ANTHROPIC"
                        omni_visible = loop == "OMNI"
                        uitars_visible = loop == "UITARS"

                        # Show/hide API key inputs based on loop selection
                        show_openai_key = not has_openai_key and (
                            loop == "OPENAI"
                            or (
                                loop == "OMNI"
                                and model_value
                                and "OpenAI" in model_value
                                and "Custom" not in model_value
                            )
                        )
                        show_anthropic_key = not has_anthropic_key and (
                            loop == "ANTHROPIC"
                            or (
                                loop == "OMNI"
                                and model_value
                                and "Claude" in model_value
                                and "Custom" not in model_value
                            )
                        )
                        show_azure_key = loop == "AZURE"

                        # Determine custom model visibility
                        is_custom_openai_api = (
                            model_value == "Custom model (OpenAI compatible API)" or loop == "AZURE"
                        )
                        is_custom_ollama = model_value == "Custom model (ollama)"
                        is_any_custom = is_custom_openai_api or is_custom_ollama

                        # Update the hidden model_choice field based on the visible dropdown
                        model_choice_value = model_value if model_value else ""

                        # Return all UI updates
                        return [
                            # Model dropdowns visibility
                            gr.update(visible=openai_visible),
                            gr.update(visible=anthropic_visible),
                            gr.update(visible=omni_visible),
                            gr.update(visible=uitars_visible),
                            # API key inputs visibility
                            gr.update(visible=show_openai_key),
                            gr.update(visible=show_anthropic_key),
                            gr.update(visible=show_azure_key),
                            # Custom model fields visibility
                            gr.update(
                                visible=is_any_custom
                            ),  # Custom model name always visible for any custom option
                            gr.update(
                                visible=is_custom_openai_api
                            ),  # Provider base URL only for OpenAI compatible API
                            gr.update(
                                visible=is_custom_openai_api
                            ),  # Provider API key only for OpenAI compatible API
                            gr.update(visible=azure_visible),
                            gr.update(visible=azure_visible),
                            # Update the hidden model_choice field
                            gr.update(value=model_choice_value),
                        ]

                    # Add custom model textbox (visible for both custom model options)
                    custom_model = gr.Textbox(
                        label="Custom Model Name",
                        placeholder="Enter custom model name (e.g., Qwen2.5-VL-7B-Instruct or llama3)",
                        value=initial_custom_model,
                        visible=(
                            initial_model == "Custom model (OpenAI compatible API)"
                            or initial_model == "Custom model (ollama)"
                        ),
                        interactive=True,
                    )

                    # Add custom provider base URL textbox (only visible for OpenAI compatible API)
                    provider_base_url = gr.Textbox(
                        label="Provider Base URL",
                        placeholder="Enter provider base URL (e.g., http://localhost:1234/v1)",
                        value=initial_provider_base_url,
                        visible=(initial_model == "Custom model (OpenAI compatible API)"),
                        interactive=True,
                    )

                    # Add custom API key textbox (only visible for OpenAI compatible API)
                    provider_api_key = gr.Textbox(
                        label="Provider API Key",
                        placeholder="Enter provider API key (if required)",
                        value="",
                        visible=(initial_model == "Custom model (OpenAI compatible API)"),
                        interactive=True,
                        type="password",
                    )

                    with gr.Group(visible=(initial_loop == "AZURE")) as azure_key_group:
                        azure_api_key_input = gr.Textbox(
                            label="Azure API Key",
                            placeholder="Enter your Azure API key",
                            value="",
                            interactive=True,
                            type="password",
                            info="Required for Azure OpenAI models",
                        )

                    azure_endpoint = gr.Textbox(
                        label="Azure Endpoint",
                        placeholder="https://YOUR-RESOURCE.openai.azure.com/",
                        value=initial_azure_endpoint,
                        visible=(initial_loop == "AZURE"),
                        interactive=True,
                    )

                    azure_deployment = gr.Textbox(
                        label="Azure Deployment Name",
                        placeholder="Enter deployment name",
                        value=initial_azure_deployment,
                        visible=(initial_loop == "AZURE"),
                        interactive=True,
                    )

                    # Connect agent_loop changes to update all UI elements
                    agent_loop.change(
                        fn=update_ui,
                        inputs=[
                            agent_loop,
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                        ],
                        outputs=[
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                            openai_key_group,
                            anthropic_key_group,
                            azure_key_group,
                            custom_model,
                            provider_base_url,
                            provider_api_key,
                            azure_endpoint,
                            azure_deployment,
                            model_choice,  # Add model_choice to outputs
                        ],
                        queue=False,  # Process immediately without queueing
                    )

                    # Connect each model dropdown to update UI
                    omni_model_choice.change(
                        fn=update_ui,
                        inputs=[
                            agent_loop,
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                        ],
                        outputs=[
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                            openai_key_group,
                            anthropic_key_group,
                            azure_key_group,
                            custom_model,
                            provider_base_url,
                            provider_api_key,
                            azure_endpoint,
                            azure_deployment,
                            model_choice,  # Add model_choice to outputs
                        ],
                        queue=False,
                    )

                    uitars_model_choice.change(
                        fn=update_ui,
                        inputs=[
                            agent_loop,
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                        ],
                        outputs=[
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                            openai_key_group,
                            anthropic_key_group,
                            azure_key_group,
                            custom_model,
                            provider_base_url,
                            provider_api_key,
                            azure_endpoint,
                            azure_deployment,
                            model_choice,  # Add model_choice to outputs
                        ],
                        queue=False,
                    )

                    openai_model_choice.change(
                        fn=update_ui,
                        inputs=[
                            agent_loop,
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                        ],
                        outputs=[
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                            openai_key_group,
                            anthropic_key_group,
                            azure_key_group,
                            custom_model,
                            provider_base_url,
                            provider_api_key,
                            azure_endpoint,
                            azure_deployment,
                            model_choice,  # Add model_choice to outputs
                        ],
                        queue=False,
                    )

                    anthropic_model_choice.change(
                        fn=update_ui,
                        inputs=[
                            agent_loop,
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                        ],
                        outputs=[
                            openai_model_choice,
                            anthropic_model_choice,
                            omni_model_choice,
                            uitars_model_choice,
                            openai_key_group,
                            anthropic_key_group,
                            azure_key_group,
                            custom_model,
                            provider_base_url,
                            provider_api_key,
                            azure_endpoint,
                            azure_deployment,
                            model_choice,  # Add model_choice to outputs
                        ],
                        queue=False,
                    )

                    save_trajectory = gr.Checkbox(
                        label="Save Trajectory",
                        value=initial_save_trajectory,
                        info="Save the agent's trajectory for debugging",
                        interactive=True,
                    )

                    recent_images = gr.Slider(
                        label="Recent Images",
                        minimum=1,
                        maximum=10,
                        value=initial_recent_images,
                        step=1,
                        info="Number of recent images to keep in context",
                        interactive=True,
                    )

            # Right column for chat interface
            with gr.Column(scale=2):
                # Add instruction text before the chat interface
                gr.Markdown(
                    "Ask me to perform tasks in a virtual macOS environment.<br>Built with <a href='https://github.com/trycua/cua' target='_blank'>github.com/trycua/cua</a>."
                )

                chatbot_history = gr.Chatbot(type="messages")
                msg = gr.Textbox(
                    placeholder="Ask me to perform tasks in a virtual macOS environment"
                )
                clear = gr.Button("Clear")

                # Add cancel button
                cancel_button = gr.Button("Cancel", variant="stop")

                # Add examples
                example_group = gr.Examples(examples=example_messages, inputs=msg)

                # Function to handle chat submission
                def chat_submit(message, history):
                    # Add user message to history
                    history.append(gr.ChatMessage(role="user", content=message))
                    return "", history

                # Function to cancel the running agent
                async def cancel_agent_task(history):
                    global global_agent
                    if global_agent and hasattr(global_agent, "_loop"):
                        print("DEBUG - Cancelling agent task")
                        # Cancel the agent loop
                        if hasattr(global_agent._loop, "cancel") and callable(
                            global_agent._loop.cancel
                        ):
                            await global_agent._loop.cancel()
                            history.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content="Task cancelled by user",
                                    metadata={"title": "❌ Cancelled"},
                                )
                            )
                        else:
                            history.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content="Could not cancel task: cancel method not found",
                                    metadata={"title": "⚠️ Warning"},
                                )
                            )
                    else:
                        history.append(
                            gr.ChatMessage(
                                role="assistant",
                                content="No active agent task to cancel",
                                metadata={"title": "ℹ️ Info"},
                            )
                        )
                    return history

                # Function to process agent response after user input
                async def process_response(
                    history,
                    openai_model_value,
                    anthropic_model_value,
                    omni_model_value,
                    uitars_model_value,
                    custom_model_value,
                    agent_loop_choice,
                    save_traj,
                    recent_imgs,
                    custom_url_value=None,
                    custom_api_key=None,
                    openai_key_input=None,
                    anthropic_key_input=None,
                    azure_endpoint_value=None,
                    azure_deployment_value=None,
                    azure_key_input=None,
                    computer_os="macos",
                    computer_provider="lume",
                    container_name="",
                    cua_cloud_api_key="",
                ):
                    if not history:
                        yield history
                        return

                    # Get the last user message
                    last_user_message = history[-1]["content"]

                    # Get the appropriate model value based on the agent loop
                    if agent_loop_choice == "OPENAI":
                        model_choice_value = openai_model_value
                    elif agent_loop_choice == "ANTHROPIC":
                        model_choice_value = anthropic_model_value
                    elif agent_loop_choice == "OMNI":
                        model_choice_value = omni_model_value
                    elif agent_loop_choice == "UITARS":
                        model_choice_value = uitars_model_value
                    else:
                        model_choice_value = "No models available"

                    # Determine if this is a custom model selection and which type
                    is_custom_openai_api = (
                        model_choice_value == "Custom model (OpenAI compatible API)"
                        or agent_loop_choice == "AZURE"
                    )
                    is_custom_ollama = model_choice_value == "Custom model (ollama)"
                    is_custom_model_selected = is_custom_openai_api or is_custom_ollama

                    # Determine the model name string to analyze: custom or from dropdown
                    if is_custom_model_selected:
                        model_string_to_analyze = custom_model_value
                    else:
                        model_string_to_analyze = (
                            model_choice_value  # Use the full UI string initially
                        )

                    try:
                        # Special case for UITARS - use MLXVLM provider or OAICOMPAT for custom
                        if agent_loop_choice == "UITARS":
                            if is_custom_openai_api:
                                provider = LLMProvider.OAICOMPAT
                                cleaned_model_name_from_func = custom_model_value
                                agent_loop_type = AgentLoop.UITARS
                                print(
                                    f"Using OAICOMPAT provider for custom UITARS model: {custom_model_value}"
                                )
                            else:
                                provider = LLMProvider.MLXVLM
                                cleaned_model_name_from_func = model_string_to_analyze
                                agent_loop_type = AgentLoop.UITARS
                                print(
                                    f"Using MLXVLM provider for UITARS model: {model_string_to_analyze}"
                                )
                        # Special case for Ollama custom model
                        elif is_custom_ollama and agent_loop_choice == "OMNI":
                            provider = LLMProvider.OLLAMA
                            cleaned_model_name_from_func = custom_model_value
                            agent_loop_type = AgentLoop.OMNI
                            print(f"Using Ollama provider for custom model: {custom_model_value}")
                        elif agent_loop_choice == "AZURE":
                            provider = LLMProvider.OAICOMPAT
                            cleaned_model_name_from_func = (
                                azure_deployment_value or MODEL_MAPPINGS["azure"]["default"]
                            )
                            agent_loop_type = AgentLoop.OMNI
                            print(
                                f"Using Azure provider with deployment: {cleaned_model_name_from_func}"
                            )
                        else:
                            # Get the provider, *cleaned* model name, and agent loop type
                            provider, cleaned_model_name_from_func, agent_loop_type = (
                                get_provider_and_model(model_string_to_analyze, agent_loop_choice)
                            )

                        print(
                            f"provider={provider} cleaned_model_name_from_func={cleaned_model_name_from_func} agent_loop_type={agent_loop_type} agent_loop_choice={agent_loop_choice}"
                        )

                        # Determine the final model name to send to the agent
                        # If custom selected, use the custom text box value, otherwise use the cleaned name
                        if agent_loop_choice == "AZURE":
                            final_model_name_to_send = (
                                azure_deployment_value or cleaned_model_name_from_func
                            )
                        else:
                            final_model_name_to_send = (
                                custom_model_value
                                if is_custom_model_selected
                                else cleaned_model_name_from_func
                            )

                        # Determine if OAICOMPAT should be used (for OpenAI compatible API custom model)
                        is_oaicompat = is_custom_openai_api or agent_loop_choice == "AZURE"

                        # Get API key based on provider determined by get_provider_and_model
                        if agent_loop_choice == "AZURE":
                            api_key = (
                                azure_key_input
                                if azure_key_input
                                else os.environ.get("AZURE_OPENAI_API_KEY", "")
                            )
                            if azure_key_input:
                                os.environ["AZURE_OPENAI_API_KEY"] = azure_key_input
                                print(
                                    f"DEBUG - Using provided Azure API key from UI and set as environment variable"
                                )
                        elif is_oaicompat and custom_api_key:
                            # Use custom API key if provided for OpenAI compatible API custom model
                            api_key = custom_api_key
                            print(
                                f"DEBUG - Using custom API key for OpenAI compatible API model: {final_model_name_to_send}"
                            )
                        elif provider == LLMProvider.OLLAMA:
                            # No API key needed for Ollama
                            api_key = ""
                            print(
                                f"DEBUG - No API key needed for Ollama model: {final_model_name_to_send}"
                            )
                        elif provider == LLMProvider.OPENAI:
                            # Use OpenAI key from input if provided, otherwise use environment variable
                            api_key = (
                                openai_key_input
                                if openai_key_input
                                else (openai_api_key or os.environ.get("OPENAI_API_KEY", ""))
                            )
                            if openai_key_input:
                                # Set the environment variable for the OpenAI API key
                                os.environ["OPENAI_API_KEY"] = openai_key_input
                                print(
                                    f"DEBUG - Using provided OpenAI API key from UI and set as environment variable"
                                )
                        elif provider == LLMProvider.ANTHROPIC:
                            # Use Anthropic key from input if provided, otherwise use environment variable
                            api_key = (
                                anthropic_key_input
                                if anthropic_key_input
                                else (anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
                            )
                            if anthropic_key_input:
                                # Set the environment variable for the Anthropic API key
                                os.environ["ANTHROPIC_API_KEY"] = anthropic_key_input
                                print(
                                    f"DEBUG - Using provided Anthropic API key from UI and set as environment variable"
                                )
                        else:
                            # For Ollama or default OAICOMPAT (without custom key), no key needed/expected
                            api_key = ""

                        cua_cloud_api_key = cua_cloud_api_key or os.environ.get("CUA_API_KEY", "")

                        # --- Save Settings Before Running Agent ---
                        current_settings = {
                            "agent_loop": agent_loop_choice,
                            "model_choice": model_choice_value,
                            "custom_model": custom_model_value,
                            "provider_base_url": custom_url_value,
                            "azure_endpoint": azure_endpoint_value,
                            "azure_deployment": azure_deployment_value,
                            "save_trajectory": save_traj,
                            "recent_images": recent_imgs,
                            "computer_os": computer_os,
                            "computer_provider": computer_provider,
                            "container_name": container_name,
                            "cua_cloud_api_key": cua_cloud_api_key,
                        }
                        save_settings(current_settings)
                        # --- End Save Settings ---

                        # Create or update the agent
                        create_agent(
                            # Provider determined by special cases and get_provider_and_model
                            provider=provider,
                            agent_loop=agent_loop_type,
                            # Pass the FINAL determined model name (cleaned or custom)
                            model_name=final_model_name_to_send,
                            api_key=api_key,
                            save_trajectory=save_traj,
                            only_n_most_recent_images=recent_imgs,
                            use_oaicompat=is_oaicompat,  # Set flag if custom model was selected
                            # Pass custom URL only if custom model was selected
                            provider_base_url=(
                                (
                                    azure_endpoint_value
                                    if agent_loop_choice == "AZURE"
                                    else custom_url_value
                                )
                                if is_oaicompat
                                else None
                            ),
                            computer_os=computer_os,
                            computer_provider=computer_provider,
                            computer_name=container_name,
                            computer_api_key=cua_cloud_api_key,
                            verbosity=logging.DEBUG,  # Added verbosity here
                        )

                        if global_agent is None:
                            # Add initial empty assistant message
                            history.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content="Failed to create agent. Check API keys and configuration.",
                                )
                            )
                            yield history
                            return

                            # Add the screenshot handler to the agent's loop if available
                        if global_agent and hasattr(global_agent, "_loop"):
                            print("DEBUG - Adding screenshot handler to agent loop")

                            # Create the screenshot handler with references to UI components
                            screenshot_handler = GradioChatScreenshotHandler(history)

                            # Add the handler to the callback manager if it exists AND is not None
                            if (
                                hasattr(global_agent._loop, "callback_manager")
                                and global_agent._loop.callback_manager is not None
                            ):
                                global_agent._loop.callback_manager.add_handler(screenshot_handler)
                                print(
                                    f"DEBUG - Screenshot handler added to callback manager with history: {id(history)}"
                                )
                            else:
                                # Optional: Log a warning if the callback manager is missing/None for a specific loop
                                print(
                                    f"WARNING - Callback manager not found or is None for loop type: {type(global_agent._loop)}. Screenshot handler not added."
                                )

                        # Stream responses from the agent
                        async for result in global_agent.run(last_user_message):
                            print(f"DEBUG - Agent response ------- START")
                            from pprint import pprint

                            pprint(result)
                            print(f"DEBUG - Agent response ------- END")

                            def generate_gradio_messages():
                                if result.get("content"):
                                    yield gr.ChatMessage(
                                        role="assistant",
                                        content=result.get("content", ""),
                                        metadata=cast(MetadataDict, result.get("metadata", {})),
                                    )
                                else:
                                    outputs = result.get("output", [])
                                    for output in outputs:
                                        if output.get("type") == "message":
                                            content = output.get("content", [])
                                            for content_part in content:
                                                if content_part.get("text"):
                                                    yield gr.ChatMessage(
                                                        role=output.get("role", "assistant"),
                                                        content=content_part.get("text", ""),
                                                        metadata=content_part.get("metadata", {}),
                                                    )
                                        elif output.get("type") == "reasoning":
                                            # if it's openAI, we only have access to a summary of the reasoning
                                            summary_content = output.get("summary", [])
                                            if summary_content:
                                                for summary_part in summary_content:
                                                    if summary_part.get("type") == "summary_text":
                                                        yield gr.ChatMessage(
                                                            role="assistant",
                                                            content=summary_part.get("text", ""),
                                                        )
                                            else:
                                                summary_content = output.get("text", "")
                                                if summary_content:
                                                    yield gr.ChatMessage(
                                                        role="assistant",
                                                        content=summary_content,
                                                    )
                                        elif output.get("type") == "computer_call":
                                            action = output.get("action", {})
                                            action_type = action.get("type", "")
                                            if action_type:
                                                action_title = f"🛠️ Performing {action_type}"
                                                if action.get("x") and action.get("y"):
                                                    action_title += (
                                                        f" at ({action['x']}, {action['y']})"
                                                    )
                                                yield gr.ChatMessage(
                                                    role="assistant",
                                                    content=f"```json\n{json.dumps(action)}\n```",
                                                    metadata={"title": action_title},
                                                )

                            for message in generate_gradio_messages():
                                history.append(message)
                                yield history

                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        # Update with error message
                        history.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
                        yield history

                # Connect the submit button to the process_response function
                submit_event = msg.submit(
                    fn=chat_submit,
                    inputs=[msg, chatbot_history],
                    outputs=[msg, chatbot_history],
                    queue=False,
                ).then(
                    fn=process_response,
                    inputs=[
                        chatbot_history,
                        openai_model_choice,
                        anthropic_model_choice,
                        omni_model_choice,
                        uitars_model_choice,
                        custom_model,
                        agent_loop,
                        save_trajectory,
                        recent_images,
                        provider_base_url,
                        provider_api_key,
                        openai_api_key_input,
                        anthropic_api_key_input,
                        azure_endpoint,
                        azure_deployment,
                        azure_api_key_input,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                    ],
                    outputs=[chatbot_history],
                    queue=True,
                )

                # Clear button functionality
                clear.click(lambda: None, None, chatbot_history, queue=False)

                # Connect cancel button to cancel function
                cancel_button.click(
                    cancel_agent_task,
                    [chatbot_history],
                    [chatbot_history],
                    queue=False,  # Process immediately without queueing
                )

                # Function to update the code display based on configuration and chat history
                def update_code_display(
                    agent_loop,
                    model_choice_val,
                    custom_model_val,
                    chat_history,
                    provider_base_url,
                    recent_images_val,
                    save_trajectory_val,
                    computer_os,
                    computer_provider,
                    container_name,
                    cua_cloud_api_key,
                    azure_endpoint_val,
                    azure_deployment_val,
                ):
                    # Extract messages from chat history
                    messages = []
                    if chat_history:
                        for msg in chat_history:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                messages.append(msg.get("content", ""))

                    # Determine provider and model based on current selection
                    if agent_loop == "AZURE":
                        provider = LLMProvider.OAICOMPAT
                        model_name = azure_deployment_val or MODEL_MAPPINGS["azure"]["default"]
                        provider_url = azure_endpoint_val
                    else:
                        provider, model_name, _ = get_provider_and_model(
                            model_choice_val or custom_model_val or "gpt-4o", agent_loop
                        )
                        provider_url = provider_base_url

                    return generate_python_code(
                        agent_loop,
                        provider,
                        model_name,
                        messages,
                        provider_url,
                        recent_images_val,
                        save_trajectory_val,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                    )

                # Update code display when configuration changes
                agent_loop.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                model_choice.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                custom_model.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                chatbot_history.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                recent_images.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                save_trajectory.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                computer_os.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                computer_provider.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                container_name.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                        azure_endpoint,
                        azure_deployment,
                    ],
                    outputs=[code_display],
                )
                cua_cloud_api_key.change(
                    update_code_display,
                    inputs=[
                        agent_loop,
                        model_choice,
                        custom_model,
                        chatbot_history,
                        provider_base_url,
                        recent_images,
                        save_trajectory,
                        computer_os,
                        computer_provider,
                        container_name,
                        cua_cloud_api_key,
                    ],
                    outputs=[code_display],
                )

    return demo


def test_cua():
    """Standalone function to launch the Gradio app."""
    demo = create_gradio_ui()
    demo.launch(share=False, inbrowser=True)  # Don't create a public link


if __name__ == "__main__":
    test_cua()
