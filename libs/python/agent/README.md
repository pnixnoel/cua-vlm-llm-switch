<div align="center">
<h1>
  <div class="image-wrapper" style="display: inline-block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" alt="logo" height="150" srcset="../../img/logo_white.png" style="display: block; margin: auto;">
      <source media="(prefers-color-scheme: light)" alt="logo" height="150" srcset="../../img/logo_black.png" style="display: block; margin: auto;">
      <img alt="Shows my svg">
    </picture>
  </div>

  [![Python](https://img.shields.io/badge/Python-333333?logo=python&logoColor=white&labelColor=333333)](#)
  [![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0)](#)
  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.com/invite/mVnXXpdE85)
  [![PyPI](https://img.shields.io/pypi/v/cua-computer?color=333333)](https://pypi.org/project/cua-computer/)
</h1>
</div>

**cua-agent** is a general Computer-Use framework for running multi-app agentic workflows targeting macOS and Linux sandbox created with Cua, supporting local (Ollama) and cloud model providers (OpenAI, Azure OpenAI, Anthropic, Groq, DeepSeek, Qwen).

### Get started with Agent

<div align="center">
    <img src="../../img/agent.png"/>
</div>

## Install

```bash
pip install "cua-agent[all]"

# or install specific loop providers
pip install "cua-agent[openai]" # OpenAI Cua Loop
pip install "cua-agent[anthropic]" # Anthropic Cua Loop
pip install "cua-agent[uitars]"    # UI-Tars support
pip install "cua-agent[omni]" # Cua Loop based on OmniParser (includes Ollama for local models)
pip install "cua-agent[ui]" # Gradio UI for the agent
pip install "cua-agent[uitars-mlx]" # MLX UI-Tars support
```

## Run

```bash
async with Computer() as macos_computer:
  # Create agent with loop and provider
  agent = ComputerAgent(
      computer=macos_computer,
      loop=AgentLoop.OPENAI,
      model=LLM(provider=LLMProvider.OPENAI)
      # or
      # loop=AgentLoop.ANTHROPIC,
      # model=LLM(provider=LLMProvider.ANTHROPIC)
      # or
      # loop=AgentLoop.OMNI,
      # model=LLM(provider=LLMProvider.OLLAMA, name="gemma3")
      # or
      # loop=AgentLoop.UITARS,
      # model=LLM(provider=LLMProvider.OAICOMPAT, name="ByteDance-Seed/UI-TARS-1.5-7B", provider_base_url="https://**************.us-east-1.aws.endpoints.huggingface.cloud/v1")
  )

  tasks = [
      "Look for a repository named trycua/cua on GitHub.",
      "Check the open issues, open the most recent one and read it.",
      "Clone the repository in users/lume/projects if it doesn't exist yet.",
      "Open the repository with an app named Cursor (on the dock, black background and white cube icon).",
      "From Cursor, open Composer if not already open.",
      "Focus on the Composer text area, then write and submit a task to help resolve the GitHub issue.",
  ]

  for i, task in enumerate(tasks):
      print(f"\nExecuting task {i}/{len(tasks)}: {task}")
      async for result in agent.run(task):
          print(result)

      print(f"\n✅ Task {i+1}/{len(tasks)} completed: {task}")
```

Refer to these notebooks for step-by-step guides on how to use the Computer-Use Agent (CUA):

- [Agent Notebook](../../notebooks/agent_nb.ipynb) - Complete examples and workflows

## Using the Gradio UI

The agent includes a Gradio-based user interface for easier interaction.

<div align="center">
    <img src="../../img/agent_gradio_ui.png"/>
</div>

To use it:

```bash
# Install with Gradio support
pip install "cua-agent[ui]"
```

### Create a simple launcher script

```python
# launch_ui.py
from agent.ui.gradio.app import create_gradio_ui

app = create_gradio_ui()
app.launch(share=False)
```

### Setting up API Keys

For the Gradio UI to show available models, you need to set API keys as environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY=your_openai_key_here

# For Anthropic models
export ANTHROPIC_API_KEY=your_anthropic_key_here

# Launch with both keys set
OPENAI_API_KEY=your_key ANTHROPIC_API_KEY=your_key python launch_ui.py
```

Without these environment variables, the UI will show "No models available" for the corresponding providers, but you can still use local models with the OMNI loop provider.

### Using Local Models

You can use local models with the OMNI loop provider by selecting "Custom model..." from the dropdown. The default provider URL is set to `http://localhost:1234/v1` which works with LM Studio. 

If you're using a different local model server:
- vLLM: `http://localhost:8000/v1`
- LocalAI: `http://localhost:8080/v1`
- Ollama with OpenAI compat API: `http://localhost:11434/v1`

The Gradio UI provides:
- Selection of different agent loops (OpenAI, Anthropic, OMNI)
- Model selection for each provider
- Configuration of agent parameters
- Chat interface for interacting with the agent

### Using UI-TARS

The UI-TARS models are available in two forms:

1. **MLX UI-TARS models** (Default): These models run locally using MLXVLM provider
   - `mlx-community/UI-TARS-1.5-7B-4bit` (default) - 4-bit quantized version
   - `mlx-community/UI-TARS-1.5-7B-6bit` - 6-bit quantized version for higher quality

   ```python
   agent = ComputerAgent(
       computer=macos_computer,
       loop=AgentLoop.UITARS,
       model=LLM(provider=LLMProvider.MLXVLM, name="mlx-community/UI-TARS-1.5-7B-4bit")
   )
   ```

2. **OpenAI-compatible UI-TARS**: For using the original ByteDance model
   - If you want to use the original ByteDance UI-TARS model via an OpenAI-compatible API, follow the [deployment guide](https://github.com/bytedance/UI-TARS/blob/main/README_deploy.md)
   - This will give you a provider URL like `https://**************.us-east-1.aws.endpoints.huggingface.cloud/v1` which you can use in the code or Gradio UI:

   ```python 
   agent = ComputerAgent(
       computer=macos_computer,
       loop=AgentLoop.UITARS,
       model=LLM(provider=LLMProvider.OAICOMPAT, name="tgi", 
                provider_base_url="https://**************.us-east-1.aws.endpoints.huggingface.cloud/v1")
   )
   ```

## Agent Loops

The `cua-agent` package provides three agent loops variations, based on different CUA models providers and techniques:

| Agent Loop | Supported Models | Description | Set-Of-Marks |
|:-----------|:-----------------|:------------|:-------------|
| `AgentLoop.OPENAI` | • `computer_use_preview` | Use OpenAI Operator CUA model | Not Required |
| `AgentLoop.ANTHROPIC` | • `claude-3-5-sonnet-20240620`<br>• `claude-3-7-sonnet-20250219` | Use Anthropic Computer-Use | Not Required |
| `AgentLoop.UITARS` | • `mlx-community/UI-TARS-1.5-7B-4bit` (default)<br>• `mlx-community/UI-TARS-1.5-7B-6bit`<br>• `ByteDance-Seed/UI-TARS-1.5-7B` (via openAI-compatible endpoint) | Uses UI-TARS models with MLXVLM (default) or OAICOMPAT providers | Not Required |
| `AgentLoop.OMNI` | • `claude-3-5-sonnet-20240620`<br>• `claude-3-7-sonnet-20250219`<br>• `gpt-4.5-preview`<br>• `gpt-4o`<br>• `gpt-4`<br>• `phi4`<br>• `phi4-mini`<br>• `gemma3`<br>• `...`<br>• `Any Ollama or OpenAI-compatible model` | Use OmniParser for element pixel-detection (SoM) and any VLMs for UI Grounding and Reasoning | OmniParser |

## AgentResponse
The `AgentResponse` class represents the structured output returned after each agent turn. It contains the agent's response, reasoning, tool usage, and other metadata. The response format aligns with the new [OpenAI Agent SDK specification](https://platform.openai.com/docs/api-reference/responses) for better consistency across different agent loops.

```python
async for result in agent.run(task):
  print("Response ID: ", result.get("id"))

  # Print detailed usage information
  usage = result.get("usage")
  if usage:
      print("\nUsage Details:")
      print(f"  Input Tokens: {usage.get('input_tokens')}")
      if "input_tokens_details" in usage:
          print(f"  Input Tokens Details: {usage.get('input_tokens_details')}")
      print(f"  Output Tokens: {usage.get('output_tokens')}")
      if "output_tokens_details" in usage:
          print(f"  Output Tokens Details: {usage.get('output_tokens_details')}")
      print(f"  Total Tokens: {usage.get('total_tokens')}")

  print("Response Text: ", result.get("text"))

  # Print tools information
  tools = result.get("tools")
  if tools:
      print("\nTools:")
      print(tools)

  # Print reasoning and tool call outputs
  outputs = result.get("output", [])
  for output in outputs:
      output_type = output.get("type")
      if output_type == "reasoning":
          print("\nReasoning Output:")
          print(output)
      elif output_type == "computer_call":
          print("\nTool Call Output:")
          print(output)
```

**Note on Settings Persistence:**

*   The Gradio UI automatically saves your configuration (Agent Loop, Model Choice, Custom Base URL, Save Trajectory state, Recent Images count) to a file named `.gradio_settings.json` in the project's root directory when you successfully run a task.
*   This allows your preferences to persist between sessions.
*   API keys entered into the custom provider field are **not** saved in this file for security reasons. Manage API keys using environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or a `.env` file.
*   It's recommended to add `.gradio_settings.json` to your `.gitignore` file.
