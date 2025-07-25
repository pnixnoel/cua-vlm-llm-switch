import sys
from pathlib import Path
import importlib.util
import types
import pytest

# Import AzureOpenAIClient directly from its file to avoid heavy package imports
project_root = Path(__file__).resolve().parents[1]
clients_dir = project_root / "libs/python/agent/agent/providers/omni/clients"
azure_path = clients_dir / "azure.py"
base_path = clients_dir / "base.py"

# Create stub packages so relative imports work without executing package __init__ files
pkg_names = [
    "agent",
    "agent.providers",
    "agent.providers.omni",
    "agent.providers.omni.clients",
]
for name in pkg_names:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(clients_dir)] if name.endswith("clients") else []
        sys.modules[name] = module

spec_base = importlib.util.spec_from_file_location(
    "agent.providers.omni.clients.base", base_path
)
base_module = importlib.util.module_from_spec(spec_base)
sys.modules[spec_base.name] = base_module
spec_base.loader.exec_module(base_module)  # type: ignore

spec = importlib.util.spec_from_file_location(
    "agent.providers.omni.clients.azure", azure_path
)
azure_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = azure_module
spec.loader.exec_module(azure_module)  # type: ignore

AzureOpenAIClient = azure_module.AzureOpenAIClient


class DummyResponse:
    def __init__(self, status: int, json_data):
        self.status = status
        self._json = json_data

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, response):
        self.response = response
        self.url = None
        self.headers = None
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def post(self, url, headers=None, json=None):
        self.url = url
        self.headers = headers
        self.payload = json
        return self.response


@pytest.mark.asyncio
async def test_run_interleaved_posts_to_azure(monkeypatch):
    expected = {"id": "1", "choices": [{"message": {"content": "hi"}}]}
    session = DummySession(DummyResponse(200, expected))
    monkeypatch.setattr("aiohttp.ClientSession", lambda: session)

    client = AzureOpenAIClient(
        api_key="key",
        provider_base_url="https://example.openai.azure.com",
        deployment_name="test-deploy",
        api_version="2024-05-01-preview",
    )

    messages = [{"role": "user", "content": "Hello"}]
    result = await client.run_interleaved(messages, "You are a bot")

    assert session.url == (
        "https://example.openai.azure.com/openai/deployments/test-deploy/chat/completions"
        "?api-version=2024-05-01-preview"
    )
    assert result == expected
    assert session.payload["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_run_interleaved_error(monkeypatch):
    session = DummySession(DummyResponse(400, {"error": {"message": "bad"}}))
    monkeypatch.setattr("aiohttp.ClientSession", lambda: session)

    client = AzureOpenAIClient(
        api_key="key",
        provider_base_url="https://example.openai.azure.com",
        deployment_name="test-deploy",
    )

    with pytest.raises(Exception, match="Azure OpenAI API error: bad"):
        await client.run_interleaved([{"role": "user", "content": "hi"}], "sys")

