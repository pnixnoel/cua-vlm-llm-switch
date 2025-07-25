"""Azure OpenAI client implementation."""

import os
import logging
from typing import Any, Dict, List, Optional
import aiohttp
import re
from .base import BaseOmniClient

logger = logging.getLogger(__name__)


class AzureOpenAIClient(BaseOmniClient):
    """Azure OpenAI API client implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        provider_base_url: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        """Initialize the Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key
            model: Deployment name to use
            provider_base_url: Azure endpoint base URL (e.g. https://my-resource.openai.azure.com)
            api_version: Azure OpenAI API version
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        super().__init__(api_key=api_key, model=model)
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No Azure OpenAI API key provided")

        if not provider_base_url:
            provider_base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if not provider_base_url:
            raise ValueError("Azure OpenAI endpoint must be provided")
        self.provider_base_url = provider_base_url.rstrip("/")
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _extract_base64_image(self, text: str) -> Optional[str]:
        """Extract base64 image data from an HTML img tag."""
        pattern = r'data:image/[^;]+;base64,([^"]+)'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    async def run_interleaved(
        self, messages: List[Dict[str, Any]], system: str, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run interleaved chat completion."""
        headers = {"Content-Type": "application/json", "api-key": self.api_key}

        final_messages = [{"role": "system", "content": system}]
        for item in messages:
            if isinstance(item, dict):
                if isinstance(item.get("content"), list):
                    final_messages.append(item)
                else:
                    base64_img = self._extract_base64_image(item["content"])
                    if base64_img:
                        final_messages.append(
                            {
                                "role": item["role"],
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                                    }
                                ],
                            }
                        )
                    else:
                        final_messages.append(
                            {
                                "role": item["role"],
                                "content": [{"type": "text", "text": item["content"]}],
                            }
                        )
            else:
                base64_img = self._extract_base64_image(item)
                if base64_img:
                    final_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                                }
                            ],
                        }
                    )
                else:
                    final_messages.append({"role": "user", "content": [{"type": "text", "text": item}]})

        payload = {"messages": final_messages, "temperature": self.temperature}
        payload["max_tokens"] = max_tokens or self.max_tokens

        endpoint_url = f"{self.provider_base_url}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, headers=headers, json=payload) as response:
                response_json = await response.json()
                if response.status != 200:
                    error_msg = response_json.get("error", {}).get("message", str(response_json))
                    logger.error(f"Error in Azure OpenAI API call: {error_msg}")
                    raise Exception(f"Azure OpenAI API error: {error_msg}")
                return response_json
