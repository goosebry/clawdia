"""Google Gemini API client for Ella AI."""
from __future__ import annotations

import logging
from typing import Any, Dict, List
import asyncio

import google.genai as genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Async client for Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str):
        """Initialize Gemini client."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate chat completion using Gemini."""
        try:
            # Convert OpenAI-style messages to Gemini format
            contents = self._convert_messages_to_contents(messages)
            
            # Generate response
            response = await self._generate_async(contents, max_tokens, temperature)
            
            return response
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> List[genai.types.Content]:
        """Convert OpenAI-style messages to Gemini Content format."""
        contents = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map roles
            if role == "system":
                # System messages become user messages with instruction prefix
                contents.append(genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=f"System instruction: {content}")]
                ))
            elif role == "user":
                contents.append(genai.types.Content(
                    role="user", 
                    parts=[genai.types.Part(text=content)]
                ))
            elif role == "assistant":
                contents.append(genai.types.Content(
                    role="model",
                    parts=[genai.types.Part(text=content)]
                ))
        
        return contents
    
    async def _generate_async(self, contents: List[genai.types.Content], max_tokens: int, temperature: float) -> str:
        """Generate response asynchronously."""
        try:
            # Run sync API in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                contents,
                max_tokens,
                temperature
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_sync(self, contents: List[genai.types.Content], max_tokens: int, temperature: float) -> str:
        """Generate response synchronously."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Gemini sync generation error: {e}")
            raise


# Global client instance
_gemini_client: GeminiClient | None = None


def get_gemini_client(api_key: str | None = None, model_name: str | None = None) -> GeminiClient:
    """Get or create global Gemini client. Fetch settings if not provided."""
    global _gemini_client
    
    if _gemini_client is None:
        from ella.config import get_settings
        settings = get_settings()
        final_api_key = api_key or settings.google_api_key
        final_model_name = model_name or settings.gemini_model
        
        if not final_api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")
            
        _gemini_client = GeminiClient(final_api_key, final_model_name)
    
    return _gemini_client