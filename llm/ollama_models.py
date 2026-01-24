"""
Ollama Model Wrapper

This module provides a wrapper for Ollama API to run LLM models.
Useful for running large models without loading them directly into memory.
"""

import requests
import json
from typing import Optional


class OllamaModel:
    """
    Wrapper for Ollama API to run LLM models.
    """
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", logger=None):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Name of the model in Ollama (e.g., "mistral:7b")
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            logger: Optional logger
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.logger = logger
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                if self.logger:
                    self.logger.info(f"✅ Connected to Ollama at {self.base_url}")
            else:
                raise ConnectionError(f"Ollama API returned status {response.status_code}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.")
    
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: User prompt
            system: Optional system message
            **kwargs: Additional parameters (temperature, top_p, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Default parameters for classification tasks
        default_params = {
            "temperature": 0.0,  # Deterministic for classification
            "top_p": 0.9,
        }
        default_params.update(kwargs)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **default_params
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"Ollama API request error: {e}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ollama generation error: {e}")
            raise
