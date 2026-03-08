"""LLM service for generating responses using Mistral API."""

from typing import List, Dict, Any, Optional
from mistralai import Mistral


class LLMService:
    """Service for interacting with Mistral LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-medium-latest"
    ) -> None:
        """Initialize LLM service.
        
        Args:
            api_key: Mistral API key
            model: LLM model name (default: mistral-medium-latest)
        """
        self.api_key = api_key
        self.model = model
        self.client: Optional[Mistral] = None

    def initialize(self) -> None:
        """Initialize Mistral client.
        
        Raises:
            ValueError: If API key is invalid or client initialization fails
        """
        self.client = Mistral(self.api_key)
        if not self.client:
            raise ValueError("Failed to initialize Mistral client")
        else:
            print("Mistral client initialized")

    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using LLM with optional context.
        
        Args:
            user_message: User's question/message
            context: Optional context text (e.g., retrieved document chunks)
            system_prompt: Optional system prompt
        
        Returns:
            Dict[str, Any]: Response with:
                - text: Generated response text
                - model: Model used
                - usage: Token usage information
        
        Raises:
            ValueError: If generation fails
        """
        messages = self.format_messages(user_message, system_prompt)
        response = self.client.chat.complete(messages=messages, model = self.model)
        if not response:
            raise ValueError("Failed to generate response")
        else:
            return response

    def format_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Format messages for Mistral API.
        
        Args:
            user_message: User's question/message
            context: Optional context text
            system_prompt: Optional system prompt
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages