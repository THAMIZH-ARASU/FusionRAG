import os
import time
import requests
from typing import Dict, Any, Optional
from config import Config

class LLMIntegration:
    """Enhanced LLM integration supporting multiple providers"""
    
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self.kwargs = kwargs
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == 'openai':
            import openai
            return openai.OpenAI(api_key=Config.OPENAI_API_KEY, **self.kwargs)
        elif self.provider == 'groq':
            return self._init_groq_client()
        elif self.provider == 'google':
            return self._init_google_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _init_groq_client(self):
        """Initialize Groq client"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return {
            'api_key': Config.GROQ_API_KEY,
            'base_url': 'https://api.groq.com/openai/v1'
        }
    
    def _init_google_client(self):
        """Initialize Google AI client"""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return {
            'api_key': Config.GOOGLE_API_KEY,
            'base_url': 'https://generativelanguage.googleapis.com/v1beta'
        }
    
    def generate_response(self, query: str, context: str, 
                         max_tokens: int = None, temperature: float = 0.7) -> str:
        """Generate response using the LLM"""
        prompt = self._build_prompt(query, context)
        
        if self.provider == 'openai':
            return self._generate_openai_response(prompt, max_tokens, temperature)
        elif self.provider == 'groq':
            return self._generate_groq_response(prompt, max_tokens, temperature)
        elif self.provider == 'google':
            return self._generate_google_response(prompt, max_tokens, temperature)
        else:
            raise NotImplementedError(f"Response generation not implemented for {self.provider}")
    
    def _generate_openai_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.kwargs.get('model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _generate_groq_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Groq API"""
        headers = {
            'Authorization': f'Bearer {self.client["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'llama3-8b-8192',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant that answers questions based on the provided context.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens or 1000,
            'temperature': temperature
        }
        
        response = requests.post(
            f"{self.client['base_url']}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.text}")
        
        return response.json()['choices'][0]['message']['content']
    
    def _generate_google_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Google AI API"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'contents': [
                {
                    'parts': [
                        {'text': 'You are a helpful assistant that answers questions based on the provided context.'},
                        {'text': prompt}
                    ]
                }
            ],
            'generationConfig': {
                'maxOutputTokens': max_tokens or 1000,
                'temperature': temperature
            }
        }
        
        response = requests.post(
            f"{self.client['base_url']}/models/gemini-pro:generateContent?key={self.client['api_key']}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Google AI API error: {response.text}")
        
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM"""
        return f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
