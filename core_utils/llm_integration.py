import openai


class LLMIntegration:
    """Integration with various LLM providers"""
    
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self.kwargs = kwargs
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == 'openai':
            return openai.OpenAI(**self.kwargs)
        elif self.provider == 'anthropic':
            # Would initialize Anthropic client
            pass
        elif self.provider == 'local':
            # For local models like Ollama
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_response(self, query: str, context: str, 
                         max_tokens: int = None, temperature: float = 0.7) -> str:
        """Generate response using the LLM"""
        prompt = self._build_prompt(query, context)
        
        if self.provider == 'openai':
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
        else:
            raise NotImplementedError(f"Response generation not implemented for {self.provider}")
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM"""
        return f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
