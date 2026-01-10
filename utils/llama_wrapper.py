from langchain.llms.base import LLM
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class LlamaLangChainWrapper(LLM):
    """
    LangChain-compatible wrapper for Groq LLaMA
    """
    model_name: str = "llama-3.1-8b-instant"  # or llama-3.1-70b-versatile
    
    @property
    def _llm_type(self) -> str:
        return "llama"
    
    def _call(self, prompt: str, stop=None) -> str:
        """
        Generate content using Groq LLaMA API
        """
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )
            
            # ✅ FIX: Access content as attribute, not dictionary
            return response.choices[0].message.content
            
        except Exception as e:
            return f"LLaMA Error: {str(e)}"
