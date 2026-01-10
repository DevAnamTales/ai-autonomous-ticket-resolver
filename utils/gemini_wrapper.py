from langchain.llms.base import LLM
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class GeminiLangChainWrapper(LLM):
    """
    LangChain-compatible wrapper for gemini-2.0-flash-lite
    """
    model_name: str = "llama-3.1-70b-versatile"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        """
        Generate content using Gemini API
        """
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature
                }
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

