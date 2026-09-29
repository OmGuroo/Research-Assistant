import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import settings
from app.logger import logger

class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def get_gemini_model() -> ChatGoogleGenerativeAI:
    if not settings.gemini_api_key or settings.gemini_api_key == "your_google_gemini_api_key_here":
        raise LLMError("Gemini API key is missing or not configured.", 401)
    
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_name,
        temperature=settings.gemini_temperature,
        google_api_key=settings.gemini_api_key,
        timeout=30.0
    )

def get_groq_model() -> ChatGroq:
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        raise LLMError("Groq API key is missing or not configured.", 401)
        
    return ChatGroq(
        model_name=settings.groq_model_name,
        temperature=settings.groq_temperature,
        groq_api_key=settings.groq_api_key,
        timeout=30.0
    )

def _extract_usage(response) -> tuple[int, int]:
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            return response.usage_metadata.get("input_tokens", 0), response.usage_metadata.get("output_tokens", 0)
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                return token_usage.get("prompt_tokens", 0), token_usage.get("completion_tokens", 0)
    except Exception:
        pass
    return 0, 0

async def _invoke_with_retry(model, prompt: str, is_structured: bool = False, max_retries: int = 2):
    for attempt in range(max_retries):
        try:
            return await model.ainvoke(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            if attempt < max_retries - 1 and ("429" in error_msg or "rate limit" in error_msg or "502" in error_msg or "503" in error_msg or "timeout" in error_msg):
                logger.warning(f"LLM transient error: {error_msg}. Retrying {attempt + 1}/{max_retries}...")
                await asyncio.sleep(2 ** attempt)
                continue
            _handle_llm_exception(e)

async def generate_response(prompt: str) -> tuple[str, int, int]:
    try:
        if settings.llm_provider.lower() == "groq":
            model = get_groq_model()
        elif settings.llm_provider.lower() == "gemini":
            model = get_gemini_model()
        else:
            raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}", 400)
            
        response = await _invoke_with_retry(model, prompt)
        in_tok, out_tok = _extract_usage(response)
        return response.content, in_tok, out_tok
    except LLMError:
        raise
    except Exception as e:
        _handle_llm_exception(e)

async def generate_structured_response(prompt: str, schema: type) -> tuple[any, int, int]:
    try:
        if settings.llm_provider.lower() == "groq":
            model = get_groq_model()
        elif settings.llm_provider.lower() == "gemini":
            model = get_gemini_model()
        else:
            raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}", 400)
            
        structured_model = model.with_structured_output(schema, include_raw=True)
        response = await _invoke_with_retry(structured_model, prompt, is_structured=True)
        
        parsed = response.get("parsed")
        raw = response.get("raw")
        in_tok, out_tok = _extract_usage(raw)
        
        if not parsed:
            raise LLMError("LLM failed to output the requested structured schema.", 500)
            
        return parsed, in_tok, out_tok
    except LLMError:
        raise
    except Exception as e:
        _handle_llm_exception(e)

def _handle_llm_exception(e: Exception):
    error_msg = str(e).lower()
    logger.error(f"LLM Exception: {str(e)}")
    if "permission_denied" in error_msg or "403" in error_msg:
        raise LLMError("Permission denied. Check if the API key is valid and has access.", 403)
    elif "resource_exhausted" in error_msg or "429" in error_msg or "rate limit" in error_msg:
        raise LLMError("Quota exceeded or rate limit reached.", 429)
    elif "not_found" in error_msg or "404" in error_msg:
        raise LLMError("The specified model was not found or is unavailable.", 404)
    elif "invalid_argument" in error_msg or "400" in error_msg or "invalid api key" in error_msg:
        raise LLMError("Invalid request or API key sent to the provider.", 400)
    elif "401" in error_msg or "unauthorized" in error_msg:
        raise LLMError("Unauthorized. Invalid API key.", 401)
    else:
        raise LLMError("An unexpected error occurred with the LLM API.", 502)
