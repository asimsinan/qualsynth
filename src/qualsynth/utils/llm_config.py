"""
LLM Configuration Module

Provides flexible LLM configuration system using Ollama with OpenAI client.
"""
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Default Ollama configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "not-needed"

# Predefined model presets for convenience
MODEL_PRESETS = {
    # Ollama models (default)
    "gemma3-m4": {"model": "gemma3-m4", "api_base": "http://localhost:11434/v1", "temperature": 0.9},
    "gemma3:12b": {"model": "gemma3:12b", "api_base": "http://localhost:11434/v1", "temperature": 0.9},
    "gemma3:4b": {"model": "gemma3:4b", "api_base": "http://localhost:11434/v1", "temperature": 0.9},
    "llama3-70b": {"model": "llama3:70b", "api_base": "http://localhost:11434/v1", "temperature": 0.7},
    "mistral-large": {"model": "mistral-large", "api_base": "http://localhost:11434/v1", "temperature": 0.7},
    "qwen2.5-72b": {"model": "qwen2.5:72b", "api_base": "http://localhost:11434/v1", "temperature": 0.7},
}


def get_llm_config(
    model_name: str,
    temperature: float = 0.7,
    timeout: int = 120,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get LLM configuration for Qualsynth using Ollama with OpenAI client.
    
    Supports Ollama models:
    - gemma3-m4 (M4-optimized, default)
    - gemma3:12b, gemma3:4b
    - llama3:70b
    - mistral-large
    - qwen2.5:72b
    
    Args:
        model_name: Model identifier (e.g., "gemma3-m4", "llama3:70b")
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        api_base: Custom API base URL (defaults to Ollama)
        api_key: Custom API key (defaults to "not-needed")
        **kwargs: Additional model-specific parameters
    
    Returns:
        Configuration dictionary for Qualsynth
    
    Examples:
        >>> # Use gemma3-m4 via Ollama (default)
        >>> config = get_llm_config("gemma3-m4")
        
        >>> # Use llama3:70b via Ollama
        >>> config = get_llm_config("llama3:70b")
        
        >>> # Use custom endpoint
        >>> config = get_llm_config(
        ...     "custom-model",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="custom-key"
        ... )
    """
    # Build base config
    config = {
        "model": model_name,
        "temperature": temperature,
    }
    
    # Priority: 1) explicit api_base, 2) OPENAI_API_BASE env var, 3) default Ollama
    if api_base:
        config["api_base"] = api_base
        config["api_key"] = api_key or DEFAULT_API_KEY
    elif os.getenv("OPENAI_API_BASE"):
        # Use OPENAI_API_BASE if set
        config["api_base"] = os.getenv("OPENAI_API_BASE")
        config["api_key"] = api_key or os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    else:
        # Default to Ollama
        config["api_base"] = DEFAULT_OLLAMA_URL
        config["api_key"] = api_key or DEFAULT_API_KEY
    
    # Add any additional parameters (e.g., max_tokens, top_p)
    config.update(kwargs)
    
    # Return config
    return {
        "config_list": [config],
        "timeout": timeout,
        "cache_seed": None,  # Disable caching for diverse generation
    }


def get_preset_config(preset_name: str, **overrides) -> Dict[str, Any]:
    """
    Get LLM config from predefined preset with optional overrides.
    
    Args:
        preset_name: Name of the preset (e.g., "gpt-4o", "claude-3.5-sonnet")
        **overrides: Parameters to override in the preset
    
    Returns:
        AutoGen-compatible LLM configuration dictionary
    
    Examples:
        >>> # Use GPT-4o preset with higher temperature
        >>> config = get_preset_config("gpt-4o", temperature=0.9)
        
        >>> # Use local Llama3 preset with longer timeout
        >>> config = get_preset_config("llama3-70b", timeout=180)
    
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in MODEL_PRESETS:
        available = list(MODEL_PRESETS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {', '.join(available)}"
        )
    
    preset = MODEL_PRESETS[preset_name].copy()
    preset.update(overrides)
    
    # Extract model_name and pass it as the first argument
    model_name = preset.pop("model")
    return get_llm_config(model_name, **preset)


def list_available_presets() -> List[str]:
    """
    List all available model presets.
    
    Returns:
        List of preset names
    """
    return list(MODEL_PRESETS.keys())


def test_llm_connection(model_name: str, **config_kwargs) -> bool:
    """
    Test connection to an LLM provider via Ollama.
    
    Args:
        model_name: Model identifier or preset name
        **config_kwargs: Additional configuration parameters
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        from openai import OpenAI
        
        # Try to get config
        if model_name in MODEL_PRESETS:
            config = get_preset_config(model_name, **config_kwargs)
        else:
            config = get_llm_config(model_name, **config_kwargs)
        
        # Extract config
        llm_config = config["config_list"][0]
        
        # Priority: 1) config api_base, 2) OPENAI_API_BASE env var, 3) default Ollama
        api_base = llm_config.get("api_base") or os.getenv("OPENAI_API_BASE") or DEFAULT_OLLAMA_URL
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY", "not-needed")
        
        # Create OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=config.get("timeout", 120)
        )
        
        # Make a simple call
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=llm_config.get("temperature", 0.7)
        )
        
        print(f"✓ Connection successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Available presets:")
    for preset in list_available_presets():
        print(f"  - {preset}")
    
    print("\nExample configurations:")
    print("\n1. GPT-4o:")
    print(get_preset_config("gpt-4o"))
    
    print("\n2. Local Llama3:")
    print(get_preset_config("llama3-70b"))
    
    print("\n3. Custom model:")
    print(get_llm_config(
        "my-model",
        api_base="http://localhost:8000/v1",
        api_key="test-key"
    ))

