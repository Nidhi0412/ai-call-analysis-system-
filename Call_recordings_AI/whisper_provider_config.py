#!/usr/bin/env python3
"""
Whisper Provider Configuration
=============================

This module provides configuration and factory methods for different Whisper providers
(OpenAI, Groq) to make switching between providers seamless.
"""

import os
from typing import Dict, Any, Optional
from enum import Enum

class WhisperProvider(Enum):
    """Available Whisper providers"""
    OPENAI = "openai"
    GROQ = "groq"

class WhisperConfig:
    """Configuration for Whisper providers"""
    
    # Provider configurations
    PROVIDERS = {
        WhisperProvider.OPENAI: {
            "base_url": "https://api.openai.com/v1",
            "models": {
                "whisper-1": "whisper-1",
                "whisper-1-large": "whisper-1-large",
                "whisper-1-large-v2": "whisper-1-large-v2"
            },
            "default_model": "whisper-1",
            "cost_per_minute": 0.006,  # $0.006 per minute
            "api_key_env": "OPENAI_API_KEY"
        },
        WhisperProvider.GROQ: {
            "base_url": "https://api.groq.com/openai/v1",
            "models": {
                "whisper-large-v3-turbo": "whisper-large-v3-turbo",
                "distil-whisper-large-v3-en": "distil-whisper-large-v3-en",
                "whisper-large-v3": "whisper-large-v3"
            },
            "default_model": "whisper-large-v3-turbo",
            "cost_per_minute": 0.00067,  # $0.04 per hour = $0.00067 per minute
            "api_key_env": "GROQ_API_KEY"
        }
    }
    
    @classmethod
    def get_provider_config(cls, provider: WhisperProvider) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        return cls.PROVIDERS.get(provider, {})
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def get_provider_models(cls, provider: WhisperProvider) -> Dict[str, str]:
        """Get available models for a provider"""
        config = cls.get_provider_config(provider)
        return config.get("models", {})
    
    @classmethod
    def get_default_model(cls, provider: WhisperProvider) -> str:
        """Get default model for a provider"""
        config = cls.get_provider_config(provider)
        return config.get("default_model", "whisper-1")
    
    @classmethod
    def get_cost_per_minute(cls, provider: WhisperProvider) -> float:
        """Get cost per minute for a provider"""
        config = cls.get_provider_config(provider)
        return config.get("cost_per_minute", 0.006)
    
    @classmethod
    def get_api_key_env(cls, provider: WhisperProvider) -> str:
        """Get environment variable name for API key"""
        config = cls.get_provider_config(provider)
        return config.get("api_key_env", "OPENAI_API_KEY")

class WhisperProviderFactory:
    """Factory for creating Whisper service instances"""
    
    @staticmethod
    def create_service(provider: WhisperProvider = None, 
                      model: str = None,
                      api_key: str = None,
                      **kwargs) -> Any:
        """
        Create a Whisper service instance
        
        Args:
            provider: Whisper provider (default: auto-detect)
            model: Model to use (default: provider's default)
            api_key: API key (default: from environment)
            **kwargs: Additional service parameters
            
        Returns:
            Whisper service instance
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = WhisperProviderFactory._auto_detect_provider()
        
        # Get provider configuration
        config = WhisperConfig.get_provider_config(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Set default model if not specified
        if model is None:
            model = config["default_model"]
        
        # Get API key from config first, then environment if not provided
        if api_key is None:
            api_key_env = config["api_key_env"]
            # Try to get API key from config first
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from config import CONFIG
                app_env = os.getenv('NODE_ENV', 'Development')
                env_config = CONFIG.get(app_env, {})
                api_key = env_config.get(api_key_env)
            except ImportError:
                # Fallback to environment variables if config not available
                api_key = os.getenv(api_key_env)
            
            if not api_key:
                raise ValueError(f"API key not found. Set {api_key_env} in config or environment variable.")
        
        # Create service based on provider
        if provider == WhisperProvider.OPENAI:
            from transcription_with_speakers import TranscriptionWithSpeakersService
            return TranscriptionWithSpeakersService(
                api_key=api_key,
                model=model,
                **kwargs
            )
        elif provider == WhisperProvider.GROQ:
            from Call_recordings_AI.groq_whisper_service import GroqWhisperService
            return GroqWhisperService(
                api_key=api_key,
                model=model,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _auto_detect_provider() -> WhisperProvider:
        """Auto-detect the best available provider"""
        # Try to get API keys from config first
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import CONFIG
            app_env = os.getenv('NODE_ENV', 'Development')
            env_config = CONFIG.get(app_env, {})
            
            # Check for Groq API key first (preferred for cost/performance)
            if env_config.get("GROQ_API_KEY"):
                return WhisperProvider.GROQ
            
            # Fall back to OpenAI
            if env_config.get("OPENAI_API_KEY"):
                return WhisperProvider.OPENAI
                
        except ImportError:
            # Fallback to environment variables if config not available
            pass
        
        # Check environment variables as fallback
        if os.getenv("GROQ_API_KEY"):
            return WhisperProvider.GROQ
        
        if os.getenv("OPENAI_API_KEY"):
            return WhisperProvider.OPENAI
        
        # Default to Groq if no keys found (user needs to set one)
        return WhisperProvider.GROQ
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about all available providers"""
        info = {}
        
        # Try to get API keys from config first
        config_keys = {}
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import CONFIG
            app_env = os.getenv('NODE_ENV', 'Development')
            env_config = CONFIG.get(app_env, {})
            config_keys = env_config
        except ImportError:
            pass
        
        for provider in WhisperProvider:
            config = WhisperConfig.get_provider_config(provider)
            api_key_env = config.get("api_key_env", "")
            
            # Check availability from config first, then environment
            available = bool(config_keys.get(api_key_env)) or bool(os.getenv(api_key_env))
            
            info[provider.value] = {
                "name": provider.value.title(),
                "models": list(config.get("models", {}).keys()),
                "default_model": config.get("default_model"),
                "cost_per_minute": config.get("cost_per_minute"),
                "api_key_env": api_key_env,
                "available": available
            }
        
        return info

# Convenience functions
def create_whisper_service(provider: str = None, **kwargs) -> Any:
    """Convenience function to create Whisper service"""
    provider_enum = None
    if provider:
        try:
            provider_enum = WhisperProvider(provider.lower())
        except ValueError:
            raise ValueError(f"Invalid provider: {provider}. Available: {[p.value for p in WhisperProvider]}")
    
    return WhisperProviderFactory.create_service(provider_enum, **kwargs)

def get_provider_comparison() -> Dict[str, Any]:
    """Get detailed comparison of all providers"""
    comparison = {}
    
    for provider in WhisperProvider:
        config = WhisperConfig.get_provider_config(provider)
        comparison[provider.value] = {
            "cost_per_minute": config.get("cost_per_minute"),
            "cost_per_hour": config.get("cost_per_minute", 0) * 60,
            "models": list(config.get("models", {}).keys()),
            "default_model": config.get("default_model"),
            "api_key_required": config.get("api_key_env"),
            "api_key_available": bool(os.getenv(config.get("api_key_env", ""))),
            "base_url": config.get("base_url")
        }
    
    return comparison

# Example usage
if __name__ == "__main__":
    print("Whisper Provider Configuration")
    print("=" * 40)
    
    # Show provider information
    info = WhisperProviderFactory.get_provider_info()
    for provider, details in info.items():
        print(f"\n{provider.upper()}:")
        print(f"  Models: {', '.join(details['models'])}")
        print(f"  Default: {details['default_model']}")
        print(f"  Cost: ${details['cost_per_minute']:.4f}/minute")
        print(f"  API Key: {'✓ Available' if details['available'] else '✗ Not set'}")
    
    # Show cost comparison
    print("\nCost Comparison (per hour):")
    comparison = get_provider_comparison()
    for provider, details in comparison.items():
        print(f"  {provider}: ${details['cost_per_hour']:.4f}")
    
    # Auto-detect best provider
    best_provider = WhisperProviderFactory._auto_detect_provider()
    print(f"\nRecommended provider: {best_provider.value}")
