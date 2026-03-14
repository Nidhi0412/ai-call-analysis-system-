#!/usr/bin/env python3
"""
Local Models Configuration
==========================

Configuration for local open-source alternatives to OpenAI services
"""

import os
from typing import Dict, Optional
from enum import Enum

# Enhanced language support
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'hin': 'Hindi',
    'es': 'Spanish',
    'spa': 'Spanish',
    'fr': 'French',
    'fra': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'ur': 'Urdu'
}

# Language detection confidence thresholds
LANGUAGE_DETECTION_THRESHOLD = 0.7
HINGLISH_DETECTION_THRESHOLD = 0.6

# Enhanced transcription settings
ENHANCED_TRANSCRIPTION_SETTINGS = {
    'force_hindi_detection': True,
    'enhanced_prompt': True,
    'segment_combining': True,
    'min_segment_length': 2.0,
    'max_segment_length': 20.0,
    'use_hinglish_prompts': True,
    'auto_translate_non_english': True
}

# Audio preprocessing settings
AUDIO_PREPROCESSING = {
    'target_sr': 16000,
    'denoise_threshold': 0.1,
    'silence_threshold': 0.01,
    'min_silence_duration': 0.5,
    'quality_analysis': True
}

class ModelProvider(Enum):
    """Available model providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCALAI = "localai"
    WHISPER_CPP = "whisper_cpp"
    FASTER_WHISPER = "faster_whisper"
    WHISPERX = "whisperx"

# LLM Configuration
LLM_CONFIG = {
    'ollama': {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama3.1:8b',  # Changed from llama3.2:3b to match installed model
        'timeout': 180,
        'max_tokens': 1000,
        'temperature': 0.1
    },
    'localai': {
        'base_url': 'http://localhost:8080',
        'model_name': 'gpt-3.5-turbo',
        'timeout': 60,
        'max_tokens': 2000,
        'temperature': 0.1
    }
}

class LocalModelsConfig:
    """
    Configuration for local model alternatives
    """
    
    def __init__(self):
        # Default to local models for local system
        self.transcription_provider = os.getenv('TRANSCRIPTION_PROVIDER', ModelProvider.FASTER_WHISPER.value)
        self.llm_provider = os.getenv('LLM_PROVIDER', ModelProvider.OLLAMA.value)
        
        # Local model configurations
        self.local_config = {
            # Ollama configuration
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'models': {
                    'transcription': 'whisper',
                    'analysis': 'llama3.1:8b',
                    'translation': 'llama3.1:8b'
                }
            },
            
            # LocalAI configuration
            'localai': {
                'base_url': os.getenv('LOCALAI_BASE_URL', 'http://localhost:8080'),
                'models': {
                    'transcription': 'whisper',
                    'analysis': 'llama3.1:8b',
                    'translation': 'llama3.1:8b'
                }
            },
            
            # Whisper.cpp configuration
            'whisper_cpp': {
                'model_path': os.getenv('WHISPER_MODEL_PATH', '/models/ggml-base.bin'),
                'model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),  # tiny, base, small, medium, large
                'language': os.getenv('WHISPER_LANGUAGE', 'auto')
            },
            
            # Faster Whisper configuration
            'faster_whisper': {
                'model_size': os.getenv('FASTER_WHISPER_MODEL_SIZE', 'base'),
                'device': os.getenv('FASTER_WHISPER_DEVICE', 'cpu'),
                'compute_type': os.getenv('FASTER_WHISPER_COMPUTE_TYPE', 'int8')
            },
            
            # WhisperX configuration
            'whisperx': {
                'model_size': os.getenv('WHISPERX_MODEL_SIZE', 'base'),
                'device': os.getenv('WHISPERX_DEVICE', 'cpu'),
                'language': os.getenv('WHISPERX_LANGUAGE', 'auto'),
                'diarize': True
            }
        }
    
    def get_transcription_config(self) -> Dict:
        """Get transcription configuration based on provider"""
        if self.transcription_provider == ModelProvider.OPENAI.value:
            return {
                'provider': 'openai',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': 'whisper-1'
            }
        elif self.transcription_provider == ModelProvider.WHISPER_CPP.value:
            return {
                'provider': 'whisper_cpp',
                **self.local_config['whisper_cpp']
            }
        elif self.transcription_provider == ModelProvider.FASTER_WHISPER.value:
            return {
                'provider': 'faster_whisper',
                **self.local_config['faster_whisper']
            }
        elif self.transcription_provider == ModelProvider.WHISPERX.value:
            return {
                'provider': 'whisperx',
                **self.local_config['whisperx']
            }
        else:
            raise ValueError(f"Unsupported transcription provider: {self.transcription_provider}")
    
    def get_llm_config(self) -> Dict:
        """Get LLM configuration based on provider"""
        if self.llm_provider == ModelProvider.OPENAI.value:
            return {
                'provider': 'openai',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': 'gpt-4o-mini'
            }
        elif self.llm_provider == ModelProvider.OLLAMA.value:
            return {
                'provider': 'ollama',
                **self.local_config['ollama']
            }
        elif self.llm_provider == ModelProvider.LOCALAI.value:
            return {
                'provider': 'localai',
                **self.local_config['localai']
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def is_local(self) -> bool:
        """Check if using local models"""
        return (self.transcription_provider != ModelProvider.OPENAI.value or 
                self.llm_provider != ModelProvider.OPENAI.value)
    
    def get_cost_estimate(self) -> Dict:
        """Get cost estimate for local vs OpenAI"""
        if self.is_local():
            return {
                'transcription_cost': 0.0,  # Free when running locally
                'llm_cost': 0.0,  # Free when running locally
                'total_cost': 0.0,
                'note': 'Local models have no API costs, only computational resources'
            }
        else:
            # OpenAI pricing estimates
            return {
                'transcription_cost': '~$0.006 per minute',
                'llm_cost': '~$0.00015 per 1K tokens (GPT-4o-mini)',
                'total_cost': 'Varies based on usage',
                'note': 'OpenAI API costs apply'
            }

# Global configuration instance
config = LocalModelsConfig() 