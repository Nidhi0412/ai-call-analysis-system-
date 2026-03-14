#!/usr/bin/env python3
"""
Local Model Adapter
==================
This adapter provides local model alternatives to OpenAI services while maintaining
the same interface as your existing TranscriptionWithSpeakersService and CallAnalysisService.
This allows you to use local models without changing your existing UI or application code.
"""

import os
import sys
import json
import time
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Simple print-based logging instead of pylogger
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

try:
    from local_models_config import config
    from local_transcription_service import LocalTranscriptionService
    from local_llm_service import LocalLLMService
    LOCAL_SERVICES_AVAILABLE = True
except ImportError as e:
    log_error(f"⚠️ Local services not available: {e}")
    LOCAL_SERVICES_AVAILABLE = False
    config = None

# Local models system doesn't need OpenAI services
EXISTING_SERVICES_AVAILABLE = False

class LocalModelAdapter:
    """
    Adapter that provides local model alternatives to OpenAI services
    """
    
    def __init__(self, use_local_models: bool = True):
        self.use_local_models = use_local_models
        self.local_transcription_service = None
        self.local_llm_service = None
        self.existing_transcription_service = None
        self.existing_analysis_service = None
        
        if use_local_models:
            self._initialize_local_services()
        else:
            self._initialize_existing_services()
    
    def _initialize_local_services(self):
        """Initialize local model services"""
        try:
            # Try to import enhanced transcription service first
            try:
                from enhanced_transcription_service import EnhancedTranscriptionService
                self.local_transcription_service = EnhancedTranscriptionService()
                log_info("✅ Enhanced transcription service initialized")
            except ImportError as e:
                log_warning(f"Enhanced transcription service not available: {e}")
                # Fallback to basic transcription service
                from local_transcription_service import LocalTranscriptionService
                self.local_transcription_service = LocalTranscriptionService()
                log_info("✅ Basic transcription service initialized")
            
            # Initialize LLM service (optional - can work without it)
            try:
                from local_llm_service import LocalLLMService
                self.local_llm_service = LocalLLMService()
                log_info("✅ Local LLM service initialized")
            except Exception as e:
                log_warning(f"Local LLM service not available: {e}")
                self.local_llm_service = None
            
        except Exception as e:
            log_error(f"❌ Failed to initialize local services: {e}")
            self.local_transcription_service = None
            self.local_llm_service = None
    
    def _initialize_existing_services(self):
        """Initialize existing OpenAI-based services (not available in local system)"""
        log_info("🔄 Local models system - OpenAI services not available")
        self.existing_transcription_service = None
        self.existing_analysis_service = None
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio using local or existing service with language support
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, will auto-detect if not provided)
            
        Returns:
            Dict containing transcription results
        """
        if self.use_local_models and self.local_transcription_service:
            log_info("🎯 Using local transcription service")
            
            # Check if enhanced transcription is available
            if hasattr(self.local_transcription_service, 'transcribe_with_enhancements'):
                log_info("Using enhanced transcription with preprocessing and language detection")
                return self.local_transcription_service.transcribe_with_enhancements(audio_path, language)
            else:
                log_info("Using basic transcription service")
                return self.local_transcription_service.transcribe_audio(audio_path, language)
        elif self.existing_transcription_service:
            log_info("🎯 Using existing transcription service")
            return self.existing_transcription_service.transcribe(audio_path, language)
        else:
            return {
                'success': False,
                'error': 'No transcription service available'
            }
    
    def analyze_call(self, transcription: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze call transcription using local or existing service with language support
        
        Args:
            transcription: Call transcription text
            language: Language of the transcription (for better analysis)
            
        Returns:
            Dict containing analysis results
        """
        if self.use_local_models and self.local_llm_service:
            log_info("🎯 Using local LLM service for analysis")
            return self.local_llm_service.analyze_call(transcription, language)
        elif self.existing_analysis_service:
            log_info("🎯 Using existing analysis service")
            return self.existing_analysis_service.analyze_call(transcription, language)
        else:
            return {
                'success': False,
                'error': 'No analysis service available'
            }
    
    def translate_text(self, text: str, source_lang: str = None, target_lang: str = "en") -> Dict[str, Any]:
        """
        Translate text using local or existing service
        
        Args:
            text: Text to translate
            source_lang: Source language (optional)
            target_lang: Target language (default: en)
            
        Returns:
            Dict containing translation results
        """
        if self.use_local_models and self.local_llm_service:
            log_info("🎯 Using local LLM service for translation")
            return self.local_llm_service.translate_text(text, source_lang, target_lang)
        elif self.existing_analysis_service:
            log_info("🎯 Using existing service for translation")
            # Assuming existing service has translation method
            return self.existing_analysis_service.translate_text(text, source_lang, target_lang)
        else:
            return {
                'success': False,
                'error': 'No translation service available'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        status = {
            'use_local_models': self.use_local_models,
            'local_services_available': LOCAL_SERVICES_AVAILABLE,
            'existing_services_available': EXISTING_SERVICES_AVAILABLE,
            'local_transcription_initialized': self.local_transcription_service is not None,
            'local_llm_initialized': self.local_llm_service is not None,
            'existing_transcription_initialized': self.existing_transcription_service is not None,
            'existing_analysis_initialized': self.existing_analysis_service is not None
        }
        
        if config:
            status['config'] = {
                'transcription_provider': config.transcription_provider,
                'llm_provider': config.llm_provider
            }
        
        return status
    
    def switch_to_local_models(self):
        """Switch to using local models"""
        log_info("🔄 Switching to local models")
        self.use_local_models = True
        if not self.local_transcription_service or not self.local_llm_service:
            self._initialize_local_services()
    
    def switch_to_existing_models(self):
        """Switch to using existing OpenAI-based models"""
        log_info("🔄 Switching to existing models")
        self.use_local_models = False
        if not self.existing_transcription_service or not self.existing_analysis_service:
            self._initialize_existing_services()

# Global adapter instance
adapter = LocalModelAdapter(use_local_models=True)

def get_adapter() -> LocalModelAdapter:
    """Get the global adapter instance"""
    return adapter

def switch_to_local_models():
    """Switch to local models"""
    adapter.switch_to_local_models()

def switch_to_existing_models():
    """Switch to existing models"""
    adapter.switch_to_existing_models()

if __name__ == "__main__":
    print("🎯 Local Model Adapter Test")
    print("=" * 40)
    
    adapter = LocalModelAdapter(use_local_models=True)
    status = adapter.get_status()
    
    print("📊 Adapter Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Adapter ready for integration with existing UI!") 