#!/usr/bin/env python3
"""
Local Transcription Service
==========================
Provides local audio transcription using Faster Whisper, WhisperX, or Whisper.cpp
"""

import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# Common infrastructure imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple print-based logging instead of pylogger
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

from local_models_config import config

class LocalTranscriptionService:
    """
    Local transcription service using open-source models
    """
    
    def __init__(self):
        self.config = config.get_transcription_config()
        self.provider = config.transcription_provider
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transcription model based on provider"""
        try:
            if self.provider == 'faster_whisper':
                self._initialize_faster_whisper()
            elif self.provider == 'whisperx':
                self._initialize_whisperx()
            elif self.provider == 'whisper_cpp':
                self._initialize_whisper_cpp()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            log_info(f"✅ {self.provider} transcription model initialized successfully")
            
        except Exception as e:
            log_error(f"❌ Failed to initialize {self.provider} model: {e}")
            raise
    
    def _initialize_faster_whisper(self):
        """Initialize Faster Whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            model_size = self.config.get('model_size', 'base')
            device = self.config.get('device', 'cpu')
            compute_type = self.config.get('compute_type', 'int8')
            
            log_info(f"Loading Faster Whisper model: {model_size} on {device}")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
    
    def _initialize_whisperx(self):
        """Initialize WhisperX model"""
        try:
            import whisperx
            
            model_size = self.config.get('model_size', 'base')
            device = self.config.get('device', 'cpu')
            
            log_info(f"Loading WhisperX model: {model_size} on {device}")
            self.model = whisperx.load_model(model_size, device)
            
        except ImportError:
            raise ImportError("whisperx not installed. Run: pip install whisperx")
    
    def _initialize_whisper_cpp(self):
        """Initialize Whisper.cpp model"""
        try:
            import whisper_cpp
            
            model_path = self.config.get('model_path')
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Whisper model not found at: {model_path}")
            
            log_info(f"Loading Whisper.cpp model from: {model_path}")
            self.model = whisper_cpp.Whisper(model_path)
            
        except ImportError:
            raise ImportError("whisper-cpp-python not installed. Run: pip install whisper-cpp-python")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio file using local model with language support
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, will auto-detect if not provided)
            
        Returns:
            Dict containing transcription results
        """
        start_time = time.time()
        
        try:
            log_info(f"Starting transcription with {self.provider}")
            if language:
                log_info(f"Language specified: {language}")
            else:
                log_info("Auto-detecting language")
            
            if self.provider == 'faster_whisper':
                result = self._transcribe_faster_whisper(audio_path, language)
            elif self.provider == 'whisperx':
                result = self._transcribe_whisperx(audio_path, language)
            elif self.provider == 'whisper_cpp':
                result = self._transcribe_whisper_cpp(audio_path, language)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['provider'] = self.provider
            
            log_info(f"✅ Transcription completed in {processing_time:.2f}s")
            log_info(f"Detected language: {result.get('language', 'unknown')}")
            return result
            
        except Exception as e:
            log_error(f"❌ Transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'provider': self.provider,
                'processing_time': time.time() - start_time
            }
    
    def _transcribe_faster_whisper(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe using Faster Whisper with language support"""
        # For Hindi/Hinglish, use English script output
        if language in ['hi', 'hin']:
            segments, info = self.model.transcribe(
                audio_path, 
                beam_size=5, 
                language=language,
                initial_prompt="Transcribe in English script for better accuracy."
            )
        else:
            # Use language if specified, otherwise auto-detect
            if language:
                segments, info = self.model.transcribe(audio_path, beam_size=5, language=language)
            else:
                segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        # Convert segments to list
        segments_list = []
        full_text = ""
        
        for segment in segments:
            segment_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': getattr(segment, 'words', [])
            }
            segments_list.append(segment_dict)
            full_text += segment.text + " "
        
        return {
            'success': True,
            'transcription': full_text.strip(),
            'segments': convert_numpy_types(segments_list),
            'language': info.language,
            'language_probability': convert_numpy_types(info.language_probability),
            'is_multilingual': info.language not in ['en', 'en-US', 'en-GB']
        }
    
    def _transcribe_whisperx(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe using WhisperX with language support"""
        # Load audio
        audio = self.model.load_audio(audio_path)
        
        # Transcribe with language support
        if language:
            result = self.model.transcribe(audio, language=language)
        else:
            result = self.model.transcribe(audio)
        
        # Get segments
        segments_list = []
        full_text = ""
        
        for segment in result['segments']:
            segment_dict = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'words': segment.get('words', [])
            }
            segments_list.append(segment_dict)
            full_text += segment['text'] + " "
        
        detected_language = result.get('language', 'unknown')
        return {
            'success': True,
            'transcription': full_text.strip(),
            'segments': convert_numpy_types(segments_list),
            'language': detected_language,
            'is_multilingual': detected_language not in ['en', 'en-US', 'en-GB']
        }
    
    def _transcribe_whisper_cpp(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe using Whisper.cpp with language support"""
        # Load and transcribe
        if language:
            result = self.model.transcribe(audio_path, language=language)
        else:
            result = self.model.transcribe(audio_path)
        
        # Parse segments
        segments_list = []
        full_text = ""
        
        for segment in result['segments']:
            segment_dict = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            }
            segments_list.append(segment_dict)
            full_text += segment['text'] + " "
        
        detected_language = result.get('language', 'unknown')
        return {
            'success': True,
            'transcription': full_text.strip(),
            'segments': segments_list,
            'language': detected_language,
            'is_multilingual': detected_language not in ['en', 'en-US', 'en-GB']
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'provider': self.provider,
            'model_loaded': self.model is not None,
            'config': self.config
        }

# Test the service
if __name__ == "__main__":
    print("🎯 Testing Local Transcription Service")
    print("=" * 40)
    
    try:
        service = LocalTranscriptionService()
        status = service.get_status()
        print(f"✅ Service initialized: {status}")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}") 