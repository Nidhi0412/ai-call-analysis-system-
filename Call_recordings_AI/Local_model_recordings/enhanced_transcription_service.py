#!/usr/bin/env python3
"""
Enhanced Transcription Service
==============================
Provides enhanced audio transcription with preprocessing, language detection, and translation
"""

import os
import sys
import tempfile
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Simple print-based logging
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    try:
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
    except ImportError:
        # If numpy is not available, return as is
        return obj

def _recognize_number_patterns(text: str) -> str:
    """
    Recognize and convert various number patterns to standard format
    """
    import re
    
    # Common number patterns in Indian English/Hinglish
    patterns = {
        r'\b(double)\s+(\d)\b': r'\2\2',  # "double nine" -> "99"
        r'\b(triple)\s+(\d)\b': r'\2\2\2',  # "triple five" -> "555"
        r'\b(zero\s+zero)\b': '00',
        r'\b(one\s+one)\b': '11',
        r'\b(two\s+two)\b': '22',
        r'\b(three\s+three)\b': '33',
        r'\b(four\s+four)\b': '44',
        r'\b(five\s+five)\b': '55',
        r'\b(six\s+six)\b': '66',
        r'\b(seven\s+seven)\b': '77',
        r'\b(eight\s+eight)\b': '88',
        r'\b(nine\s+nine)\b': '99',
    }
    
    processed_text = text
    for pattern, replacement in patterns.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    return processed_text

# Import configuration with error handling
try:
    from local_models_config import (
        config,
        SUPPORTED_LANGUAGES,
        LANGUAGE_DETECTION_THRESHOLD,
        HINGLISH_DETECTION_THRESHOLD,
        ENHANCED_TRANSCRIPTION_SETTINGS,
        AUDIO_PREPROCESSING
    )
except ImportError as e:
    log_warning(f"Could not import local_models_config: {e}")
    # Provide fallback defaults
    config = None
    SUPPORTED_LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'bn': 'Bengali', 'mr': 'Marathi'}
    LANGUAGE_DETECTION_THRESHOLD = 0.5
    HINGLISH_DETECTION_THRESHOLD = 0.3
    ENHANCED_TRANSCRIPTION_SETTINGS = {
        'segment_combining': True,
        'min_segment_length': 1.0,
        'max_segment_length': 10.0
    }
    AUDIO_PREPROCESSING = {
        'target_sr': 16000,
        'denoise_threshold': 0.1
    }

class EnhancedTranscriptionService:
    """
    Enhanced transcription service with preprocessing and language detection
    """
    
    def __init__(self):
        if config is None:
            # Use fallback configuration
            self.config = {
                'model_size': 'base',
                'device': 'cpu',
                'compute_type': 'int8'
            }
            self.provider = 'faster_whisper'  # Default to faster_whisper
        else:
            self.config = config.get_transcription_config()
            self.provider = config.transcription_provider
            
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transcription model"""
        try:
            if self.provider == 'faster_whisper':
                self._initialize_faster_whisper()
            elif self.provider == 'whisperx':
                self._initialize_whisperx()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            log_info(f"✅ Enhanced {self.provider} transcription model initialized")
            
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
        except Exception as e:
            raise Exception(f"WhisperX initialization failed: {e}")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess audio file for better transcription quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (preprocessed_path, quality_metrics)
        """
        try:
            # Try to import audio processing libraries
            try:
                import librosa
                import soundfile as sf
                import numpy as np
                from scipy import signal
                AUDIO_LIBS_AVAILABLE = True
            except ImportError as e:
                log_warning(f"Audio processing libraries not available: {e}")
                log_warning("Skipping audio preprocessing, using original file")
                AUDIO_LIBS_AVAILABLE = False
            
            if not AUDIO_LIBS_AVAILABLE:
                return audio_path, {'preprocessing_skipped': True, 'reason': 'Missing audio libraries'}
            
            log_info("Starting audio preprocessing")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to target sample rate
            if sr != AUDIO_PREPROCESSING['target_sr']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_PREPROCESSING['target_sr'])
                sr = AUDIO_PREPROCESSING['target_sr']
            
            # Analyze audio quality
            quality_metrics = self._analyze_audio_quality(audio, sr)
            
            # Apply denoising if needed
            if quality_metrics.get('needs_denoising', False):
                log_info("Applying denoising")
                audio = self._denoise_audio(audio, sr)
            
            # Remove long silences
            audio = self._remove_silences(audio, sr)
            
            # Save preprocessed audio
            preprocessed_path = audio_path.replace('.wav', '_preprocessed.wav')
            sf.write(preprocessed_path, audio, sr)
            
            # Verify the preprocessed audio file
            import os
            if os.path.exists(preprocessed_path):
                file_size = os.path.getsize(preprocessed_path)
                log_info(f"✅ Audio preprocessing completed: {preprocessed_path} (size: {file_size} bytes)")
            else:
                log_warning(f"⚠️ Preprocessed audio file not created: {preprocessed_path}")
            
            return preprocessed_path, quality_metrics
            
        except Exception as e:
            log_error(f"❌ Audio preprocessing failed: {e}")
            return audio_path, {'preprocessing_failed': True, 'error': str(e)}
    
    def _analyze_audio_quality(self, audio, sr: int) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        try:
            import numpy as np
            import librosa
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Calculate SNR (simplified)
            noise_floor = np.percentile(np.abs(audio), 10)
            signal_level = np.percentile(np.abs(audio), 90)
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            
            # Zero crossing rate
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Noise consistency (simplified)
            noise_consistency = np.std(audio[audio < np.percentile(audio, 20)])
            
            # Determine if denoising is needed
            needs_denoising = noise_consistency > AUDIO_PREPROCESSING['denoise_threshold']
            
            return {
                'rms': float(rms),
                'snr_db': float(snr_db),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zero_crossing_rate),
                'noise_consistency': float(noise_consistency),
                'duration': len(audio) / sr,
                'needs_denoising': needs_denoising,
                'denoising_reason': [f"High noise consistency: {noise_consistency:.3f} > {AUDIO_PREPROCESSING['denoise_threshold']}"] if needs_denoising else []
            }
            
        except ImportError as e:
            log_warning(f"Audio quality analysis failed - missing libraries: {e}")
            return {'needs_denoising': False, 'analysis_skipped': True, 'reason': 'Missing audio libraries'}
        except Exception as e:
            log_warning(f"Audio quality analysis failed: {e}")
            return {'needs_denoising': False, 'analysis_failed': True, 'error': str(e)}
    
    def _denoise_audio(self, audio, sr: int):
        """Apply denoising to audio"""
        try:
            import numpy as np
            import librosa
            
            # Simple spectral subtraction
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * sr / 512)  # Assuming 512 hop length
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            cleaned_magnitude = magnitude - 0.5 * noise_spectrum
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft)
            
            return cleaned_audio
            
        except ImportError as e:
            log_warning(f"Denoising failed - missing libraries: {e}")
            return audio
        except Exception as e:
            log_warning(f"Denoising failed: {e}")
            return audio
    
    def _remove_silences(self, audio, sr: int):
        """Remove long silences from audio"""
        try:
            import numpy as np
            import librosa
            
            # Detect non-silent regions
            non_silent_ranges = librosa.effects.split(
                audio, 
                top_db=30, 
                frame_length=2048, 
                hop_length=512
            )
            
            # Combine non-silent regions
            audio_chunks = []
            for start, end in non_silent_ranges:
                audio_chunks.append(audio[start:end])
            
            if audio_chunks:
                return np.concatenate(audio_chunks)
            else:
                return audio
                
        except ImportError as e:
            log_warning(f"Silence removal failed - missing libraries: {e}")
            return audio
        except Exception as e:
            log_warning(f"Silence removal failed: {e}")
            return audio
    
    def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect language with confidence scores
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with language detection results
        """
        try:
            log_info("Starting language detection")
            
            if self.provider == 'faster_whisper':
                # Use a small sample for language detection
                segments, info = self.model.transcribe(
                    audio_path, 
                    beam_size=5, 
                    language=None,  # Force auto-detection
                    task='transcribe'
                )
                
                detected_language = info.language
                confidence = info.language_probability
                
            elif self.provider == 'whisperx':
                # Load audio
                audio = self.model.load_audio(audio_path)
                
                # Detect language
                result = self.model.transcribe(audio)
                detected_language = result.get('language', 'unknown')
                confidence = 0.8  # WhisperX doesn't provide confidence
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Check if it's Hindi/Hinglish
            is_hindi = detected_language in ['hi', 'hin']
            is_hinglish = self._detect_hinglish(audio_path) if is_hindi else False
            
            result = {
                'detected_language': detected_language,
                'confidence': convert_numpy_types(confidence),
                'is_hindi': is_hindi,
                'is_hinglish': is_hinglish,
                'language_name': SUPPORTED_LANGUAGES.get(detected_language, detected_language),
                'is_multilingual': detected_language not in ['en', 'en-US', 'en-GB']
            }
            
            log_info(f"Language detected: {detected_language} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            log_error(f"❌ Language detection failed: {e}")
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'is_hindi': False,
                'is_hinglish': False,
                'language_name': 'Unknown',
                'is_multilingual': False
            }
    
    def _detect_hinglish(self, audio_path: str) -> bool:
        """Detect if the content is Hinglish (Hindi + English mix)"""
        try:
            # Simple Hinglish detection using transcription analysis
            if self.provider == 'faster_whisper':
                segments, info = self.model.transcribe(
                    audio_path, 
                    beam_size=5, 
                    language='hi',  # Force Hindi
                    task='transcribe'
                )
                
                # Get transcription text
                text = " ".join([segment.text for segment in segments])
                
                # Check for English words in Hindi transcription
                english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                english_count = sum(1 for word in english_words if word.lower() in text.lower())
                
                # If more than 20% of common English words are present, likely Hinglish
                return english_count > len(english_words) * 0.2
                
            return False
            
        except Exception as e:
            log_warning(f"Hinglish detection failed: {e}")
            return False
    
    def transcribe_with_enhancements(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Enhanced transcription with preprocessing and language detection
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Dict containing enhanced transcription results
        """
        start_time = time.time()
        
        try:
            log_info("Starting enhanced transcription")
            
            # Step 1: Preprocess audio
            preprocessed_path, quality_metrics = self.preprocess_audio(audio_path)
            
            # Step 2: Detect language if not specified
            if not language:
                language_info = self.detect_language(preprocessed_path)
                language = language_info['detected_language']
                is_hinglish = language_info['is_hinglish']
            else:
                language_info = {
                    'detected_language': language,
                    'confidence': 1.0,
                    'is_hindi': language in ['hi', 'hin'],
                    'is_hinglish': False,
                    'language_name': SUPPORTED_LANGUAGES.get(language, language),
                    'is_multilingual': language not in ['en', 'en-US', 'en-GB']
                }
                is_hinglish = False
            
            # Step 3: Transcribe with language-specific settings
            if self.provider == 'faster_whisper':
                result = self._transcribe_faster_whisper_enhanced(preprocessed_path, language, is_hinglish)
            elif self.provider == 'whisperx':
                result = self._transcribe_whisperx_enhanced(preprocessed_path, language, is_hinglish)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Step 4: Add metadata
            processing_time = time.time() - start_time
            result.update({
                'processing_time': processing_time,
                'provider': self.provider,
                'language_info': convert_numpy_types(language_info),
                'quality_metrics': convert_numpy_types(quality_metrics),
                'preprocessed_audio_path': preprocessed_path,
                'enhancements_applied': {
                    'audio_preprocessing': True,
                    'language_detection': not language,
                    'hinglish_detection': is_hinglish,
                    'segment_combining': ENHANCED_TRANSCRIPTION_SETTINGS['segment_combining']
                }
            })
            
            log_info(f"✅ Enhanced transcription completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            log_error(f"❌ Enhanced transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'provider': self.provider,
                'processing_time': time.time() - start_time
            }
    
    def _transcribe_faster_whisper_enhanced(self, audio_path: str, language: str, is_hinglish: bool) -> Dict[str, Any]:
        """Enhanced transcription using Faster Whisper"""
        # For Hindi/Hinglish, use English script output for better accuracy
        if language in ['hi', 'hin'] or is_hinglish:
            # Enhanced prompt for Indian code-mixing
            indian_code_mixing_prompt = """
This is a customer service call in India with:
- Hindi as primary language with regional accent (Telugu/Bengali/Marathi/Punjabi/Gujarati)
- English words and phrases mixed in naturally
- Code-switching between Hindi, regional language, and English
- Customer service context with technical terms
- Regional pronunciation variations

IMPORTANT COMPANY-SPECIFIC KEYWORDS TO RECOGNIZE ACCURATELY:
- Company Names: Yanolja, Yanolja Cloud Solution, Ezee, Ultra Viewer, AnyDesk
- Technical Terms: Cloud Solution, Viewer, Remote Desktop, AnyDesk, TeamViewer
- Email/Contact: Gmail, Outlook, Yahoo, Hotmail, Mail ID, Email ID
- Common Phrases: "Yanolja Cloud Solution", "Ezee software", "Ultra Viewer", "AnyDesk connection"
- Business Terms: Booking, Reservation, Hotel, Restaurant, Room Charge, Billing
- Technical Support: Remote access, Screen sharing, Connection issues, Login problems
- Hotel Business Terms: Revenue, CityLedger, Front Desk, Check-in, Check-out, Room Rate, Occupancy, RevPAR, ADR, Hotel Management, PMS (Property Management System)
- Financial Terms: Invoice, Payment, Credit Card, Debit Card, Cash, Receipt, Tax, GST, Service Charge, Cancellation Fee, No-show Fee
- Hotel Operations: Housekeeping, Maintenance, Concierge, Bell Boy, Room Service, Housekeeping, Laundry, Mini Bar, Room Key, Key Card
- Revenue Management: Rate Management, Dynamic Pricing, Yield Management, Revenue Optimization, Market Segmentation, Demand Forecasting
- CityLedger Terms: City Ledger, Corporate Account, Direct Billing, Credit Limit, Payment Terms, Aging Report, Outstanding Balance, Credit Application
- Common Agent Requests: Hotel Code, Property Code, AnyDesk Number, Ultra Viewer Password, Mail ID Password, Login Credentials, Access Code, System Password
- Number Recognition Patterns: 
  * Direct numbers: "1, 2, 7, 8" or "one two seven eight"
  * Spelled out: "double nine seven four zero" = "99740"
  * Mixed format: "nine double seven four" = "9774"
  * Repeated digits: "double zero" = "00", "triple five" = "555"
  * Common patterns: "zero zero" = "00", "one one" = "11", "double zero" = "00"
- Authentication Terms: Username, Password, Login, Credentials, Access Code, PIN, OTP, Verification Code, Security Code

Please transcribe accurately maintaining:
- Mixed language structure in English script
- Regional accent patterns
- English technical terms
- Natural code-switching
- Customer service terminology
- Proper transliteration of Hindi/regional words
- Accurate recognition of company names and technical terms
- Proper spelling of email addresses and technical terms
"""
            
            # Use English script output for Hinglish
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                task='transcribe',
                condition_on_previous_text=True,
                initial_prompt=indian_code_mixing_prompt
            )
        else:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                task='transcribe'
            )
        
        # Process segments
        segments_list = []
        full_text = ""
        
        # Convert generator to list to ensure we can iterate properly
        segments_list_obj = list(segments)
        log_info(f"Processing {len(segments_list_obj)} segments")
        log_info(f"Segments type: {type(segments_list_obj)}")
        log_info(f"Segments object: {segments_list_obj}")
        
        segment_count = 0
        for i, segment in enumerate(segments_list_obj):
            segment_count += 1
            log_info(f"Processing segment {i+1} (count: {segment_count})")
            log_info(f"Segment object: {segment}")
            log_info(f"Segment type: {type(segment)}")
            
            segment_text = segment.text.strip()
            log_info(f"Segment {i+1}: {segment.start:.1f}s - {segment.end:.1f}s: '{segment_text}' (length: {len(segment_text)})")
            
            # Debug: Check if segment has any content at all
            if hasattr(segment, 'text') and segment.text:
                log_info(f"  Raw segment text: '{segment.text}'")
            else:
                log_warning(f"  Segment {i+1} has no text attribute or empty text")
            
            # Debug: Check segment object attributes
            log_info(f"  Segment {i+1} attributes: {dir(segment)}")
            log_info(f"  Segment {i+1} text attribute: {getattr(segment, 'text', 'NO_TEXT_ATTR')}")
            log_info(f"  Segment {i+1} start: {getattr(segment, 'start', 'NO_START')}")
            log_info(f"  Segment {i+1} end: {getattr(segment, 'end', 'NO_END')}")
            
            # Check if segment has text
            if segment_text:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment_text,
                    'words': getattr(segment, 'words', [])
                }
                segments_list.append(segment_dict)
                full_text += segment_text + " "
            else:
                log_warning(f"Empty segment {i+1}: {segment.start:.1f}s - {segment.end:.1f}s")
        
        log_info(f"Finished processing {segment_count} segments")
        log_info(f"Total text length: {len(full_text)} characters")
        log_info(f"Non-empty segments: {len(segments_list)} out of {len(segments_list_obj)}")
        
        # If no text was extracted, try a different approach
        if not full_text.strip():
            log_warning("No text extracted from segments, trying alternative approach")
            # Try transcribing without preprocessing
            original_audio_path = audio_path.replace('_preprocessed.wav', '.wav')
            log_info(f"Trying original audio: {original_audio_path}")
            
            try:
                segments_alt, info_alt = self.model.transcribe(
                    original_audio_path,
                    beam_size=5,
                    language=language,
                    task='transcribe'
                )
                
                log_info(f"Alternative approach - Processing {len(list(segments_alt))} segments")
                
                # Convert generator to list for alternative approach
                segments_alt_list = list(segments_alt)
                log_info(f"Alternative approach - Converted to list: {len(segments_alt_list)} segments")
                
                full_text = ""
                segments_list = []
                for i, segment in enumerate(segments_alt_list):
                    segment_text = segment.text.strip()
                    log_info(f"Alt Segment {i+1}: {segment.start:.1f}s - {segment.end:.1f}s: '{segment_text}' (length: {len(segment_text)})")
                    
                    # Debug: Check segment object attributes for alternative approach
                    log_info(f"  Alt Segment {i+1} attributes: {dir(segment)}")
                    log_info(f"  Alt Segment {i+1} text attribute: {getattr(segment, 'text', 'NO_TEXT_ATTR')}")
                    
                    if segment_text:
                        segment_dict = {
                            'start': segment.start,
                            'end': segment.end,
                            'text': segment_text,
                            'words': getattr(segment, 'words', [])
                        }
                        segments_list.append(segment_dict)
                        full_text += segment_text + " "
                
                log_info(f"Alternative approach - Total text length: {len(full_text)} characters")
                log_info(f"Alternative approach - Non-empty segments: {len(segments_list)}")
                
            except Exception as e:
                log_error(f"Alternative approach failed: {e}")
                # If alternative also fails, try with English language
                try:
                    log_info("Trying with English language as fallback")
                    segments_eng, info_eng = self.model.transcribe(
                        original_audio_path,
                        beam_size=5,
                        language='en',
                        task='transcribe'
                    )
                    
                    full_text = ""
                    segments_list = []
                    for segment in segments_eng:
                        segment_text = segment.text.strip()
                        if segment_text:
                            segment_dict = {
                                'start': segment.start,
                                'end': segment.end,
                                'text': segment_text,
                                'words': getattr(segment, 'words', [])
                            }
                            segments_list.append(segment_dict)
                            full_text += segment_text + " "
                    
                    log_info(f"English fallback - Total text length: {len(full_text)} characters")
                    
                except Exception as e2:
                    log_error(f"English fallback also failed: {e2}")
        
        # Combine short segments if enabled
        if ENHANCED_TRANSCRIPTION_SETTINGS['segment_combining']:
            segments_list = self._combine_short_segments(segments_list)
        
        return {
            'success': True,
            'transcription': full_text.strip(),
            'segments': convert_numpy_types(segments_list),
            'language': info.language,
            'language_probability': convert_numpy_types(info.language_probability),
            'is_multilingual': info.language not in ['en', 'en-US', 'en-GB'],
            'method': 'enhanced_english_script' if is_hinglish else 'enhanced_standard'
        }
    
    def _transcribe_whisperx_enhanced(self, audio_path: str, language: str, is_hinglish: bool) -> Dict[str, Any]:
        """Enhanced transcription using WhisperX"""
        try:
            import whisperx
            
            # Load audio
            audio = self.model.load_audio(audio_path)
            
            # Transcribe
            result = self.model.transcribe(audio, language=language)
            
            # Process segments
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
            
            # Combine short segments if enabled
            if ENHANCED_TRANSCRIPTION_SETTINGS['segment_combining']:
                segments_list = self._combine_short_segments(segments_list)
            
            detected_language = result.get('language', 'unknown')
            return {
                'success': True,
                'transcription': full_text.strip(),
                'segments': convert_numpy_types(segments_list),
                'language': detected_language,
                'is_multilingual': detected_language not in ['en', 'en-US', 'en-GB'],
                'method': 'enhanced_auto_hinglish' if is_hinglish else 'enhanced_standard'
            }
            
        except ImportError:
            raise ImportError("whisperx not installed. Run: pip install whisperx")
        except Exception as e:
            raise Exception(f"WhisperX transcription failed: {e}")
    
    def _combine_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Combine short segments for better readability"""
        if not segments:
            return segments
        
        combined_segments = []
        current_segment = segments[0].copy()
        
        for segment in segments[1:]:
            segment_duration = segment['end'] - segment['start']
            
            # If segment is too short, combine with previous
            if segment_duration < ENHANCED_TRANSCRIPTION_SETTINGS['min_segment_length']:
                current_segment['end'] = segment['end']
                current_segment['text'] += ' ' + segment['text']
                # Handle words safely
                if 'words' in current_segment and current_segment['words'] is None:
                    current_segment['words'] = []
                if 'words' in segment and segment['words'] is not None:
                    if current_segment['words'] is None:
                        current_segment['words'] = []
                    current_segment['words'].extend(segment['words'])
            else:
                # Check if current combined segment is too long
                combined_duration = current_segment['end'] - current_segment['start']
                if combined_duration > ENHANCED_TRANSCRIPTION_SETTINGS['max_segment_length']:
                    # Split the combined segment
                    mid_point = (current_segment['start'] + current_segment['end']) / 2
                    first_half = current_segment.copy()
                    first_half['end'] = mid_point
                    combined_segments.append(first_half)
                    
                    current_segment = segment.copy()
                    current_segment['start'] = mid_point
                else:
                    combined_segments.append(current_segment)
                    current_segment = segment.copy()
        
        combined_segments.append(current_segment)
        return combined_segments
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'provider': self.provider,
            'model_loaded': self.model is not None,
            'enhanced_features': True,
            'config': self.config
        }

# Test the service
if __name__ == "__main__":
    print("🎯 Testing Enhanced Transcription Service")
    print("=" * 50)
    
    try:
        service = EnhancedTranscriptionService()
        status = service.get_status()
        print(f"✅ Service initialized: {status}")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}") 