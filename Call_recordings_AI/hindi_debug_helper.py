import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf
from pydub import AudioSegment

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pylogger with error handling
try:
    from pylogger import pylogger
    logger = pylogger("/home/saas/logs/innex", "call_recordings_ai")
except ImportError:
    # Mock logger for testing/development
    class MockLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log_it(self, data): 
            print(f"LOG: {data}")
        def info(self, msg): 
            print(f"INFO: {msg}")
        def error(self, msg): 
            print(f"ERROR: {msg}")
        def warning(self, msg): 
            print(f"WARNING: {msg}")
        def debug(self, msg): 
            print(f"DEBUG: {msg}")
    logger = MockLogger()

from Call_recordings_AI.transcription_with_speakers import TranscriptionWithSpeakersService

class HindiDebugHelper:
    """
    Comprehensive debugging and enhancement tools for Hindi audio processing
    """
    
    def __init__(self, api_key=None):
        """Initialize the Hindi debug helper"""
        self.transcription_service = TranscriptionWithSpeakersService(api_key=api_key) if api_key else None
        
        # Hindi-specific language codes and variations
        self.hindi_variations = ['hi', 'hin', 'hindi', 'hindi-in', 'hi-IN']
        
        # Common Hindi words for validation
        self.hindi_validation_words = [
            'नमस्ते', 'धन्यवाद', 'कैसे', 'हैं', 'में', 'का', 'की', 'के', 'है', 'हैं',
            'hello', 'thank', 'how', 'are', 'in', 'of', 'the', 'is', 'am'
        ]
        
        logger.log_it({
            "logType": "info",
            "prefix": "hindi_debug_helper",
            "logData": "HindiDebugHelper initialized with validation tools"
        })
    
    def analyze_hindi_audio_quality(self, audio_path: str) -> Dict:
        """
        Comprehensive analysis of Hindi audio quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Detailed Hindi-specific audio analysis
        """
        try:
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": f"Analyzing Hindi audio quality: {audio_path}"
            })
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Basic audio metrics
            duration = len(y) / sr
            rms = np.sqrt(np.mean(y**2))
            
            # Spectral analysis for Hindi speech characteristics
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Hindi speech typically has specific frequency characteristics
            # Analyze frequency bands relevant to Hindi phonetics
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Hindi vowels and consonants have specific frequency ranges
            low_freq_energy = np.mean(magnitude[freqs < 500])
            mid_freq_energy = np.mean(magnitude[(freqs >= 500) & (freqs < 2000)])
            high_freq_energy = np.mean(magnitude[freqs >= 2000])
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            
            # MFCC features for speech analysis
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Voice activity detection
            # Hindi speech has specific patterns
            energy = librosa.feature.rms(y=y).flatten()
            energy_threshold = np.percentile(energy, 30)
            speech_segments = energy > energy_threshold
            speech_ratio = np.sum(speech_segments) / len(speech_segments)
            
            # Analyze silence patterns (important for Hindi)
            silence_threshold = np.percentile(energy, 20)
            silence_segments = energy < silence_threshold
            silence_ratio = np.sum(silence_segments) / len(silence_segments)
            
            # Calculate SNR
            signal_level = np.percentile(magnitude, 90)
            noise_floor = np.percentile(magnitude, 10)
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            # Hindi-specific analysis
            analysis = {
                "basic_metrics": {
                    "duration_seconds": float(duration),
                    "sample_rate": sr,
                    "rms": float(rms),
                    "snr_db": float(snr_db)
                },
                "spectral_analysis": {
                    "spectral_centroid": float(spectral_centroid),
                    "spectral_rolloff": float(spectral_rolloff),
                    "spectral_bandwidth": float(spectral_bandwidth),
                    "low_freq_energy": float(low_freq_energy),
                    "mid_freq_energy": float(mid_freq_energy),
                    "high_freq_energy": float(high_freq_energy)
                },
                "speech_analysis": {
                    "speech_ratio": float(speech_ratio),
                    "silence_ratio": float(silence_ratio),
                    "energy_threshold": float(energy_threshold),
                    "silence_threshold": float(silence_threshold)
                },
                "mfcc_features": {
                    "mfcc_mean": mfcc_mean.tolist(),
                    "mfcc_std": mfcc_std.tolist()
                },
                "hindi_specific": {
                    "has_adequate_speech": speech_ratio > 0.3,
                    "has_proper_silence": 0.1 < silence_ratio < 0.7,
                    "frequency_balance": "good" if mid_freq_energy > low_freq_energy * 0.5 else "poor",
                    "recommended_preprocessing": self._get_hindi_preprocessing_recommendations(
                        snr_db, speech_ratio, silence_ratio, rms
                    )
                }
            }
            
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Hindi audio analysis completed",
                    "file": os.path.basename(audio_path),
                    "duration": f"{duration:.1f}s",
                    "speech_ratio": f"{speech_ratio:.2f}",
                    "snr_db": f"{snr_db:.1f}",
                    "recommendations": analysis["hindi_specific"]["recommended_preprocessing"]
                }
            })
            
            return analysis
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Hindi audio analysis failed",
                    "file": os.path.basename(audio_path),
                    "error": str(e)
                }
            })
            return {"error": str(e)}
    
    def _get_hindi_preprocessing_recommendations(self, snr_db: float, speech_ratio: float, 
                                               silence_ratio: float, rms: float) -> List[str]:
        """
        Get Hindi-specific preprocessing recommendations
        
        Args:
            snr_db: Signal-to-noise ratio
            speech_ratio: Ratio of speech segments
            silence_ratio: Ratio of silence segments
            rms: Root mean square energy
            
        Returns:
            List of preprocessing recommendations
        """
        recommendations = []
        
        if snr_db < 10:
            recommendations.append("Apply aggressive noise reduction")
        elif snr_db < 15:
            recommendations.append("Apply moderate noise reduction")
        else:
            recommendations.append("Skip noise reduction (good SNR)")
        
        if speech_ratio < 0.2:
            recommendations.append("Audio may be too quiet - increase gain")
        elif speech_ratio > 0.8:
            recommendations.append("Audio may be too loud - normalize levels")
        
        if silence_ratio < 0.05:
            recommendations.append("Remove excessive silence")
        elif silence_ratio > 0.8:
            recommendations.append("Audio may be mostly silence")
        
        if rms < 0.01:
            recommendations.append("Very low signal - increase gain")
        elif rms > 0.5:
            recommendations.append("Very high signal - normalize")
        
        return recommendations
    
    async def debug_hindi_transcription(self, audio_path: str) -> Dict:
        """
        Comprehensive debugging of Hindi transcription process
        
        Args:
            audio_path: Path to Hindi audio file
            
        Returns:
            dict: Detailed debugging information
        """
        try:
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": f"Starting Hindi transcription debug: {audio_path}"
            })
            
            # Step 1: Audio quality analysis
            quality_analysis = self.analyze_hindi_audio_quality(audio_path)
            
            # Step 2: Test transcription with different settings
            debug_results = {
                "audio_quality": quality_analysis,
                "transcription_tests": {},
                "recommendations": []
            }
            
            # Test 1: Standard transcription
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": "Test 1: Standard transcription"
            })
            
            if self.transcription_service is None:
                logger.log_it({
                    "logType": "error",
                    "prefix": "hindi_debug_helper",
                    "logData": {
                        "message": "Hindi transcription debug failed",
                        "file": audio_path,
                        "error": "'NoneType' object has no attribute 'transcribe_with_speakers'"
                    }
                })
                return {
                    "status": "error",
                    "error": "Transcription service not initialized. API key required.",
                    "audio_quality": quality_analysis
                }
            
            result1 = self.transcription_service.transcribe_with_speakers(audio_path)
            debug_results["transcription_tests"]["standard"] = {
                "status": result1.get("status"),
                "detected_language": result1.get("detected_language"),
                "text_length": len(result1.get("transcribed_text", "")),
                "duration": result1.get("duration"),
                "segments_count": len(result1.get("segments", [])),
                "text_sample": result1.get("transcribed_text", "")[:200] + "..." if len(result1.get("transcribed_text", "")) > 200 else result1.get("transcribed_text", "")
            }
            
            # Test 2: Force Hindi language detection
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": "Test 2: Forced Hindi transcription"
            })
            
            # Create a modified transcription call with Hindi language hint
            try:
                if self.transcription_service is None:
                    result2 = {
                        "status": "error",
                        "error": "Transcription service not available"
                    }
                else:
                    with open(audio_path, "rb") as audio_file:
                        transcript = self.transcription_service.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="verbose_json",
                            language="hi",  # Force Hindi
                            timestamp_granularities=["segment"]
                        )
                    
                    result2 = {
                        "status": "success",
                        "detected_language": transcript.language,
                        "transcribed_text": transcript.text,
                        "duration": transcript.duration,
                        "segments": transcript.segments if hasattr(transcript, 'segments') else []
                    }
            except Exception as e:
                result2 = {
                    "status": "error",
                    "error": str(e)
                }
            
            debug_results["transcription_tests"]["forced_hindi"] = {
                "status": result2.get("status"),
                "detected_language": result2.get("detected_language"),
                "text_length": len(result2.get("transcribed_text", "")),
                "duration": result2.get("duration"),
                "segments_count": len(result2.get("segments", [])),
                "text_sample": result2.get("transcribed_text", "")[:200] + "..." if len(result2.get("transcribed_text", "")) > 200 else result2.get("transcribed_text", "")
            }
            
            # Test 3: Translation quality test
            if result1.get("status") == "success" and result1.get("transcribed_text"):
                logger.log_it({
                    "logType": "info",
                    "prefix": "hindi_debug_helper",
                    "logData": "Test 3: Translation quality test"
                })
                
                # Test translation of first segment
                segments = result1.get("segments", [])
                if segments and self.transcription_service is not None:
                    first_segment_text = segments[0].get("text", "")
                    if first_segment_text:
                        translation_result = await self.transcription_service.translate_with_openai(
                            first_segment_text, 
                            result1.get("detected_language", "hi")
                        )
                        
                        debug_results["transcription_tests"]["translation_test"] = {
                            "original_text": first_segment_text,
                            "translated_text": translation_result.get("translated_text", ""),
                            "translation_status": translation_result.get("status"),
                            "source_language": translation_result.get("source_language")
                        }
            
            # Generate recommendations
            debug_results["recommendations"] = self._generate_hindi_recommendations(debug_results)
            
            # Log comprehensive results
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Hindi transcription debug completed",
                    "file": os.path.basename(audio_path),
                    "standard_status": debug_results["transcription_tests"]["standard"]["status"],
                    "forced_hindi_status": debug_results["transcription_tests"]["forced_hindi"]["status"],
                    "recommendations_count": len(debug_results["recommendations"])
                }
            })
            
            return debug_results
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Hindi transcription debug failed",
                    "file": os.path.basename(audio_path),
                    "error": str(e)
                }
            })
            return {"error": str(e)}
    
    def _generate_hindi_recommendations(self, debug_results: Dict) -> List[str]:
        """
        Generate Hindi-specific recommendations based on debug results
        
        Args:
            debug_results: Debug results from transcription tests
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check audio quality
        quality = debug_results.get("audio_quality", {})
        if "hindi_specific" in quality:
            hindi_specific = quality["hindi_specific"]
            if not hindi_specific.get("has_adequate_speech"):
                recommendations.append("Audio may be too quiet - consider increasing gain")
            if hindi_specific.get("frequency_balance") == "poor":
                recommendations.append("Poor frequency balance - consider audio enhancement")
        
        # Check transcription results
        tests = debug_results.get("transcription_tests", {})
        
        # Compare standard vs forced Hindi
        standard = tests.get("standard", {})
        forced_hindi = tests.get("forced_hindi", {})
        
        if standard.get("status") == "success" and forced_hindi.get("status") == "success":
            standard_text_len = standard.get("text_length", 0)
            forced_text_len = forced_hindi.get("text_length", 0)
            
            if forced_text_len > standard_text_len * 1.2:
                recommendations.append("Forcing Hindi language detection improves results")
            elif standard_text_len > forced_text_len * 1.2:
                recommendations.append("Auto-detection works better than forced Hindi")
        
        # Check translation quality
        translation_test = tests.get("translation_test", {})
        if translation_test.get("translation_status") == "success":
            original = translation_test.get("original_text", "")
            translated = translation_test.get("translated_text", "")
            
            if len(translated) < len(original) * 0.5:
                recommendations.append("Translation quality may be poor - consider manual review")
            elif "coroutine" in translated.lower():
                recommendations.append("Translation returned coroutine object - check async handling")
        
        # General recommendations
        if standard.get("text_length", 0) < 50:
            recommendations.append("Very short transcription - check audio quality")
        
        if standard.get("detected_language") != "hi":
            recommendations.append(f"Language detected as {standard.get('detected_language')} instead of Hindi")
        
        return recommendations
    
    def create_hindi_debug_report(self, debug_results: Dict, output_file: str):
        """
        Create a comprehensive Hindi debug report
        
        Args:
            debug_results: Debug results
            output_file: Path to save the report
        """
        try:
            report = {
                "timestamp": str(np.datetime64('now')),
                "debug_results": debug_results,
                "summary": {
                    "audio_quality_score": self._calculate_audio_quality_score(debug_results),
                    "transcription_quality_score": self._calculate_transcription_quality_score(debug_results),
                    "overall_recommendation": self._get_overall_recommendation(debug_results)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.log_it({
                "logType": "info",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Hindi debug report created",
                    "file": output_file
                }
            })
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "hindi_debug_helper",
                "logData": {
                    "message": "Failed to create Hindi debug report",
                    "error": str(e)
                }
            })
    
    def _calculate_audio_quality_score(self, debug_results: Dict) -> float:
        """Calculate audio quality score (0-100)"""
        quality = debug_results.get("audio_quality", {})
        if "error" in quality:
            return 0.0
        
        score = 50.0  # Base score
        
        # Adjust based on Hindi-specific metrics
        hindi_specific = quality.get("hindi_specific", {})
        if hindi_specific.get("has_adequate_speech"):
            score += 20
        if hindi_specific.get("has_proper_silence"):
            score += 15
        if hindi_specific.get("frequency_balance") == "good":
            score += 15
        
        return min(100.0, max(0.0, score))
    
    def _calculate_transcription_quality_score(self, debug_results: Dict) -> float:
        """Calculate transcription quality score (0-100)"""
        tests = debug_results.get("transcription_tests", {})
        score = 0.0
        
        # Check if transcription was successful
        standard = tests.get("standard", {})
        if standard.get("status") == "success":
            score += 40
            
            # Check text length
            text_length = standard.get("text_length", 0)
            if text_length > 100:
                score += 30
            elif text_length > 50:
                score += 20
            elif text_length > 10:
                score += 10
        
        # Check language detection
        if standard.get("detected_language") == "hi":
            score += 20
        elif standard.get("detected_language") in ["en", "unknown"]:
            score += 10
        
        # Check translation
        translation_test = tests.get("translation_test", {})
        if translation_test.get("translation_status") == "success":
            score += 10
        
        return min(100.0, max(0.0, score))
    
    def _get_overall_recommendation(self, debug_results: Dict) -> str:
        """Get overall recommendation based on debug results"""
        audio_score = self._calculate_audio_quality_score(debug_results)
        transcription_score = self._calculate_transcription_quality_score(debug_results)
        
        if audio_score < 30:
            return "Poor audio quality - preprocessing needed"
        elif transcription_score < 30:
            return "Poor transcription quality - check audio and language settings"
        elif transcription_score < 60:
            return "Moderate quality - some improvements possible"
        else:
            return "Good quality - minor optimizations only"

async def main():
    """Example usage of Hindi debug helper"""
    debug_helper = HindiDebugHelper()
    
    # Test with a Hindi audio file
    test_file = "Call_recordings_AI/results/preprocessed_audio/hindi_test.wav"
    
    if os.path.exists(test_file):
        print("Running Hindi debug analysis...")
        debug_results = await debug_helper.debug_hindi_transcription(test_file)
        
        # Create debug report
        debug_helper.create_hindi_debug_report(debug_results, "hindi_debug_report.json")
        
        print("Debug analysis completed. Check hindi_debug_report.json for details.")
    else:
        print(f"Test file not found: {test_file}")
        print("Please provide a Hindi audio file for testing.")

if __name__ == "__main__":
    asyncio.run(main()) 