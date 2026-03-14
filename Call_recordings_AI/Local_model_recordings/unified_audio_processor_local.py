#!/usr/bin/env python3
"""
Unified Audio Processing System - Local Models Version
=====================================================

This module provides a unified interface for audio preprocessing and speaker diarization,
specifically adapted for local models without OpenAI dependencies.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Audio processing libraries
from pydub import AudioSegment, silence
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal
from scipy.stats import pearsonr

# Machine learning
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Simple print-based logging for local models
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

class UnifiedAudioProcessorLocal:
    """
    Unified audio processing system that combines preprocessing and speaker diarization
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 silence_threshold_db: int = -14,
                 min_silence_len: int = 1500,
                 keep_silence: int = 300,
                 snr_threshold: float = 15.0,
                 noise_threshold: float = 0.1,
                 n_speakers: int = 2):
        """
        Initialize the unified audio processor
        
        Args:
            target_sr: Target sample rate (default: 16000 for Whisper)
            silence_threshold_db: Silence threshold in dB relative to audio level
            min_silence_len: Minimum silence length to remove (ms)
            keep_silence: Amount of silence to keep around speech (ms)
            snr_threshold: SNR threshold above which denoising is skipped
            noise_threshold: Noise level threshold for denoising decision
            n_speakers: Number of speakers to detect
        """
        self.target_sr = target_sr
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.snr_threshold = snr_threshold
        self.noise_threshold = noise_threshold
        self.n_speakers = n_speakers
        
        # Initialize speaker models
        self.speaker_models = {}
        
        log_info(f"UnifiedAudioProcessorLocal initialized with target_sr={target_sr}, n_speakers={n_speakers}")
    
    def analyze_audio_quality(self, audio_path: str) -> dict:
        """
        Analyze audio quality to determine if denoising is needed
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Audio quality metrics
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Calculate RMS (Root Mean Square) for signal strength
            rms = np.sqrt(np.mean(y**2))
            
            # Calculate SNR approximation
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Estimate noise floor from lower frequency bands
            noise_floor = np.percentile(magnitude, 10)
            signal_level = np.percentile(magnitude, 90)
            
            # Calculate SNR
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            # Analyze frequency spectrum
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            
            # Detect background noise characteristics
            noise_bands = magnitude[:, magnitude.mean(axis=0) < np.percentile(magnitude.mean(axis=0), 20)]
            noise_consistency = np.std(noise_bands) / (np.mean(noise_bands) + 1e-10)
            
            # Calculate zero crossing rate (indicates noise vs speech)
            zcr = librosa.feature.zero_crossing_rate(y).mean()
            
            # Convert all NumPy types to Python native types
            quality_metrics = {
                "rms": float(rms),
                "snr_db": float(snr_db),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "noise_consistency": float(noise_consistency),
                "zero_crossing_rate": float(zcr),
                "duration": float(len(y) / sr)
            }
            
            # Determine if denoising is needed
            needs_denoising = bool(
                snr_db < self.snr_threshold or
                noise_consistency > self.noise_threshold or
                rms < 0.01  # Very low signal
            )
            
            quality_metrics["needs_denoising"] = needs_denoising
            quality_metrics["denoising_reason"] = []
            
            if snr_db < self.snr_threshold:
                quality_metrics["denoising_reason"].append(f"Low SNR: {snr_db:.1f}dB < {self.snr_threshold}dB")
            if noise_consistency > self.noise_threshold:
                quality_metrics["denoising_reason"].append(f"High noise consistency: {noise_consistency:.3f} > {self.noise_threshold}")
            if rms < 0.01:
                quality_metrics["denoising_reason"].append(f"Very low signal strength: {rms:.4f}")
            
            log_info(f"Audio quality analysis completed for {os.path.basename(audio_path)}. Needs denoising: {needs_denoising}")
            
            return quality_metrics
            
        except Exception as e:
            log_error(f"Audio quality analysis failed for {os.path.basename(audio_path)}. Error: {e}")
            # Default to denoising if analysis fails
            return {
                "needs_denoising": True,
                "denoising_reason": ["Analysis failed, defaulting to denoising"],
                "error": str(e)
            }
    
    def preprocess_audio(self, input_path: str, output_path: str) -> bool:
        """
        Preprocess audio file with smart denoising and silence removal
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save preprocessed audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        temp_mono_path = None
        temp_denoised_path = None
        
        try:
            log_info(f"Processing audio file: {input_path}")
            
            # Step 1: Load and convert to mono, resample
            log_info("Step 1: Converting to mono and resampling...")
            sound = AudioSegment.from_wav(input_path)
            sound = sound.set_channels(1).set_frame_rate(self.target_sr)
            
            # Step 2: Export to temporary file for analysis and processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_mono:
                sound.export(temp_mono.name, format="wav")
                temp_mono_path = temp_mono.name
            
            # Step 3: Analyze audio quality
            log_info("Step 2: Analyzing audio quality...")
            quality_metrics = self.analyze_audio_quality(temp_mono_path)
            
            # Step 4: Smart denoising decision
            if quality_metrics.get("needs_denoising", True):
                log_info(f"Step 3: Denoising audio (quality analysis indicated need). Reasons: {quality_metrics.get('denoising_reason', [])}")
                
                try:
                    y, sr = librosa.load(temp_mono_path, sr=self.target_sr)
                    y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False)
                    
                    # Export denoised audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_denoised:
                        sf.write(temp_denoised.name, y_denoised, sr)
                        temp_denoised_path = temp_denoised.name
                        
                    log_info("Denoising completed successfully")
                    
                except Exception as denoise_error:
                    log_warning(f"Denoising failed for {os.path.basename(input_path)}. Proceeding without denoising. Error: {denoise_error}")
                    # If denoising fails, use the original audio
                    temp_denoised_path = temp_mono_path
            else:
                log_info(f"Step 3: Skipping denoising (audio quality is good) for {os.path.basename(input_path)}. SNR: {quality_metrics.get('snr_db', 0):.1f}dB")
                # Skip denoising, use mono audio directly
                temp_denoised_path = temp_mono_path
            
            # Step 5: Remove long silences
            log_info("Step 4: Removing long silences...")
            processed = AudioSegment.from_wav(temp_denoised_path)
            silence_thresh = processed.dBFS + self.silence_threshold_db
            
            chunks = silence.split_on_silence(
                processed,
                min_silence_len=self.min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=self.keep_silence
            )
            
            # Step 6: Combine chunks and export final audio
            log_info("Step 5: Combining audio chunks...")
            if chunks:
                final_audio = sum(chunks)
            else:
                final_audio = processed
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_audio.export(output_path, format="wav")
            
            # Cleanup temporary files
            if temp_mono_path and os.path.exists(temp_mono_path) and temp_mono_path != temp_denoised_path:
                os.remove(temp_mono_path)
            if temp_denoised_path and os.path.exists(temp_denoised_path) and temp_denoised_path != temp_mono_path:
                os.remove(temp_denoised_path)
            
            log_info(f"Audio preprocessing completed successfully for {os.path.basename(input_path)}. Output: {output_path}")
            return True
            
        except Exception as e:
            log_error(f"Error processing audio file {input_path}. Error: {e}")
            # Cleanup temp files if they exist
            if temp_mono_path and os.path.exists(temp_mono_path):
                try:
                    os.remove(temp_mono_path)
                except:
                    pass
            if temp_denoised_path and os.path.exists(temp_denoised_path):
                try:
                    os.remove(temp_denoised_path)
                except:
                    pass
            return False
    
    def extract_voice_features(self, audio: np.ndarray) -> Dict:
        """
        Extract comprehensive voice features for speaker identification
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Voice fingerprint features
        """
        try:
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # 2. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.target_sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            # 3. Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)
            
            # 4. Formant Features (vowel resonances)
            formants = self._extract_formants(audio)
            
            # 5. Prosody Features (rhythm, stress, intonation)
            prosody = self._extract_prosody(audio)
            
            # 6. Voice Quality Features
            voice_quality = self._extract_voice_quality(audio)
            
            features = {
                'mfcc': mfccs,
                'mfcc_delta': mfcc_delta,
                'mfcc_delta2': mfcc_delta2,
                'pitch': pitch_values,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'formants': formants,
                'prosody': prosody,
                'voice_quality': voice_quality
            }
            
            return features
            
        except Exception as e:
            log_error(f"Voice feature extraction failed. Error: {e}")
            return {}
    
    def extract_speaker_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive speaker features for ML-based diarization
        
        Args:
            audio: Audio signal
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            # Extract voice fingerprint
            voice_features = self.extract_voice_features(audio)
            
            # Extract acoustic analysis
            acoustic_features = self._analyze_acoustics(audio)
            
            # Combine features into a single vector
            feature_vector = []
            
            # Add MFCC features
            if 'mfcc' in voice_features:
                mfcc_mean = np.mean(voice_features['mfcc'], axis=1)
                mfcc_std = np.std(voice_features['mfcc'], axis=1)
                feature_vector.extend(mfcc_mean)
                feature_vector.extend(mfcc_std)
            
            # Add pitch features
            if 'pitch' in voice_features and len(voice_features['pitch']) > 0:
                feature_vector.extend([
                    np.mean(voice_features['pitch']),
                    np.std(voice_features['pitch']),
                    np.min(voice_features['pitch']),
                    np.max(voice_features['pitch'])
                ])
            else:
                feature_vector.extend([0, 0, 0, 0])
            
            # Add formant features
            if 'formants' in voice_features:
                formants = voice_features['formants']
                feature_vector.extend([
                    formants.get('f1_mean', 0),
                    formants.get('f2_mean', 0),
                    formants.get('f3_mean', 0),
                    formants.get('f1_std', 0),
                    formants.get('f2_std', 0),
                    formants.get('f3_std', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0, 0, 0, 0])
            
            # Add prosody features
            if 'prosody' in voice_features:
                prosody = voice_features['prosody']
                feature_vector.extend([
                    prosody.get('energy_mean', 0),
                    prosody.get('energy_std', 0),
                    prosody.get('zcr_mean', 0),
                    prosody.get('speaking_rate', 0),
                    prosody.get('intonation_mean', 0),
                    prosody.get('intonation_std', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0, 0, 0, 0])
            
            # Add voice quality features
            if 'voice_quality' in voice_features:
                vq = voice_features['voice_quality']
                feature_vector.extend([
                    vq.get('jitter', 0),
                    vq.get('shimmer', 0),
                    vq.get('hnr', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # Add acoustic features
            if 'spectral' in acoustic_features:
                spectral = acoustic_features['spectral']
                feature_vector.extend([
                    spectral.get('spectral_centroid_mean', 0),
                    spectral.get('spectral_centroid_std', 0),
                    spectral.get('spectral_rolloff_mean', 0),
                    spectral.get('spectral_rolloff_std', 0),
                    spectral.get('spectral_bandwidth_mean', 0),
                    spectral.get('spectral_bandwidth_std', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0, 0, 0, 0])
            
            return np.array(feature_vector)
            
        except Exception as e:
            log_error(f"Feature extraction failed. Error: {e}")
            return np.zeros(100)  # Return zero vector as fallback
    
    def cluster_speakers(self, features: List[np.ndarray]) -> List[str]:
        """
        Cluster speakers using K-means when no trained models available
        
        Args:
            features: List of feature vectors
            
        Returns:
            List[str]: Speaker labels
        """
        try:
            features_array = np.array(features)
            
            # Use K-means clustering
            kmeans = KMeans(n_clusters=self.n_speakers, random_state=42)
            cluster_labels = kmeans.fit_predict(features_array)
            
            # Convert cluster labels to speaker labels
            speaker_labels = [f"Speaker {label + 1}" for label in cluster_labels]
            
            log_info(f"Speaker clustering completed. Segments: {len(features)}, Speakers: {self.n_speakers}, Distribution: {dict(zip(*np.unique(speaker_labels, return_counts=True)))}")
            
            return speaker_labels
            
        except Exception as e:
            log_error(f"Speaker clustering failed. Error: {e}")
            # Fallback to alternating pattern
            return [f"Speaker {(i % self.n_speakers) + 1}" for i in range(len(features))]
    
    def process_with_advanced_diarization(self, audio_path: str, segments: List) -> List[Dict]:
        """
        Process audio with advanced speaker diarization
        
        Args:
            audio_path: Path to audio file
            segments: List of transcription segments
            
        Returns:
            List[Dict]: Processed segments with speaker labels
        """
        try:
            log_info("Starting advanced speaker diarization")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Extract segment info and create audio segments
            segment_info = []
            
            for i, segment in enumerate(segments):
                if hasattr(segment, 'start'):
                    start_time = segment.start
                    end_time = segment.end
                    text = segment.text
                    confidence = getattr(segment, 'confidence', 1.0)
                else:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '')
                    confidence = segment.get('confidence', 1.0)
                
                # Extract audio segment for this time window
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                if start_sample < len(audio) and end_sample <= len(audio):
                    audio_segment = audio[start_sample:end_sample]
                    
                    segment_info.append({
                        'index': i,
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'confidence': confidence,
                        'audio_segment': audio_segment
                    })
                else:
                    # Fallback for invalid time ranges
                    segment_info.append({
                        'index': i,
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'confidence': confidence,
                        'audio_segment': None
                    })
            
            # Extract features for all segments
            features = []
            valid_segments = []
            
            for info in segment_info:
                if info['audio_segment'] is not None and len(info['audio_segment']) > 0:
                    try:
                        feature_vector = self.extract_speaker_features(info['audio_segment'])
                        features.append(feature_vector)
                        valid_segments.append(info)
                    except Exception as e:
                        log_warning(f"Feature extraction failed for segment {info['index']}. Error: {e}")
                        # Add fallback feature vector
                        features.append(np.zeros(100))
                        valid_segments.append(info)
                else:
                    # Add fallback for segments without audio
                    features.append(np.zeros(100))
                    valid_segments.append(info)
            
            # Cluster speakers
            if len(features) > 0:
                speaker_labels = self.cluster_speakers(features)
            else:
                # Fallback if no features extracted
                speaker_labels = [f"Speaker {(i % self.n_speakers) + 1}" for i in range(len(valid_segments))]
            
            # Create processed segments with speaker labels
            processed_segments = []
            for i, info in enumerate(valid_segments):
                if i < len(speaker_labels):
                    speaker = speaker_labels[i]
                else:
                    speaker = f"Speaker {(i % self.n_speakers) + 1}"  # Fallback
                
                processed_segment = {
                    "start": info['start'],
                    "end": info['end'],
                    "text": info['text'],
                    "confidence": info['confidence'],
                    "speaker": speaker,
                    "speaker_type": "Agent" if speaker == "Speaker 1" else "Caller",
                    "segment_id": info['index'] + 1,
                    "diarization_method": "advanced_ml"
                }
                
                processed_segments.append(processed_segment)
            
            log_info(f"Advanced diarization completed successfully. Segments processed: {len(processed_segments)}, Method: advanced_ml")
            
            return processed_segments
            
        except Exception as e:
            log_error(f"Advanced diarization failed. Error: {e}")
            # Return empty list on failure
            return []
    
    # Helper methods for feature extraction
    def _extract_formants(self, audio: np.ndarray) -> Dict:
        """Extract formant frequencies (vowel resonances)"""
        try:
            frame_length = int(0.025 * self.target_sr)  # 25ms frames
            hop_length = int(0.010 * self.target_sr)    # 10ms hop
            
            formants = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                frame = frame * np.hamming(len(frame))
                lpc_coeffs = librosa.lpc(frame, order=12)
                roots = np.roots(lpc_coeffs)
                angles = np.angle(roots)
                freqs = angles * self.target_sr / (2 * np.pi)
                freqs = freqs[freqs > 0]
                freqs = np.sort(freqs)
                
                if len(freqs) >= 3:
                    formants.append(freqs[:3])
            
            if formants:
                formants = np.array(formants)
                return {
                    'f1_mean': np.mean(formants[:, 0]) if len(formants) > 0 else 0,
                    'f2_mean': np.mean(formants[:, 1]) if len(formants) > 0 else 0,
                    'f3_mean': np.mean(formants[:, 2]) if len(formants) > 0 else 0,
                    'f1_std': np.std(formants[:, 0]) if len(formants) > 0 else 0,
                    'f2_std': np.std(formants[:, 1]) if len(formants) > 0 else 0,
                    'f3_std': np.std(formants[:, 2]) if len(formants) > 0 else 0
                }
            else:
                return {'f1_mean': 0, 'f2_mean': 0, 'f3_mean': 0, 'f1_std': 0, 'f2_std': 0, 'f3_std': 0}
                
        except Exception as e:
            return {'f1_mean': 0, 'f2_mean': 0, 'f3_mean': 0, 'f1_std': 0, 'f2_std': 0, 'f3_std': 0}
    
    def _extract_prosody(self, audio: np.ndarray) -> Dict:
        """Extract prosody features (rhythm, stress, intonation)"""
        try:
            energy = librosa.feature.rms(y=audio)
            zcr = librosa.feature.zero_crossing_rate(audio)
            speaking_rate = self._estimate_speaking_rate(audio)
            intonation = self._extract_intonation(audio)
            
            return {
                'energy_mean': np.mean(energy),
                'energy_std': np.std(energy),
                'zcr_mean': np.mean(zcr),
                'speaking_rate': speaking_rate,
                'intonation_mean': np.mean(intonation) if len(intonation) > 0 else 0,
                'intonation_std': np.std(intonation) if len(intonation) > 0 else 0
            }
            
        except Exception as e:
            return {
                'energy_mean': 0, 'energy_std': 0, 'zcr_mean': 0,
                'speaking_rate': 0, 'intonation_mean': 0, 'intonation_std': 0
            }
    
    def _extract_voice_quality(self, audio: np.ndarray) -> Dict:
        """Extract voice quality features"""
        try:
            jitter = self._calculate_jitter(audio)
            shimmer = self._calculate_shimmer(audio)
            hnr = self._calculate_hnr(audio)
            
            return {
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr
            }
            
        except Exception as e:
            return {'jitter': 0, 'shimmer': 0, 'hnr': 0}
    
    def _analyze_acoustics(self, audio: np.ndarray) -> Dict:
        """Comprehensive acoustic analysis"""
        try:
            analysis = {}
            
            # Spectral Analysis
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)
            
            analysis['spectral'] = {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth))
            }
            
            return analysis
            
        except Exception as e:
            return {}
    
    def _estimate_speaking_rate(self, audio: np.ndarray) -> float:
        """Estimate speaking rate in words per minute"""
        try:
            energy = librosa.feature.rms(y=audio)
            peaks, _ = signal.find_peaks(energy.flatten(), height=np.mean(energy))
            duration = len(audio) / self.target_sr
            word_rate = len(peaks) / duration * 60
            return min(word_rate, 300)
        except Exception as e:
            return 150.0
    
    def _extract_intonation(self, audio: np.ndarray) -> np.ndarray:
        """Extract intonation contour"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.target_sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                if magnitudes[index, t] > np.percentile(magnitudes, 85):
                    pitch_values.append(pitches[index, t])
            return np.array(pitch_values)
        except Exception as e:
            return np.array([])
    
    def _calculate_jitter(self, audio: np.ndarray) -> float:
        """Calculate jitter (pitch variation)"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.target_sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            if len(pitch_values) < 2:
                return 0.0
            
            jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values)
            return float(jitter)
        except Exception as e:
            return 0.0
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """Calculate shimmer (amplitude variation)"""
        try:
            frame_length = int(0.025 * self.target_sr)
            hop_length = int(0.010 * self.target_sr)
            
            amplitudes = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                amplitude = np.sqrt(np.mean(frame ** 2))
                amplitudes.append(amplitude)
            
            amplitudes = np.array(amplitudes)
            
            if len(amplitudes) < 2:
                return 0.0
            
            shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
            return float(shimmer)
        except Exception as e:
            return 0.0
    
    def _calculate_hnr(self, audio: np.ndarray) -> float:
        """Calculate harmonics-to-noise ratio"""
        try:
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(audio ** 2) * 0.1
            hnr = 10 * np.log10(signal_power / noise_power)
            return float(hnr)
        except Exception as e:
            return 0.0

def main():
    """Example usage of the unified audio processor"""
    
    # Initialize processor
    processor = UnifiedAudioProcessorLocal(n_speakers=2)
    
    # Example: Preprocess audio
    input_file = "Call_recordings_AI/results/preprocessed_audio/sample.wav"
    output_file = "Call_recordings_AI/results/preprocessed_audio/processed_sample.wav"
    
    if os.path.exists(input_file):
        # Preprocess audio
        success = processor.preprocess_audio(input_file, output_file)
        
        if success:
            print("Audio preprocessing completed successfully")
            
            # Example: Analyze audio quality
            quality = processor.analyze_audio_quality(output_file)
            print(f"Audio quality metrics: {quality}")
        else:
            print("Audio preprocessing failed")
    else:
        print(f"Input file not found: {input_file}")

if __name__ == "__main__":
    main() 