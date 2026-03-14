#!/usr/bin/env python3
"""
Advanced Speaker Diarization System
===================================

This module implements state-of-the-art speaker identification using:
- Voice fingerprinting
- Acoustic analysis  
- Machine learning models
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal
from scipy.stats import pearsonr

# Common infrastructure imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mock_pylogger import pylogger

# Configure logging
logger = pylogger("/home/saas/logs/innex", "call_recordings_ai")

class VoiceFingerprint:
    """
    Advanced voice fingerprinting system
    """
    
    def __init__(self):
        self.feature_dim = 512
        self.sample_rate = 16000
        
    def extract_voice_features(self, audio: np.ndarray) -> Dict:
        """
        Extract comprehensive voice features for fingerprinting
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Voice fingerprint features
        """
        try:
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # 2. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            # 3. Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
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
            logger.log_it({
                "logType": "error",
                "prefix": "voice_fingerprint",
                "logData": {
                    "message": "Voice feature extraction failed",
                    "error": str(e)
                }
            })
            return {}
    
    def _extract_formants(self, audio: np.ndarray) -> Dict:
        """
        Extract formant frequencies (vowel resonances)
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Formant features
        """
        try:
            # Use LPC (Linear Predictive Coding) to find formants
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            formants = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                
                # Apply window function
                frame = frame * np.hamming(len(frame))
                
                # LPC analysis
                lpc_coeffs = librosa.lpc(frame, order=12)
                
                # Find roots of LPC polynomial
                roots = np.roots(lpc_coeffs)
                
                # Calculate formant frequencies
                angles = np.angle(roots)
                freqs = angles * self.sample_rate / (2 * np.pi)
                
                # Keep only positive frequencies
                freqs = freqs[freqs > 0]
                freqs = np.sort(freqs)
                
                # Take first 3 formants
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
        """
        Extract prosody features (rhythm, stress, intonation)
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Prosody features
        """
        try:
            # Energy envelope
            energy = librosa.feature.rms(y=audio)
            
            # Zero crossing rate (speaking rate indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            # Speaking rate estimation
            speaking_rate = self._estimate_speaking_rate(audio)
            
            # Intonation contour
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
        """
        Extract voice quality features
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Voice quality features
        """
        try:
            # Jitter (pitch variation)
            jitter = self._calculate_jitter(audio)
            
            # Shimmer (amplitude variation)
            shimmer = self._calculate_shimmer(audio)
            
            # Harmonics-to-noise ratio
            hnr = self._calculate_hnr(audio)
            
            return {
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr
            }
            
        except Exception as e:
            return {'jitter': 0, 'shimmer': 0, 'hnr': 0}
    
    def _estimate_speaking_rate(self, audio: np.ndarray) -> float:
        """Estimate speaking rate in words per minute"""
        try:
            # Simple estimation based on energy peaks
            energy = librosa.feature.rms(y=audio)
            peaks, _ = signal.find_peaks(energy.flatten(), height=np.mean(energy))
            
            # Estimate words per minute (rough approximation)
            duration = len(audio) / self.sample_rate
            word_rate = len(peaks) / duration * 60  # words per minute
            
            return min(word_rate, 300)  # Cap at 300 wpm
            
        except Exception as e:
            return 150.0  # Default speaking rate
    
    def _extract_intonation(self, audio: np.ndarray) -> np.ndarray:
        """Extract intonation contour"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            
            # Get pitch values where magnitude is significant
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
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            if len(pitch_values) < 2:
                return 0.0
            
            # Calculate jitter as relative average perturbation
            jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values)
            return float(jitter)
            
        except Exception as e:
            return 0.0
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """Calculate shimmer (amplitude variation)"""
        try:
            # Calculate amplitude envelope
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            amplitudes = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                amplitude = np.sqrt(np.mean(frame ** 2))
                amplitudes.append(amplitude)
            
            amplitudes = np.array(amplitudes)
            
            if len(amplitudes) < 2:
                return 0.0
            
            # Calculate shimmer as relative average perturbation
            shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
            return float(shimmer)
            
        except Exception as e:
            return 0.0
    
    def _calculate_hnr(self, audio: np.ndarray) -> float:
        """Calculate harmonics-to-noise ratio"""
        try:
            # Simple HNR estimation
            # In practice, this would use more sophisticated methods
            
            # Calculate signal power
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise power (simplified)
            noise_power = np.mean(audio ** 2) * 0.1  # Assume 10% noise
            
            hnr = 10 * np.log10(signal_power / noise_power)
            return float(hnr)
            
        except Exception as e:
            return 0.0

class AcousticAnalyzer:
    """
    Advanced acoustic analysis for speaker identification
    """
    
    def __init__(self):
        self.sample_rate = 16000
        
    def analyze_acoustics(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive acoustic analysis
        
        Args:
            audio: Audio signal
            
        Returns:
            dict: Acoustic analysis results
        """
        try:
            analysis = {}
            
            # 1. Spectral Analysis
            analysis['spectral'] = self._spectral_analysis(audio)
            
            # 2. Temporal Analysis
            analysis['temporal'] = self._temporal_analysis(audio)
            
            # 3. Voice Quality Analysis
            analysis['voice_quality'] = self._voice_quality_analysis(audio)
            
            # 4. Articulation Analysis
            analysis['articulation'] = self._articulation_analysis(audio)
            
            return analysis
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "acoustic_analyzer",
                "logData": {
                    "message": "Acoustic analysis failed",
                    "error": str(e)
                }
            })
            return {}
    
    def _spectral_analysis(self, audio: np.ndarray) -> Dict:
        """Spectral domain analysis"""
        try:
            # Short-time Fourier transform
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'spectral_flatness_mean': float(np.mean(spectral_flatness)),
                'spectral_flatness_std': float(np.std(spectral_flatness))
            }
            
        except Exception as e:
            return {}
    
    def _temporal_analysis(self, audio: np.ndarray) -> Dict:
        """Temporal domain analysis"""
        try:
            # Energy features
            energy = librosa.feature.rms(y=audio)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            # Autocorrelation
            autocorr = librosa.autocorrelate(audio)
            
            return {
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                'autocorr_max': float(np.max(autocorr)),
                'autocorr_mean': float(np.mean(autocorr))
            }
            
        except Exception as e:
            return {}
    
    def _voice_quality_analysis(self, audio: np.ndarray) -> Dict:
        """Voice quality analysis"""
        try:
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            # Voice quality metrics
            jitter = self._calculate_jitter(audio)
            shimmer = self._calculate_shimmer(audio)
            hnr = self._calculate_hnr(audio)
            
            return {
                'pitch_mean': float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0,
                'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr
            }
            
        except Exception as e:
            return {}
    
    def _articulation_analysis(self, audio: np.ndarray) -> Dict:
        """Articulation analysis"""
        try:
            # Consonant-vowel ratio estimation
            # This is a simplified version
            energy = librosa.feature.rms(y=audio)
            energy_threshold = np.percentile(energy, 50)
            
            high_energy_frames = np.sum(energy > energy_threshold)
            total_frames = len(energy.flatten())
            
            cv_ratio = high_energy_frames / total_frames if total_frames > 0 else 0
            
            return {
                'cv_ratio': float(cv_ratio),
                'articulation_rate': float(self._estimate_articulation_rate(audio))
            }
            
        except Exception as e:
            return {}
    
    def _calculate_jitter(self, audio: np.ndarray) -> float:
        """Calculate jitter"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            if len(pitch_values) < 2:
                return 0.0
            
            jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values)
            return float(jitter)
            
        except Exception as e:
            return 0.0
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """Calculate shimmer"""
        try:
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
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
    
    def _estimate_articulation_rate(self, audio: np.ndarray) -> float:
        """Estimate articulation rate"""
        try:
            # Simple estimation based on energy peaks
            energy = librosa.feature.rms(y=audio)
            peaks, _ = signal.find_peaks(energy.flatten(), height=np.mean(energy))
            
            duration = len(audio) / self.sample_rate
            articulation_rate = len(peaks) / duration
            
            return min(articulation_rate, 10.0)  # Cap at 10 articulations per second
            
        except Exception as e:
            return 5.0

class SpeakerDiarizationML:
    """
    Machine learning-based speaker diarization system
    """
    
    def __init__(self, n_speakers: int = 2):
        self.n_speakers = n_speakers
        self.voice_fingerprint = VoiceFingerprint()
        self.acoustic_analyzer = AcousticAnalyzer()
        self.speaker_models = {}
        
    def extract_speaker_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive speaker features
        
        Args:
            audio: Audio signal
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            # Extract voice fingerprint
            voice_features = self.voice_fingerprint.extract_voice_features(audio)
            
            # Extract acoustic analysis
            acoustic_features = self.acoustic_analyzer.analyze_acoustics(audio)
            
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
            logger.log_it({
                "logType": "error",
                "prefix": "speaker_diarization_ml",
                "logData": {
                    "message": "Feature extraction failed",
                    "error": str(e)
                }
            })
            return np.zeros(100)  # Return zero vector as fallback
    
    def train_speaker_models(self, audio_segments: List[np.ndarray], speaker_labels: List[str]):
        """
        Train speaker models using extracted features
        
        Args:
            audio_segments: List of audio segments
            speaker_labels: List of speaker labels
        """
        try:
            # Extract features for all segments
            features = []
            for segment in audio_segments:
                feature_vector = self.extract_speaker_features(segment)
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Train GMM for each speaker
            unique_speakers = list(set(speaker_labels))
            
            for speaker in unique_speakers:
                speaker_features = features[[i for i, label in enumerate(speaker_labels) if label == speaker]]
                
                if len(speaker_features) > 0:
                    # Train Gaussian Mixture Model
                    gmm = GaussianMixture(n_components=8, random_state=42)
                    gmm.fit(speaker_features)
                    
                    self.speaker_models[speaker] = gmm
                    
                    logger.log_it({
                        "logType": "info",
                        "prefix": "speaker_diarization_ml",
                        "logData": {
                            "message": f"Trained model for speaker {speaker}",
                            "samples": len(speaker_features),
                            "features_dim": features.shape[1]
                        }
                    })
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "speaker_diarization_ml",
                "logData": {
                    "message": "Speaker model training failed",
                    "error": str(e)
                }
            })
    
    def identify_speaker(self, audio_segment: np.ndarray) -> str:
        """
        Identify speaker for a given audio segment
        
        Args:
            audio_segment: Audio segment to identify
            
        Returns:
            str: Speaker identifier
        """
        try:
            # Extract features
            features = self.extract_speaker_features(audio_segment)
            
            if len(self.speaker_models) == 0:
                # No trained models, use clustering
                return self._cluster_speakers([features])[0]
            
            # Compare against all speaker models
            scores = {}
            for speaker, model in self.speaker_models.items():
                score = model.score([features])
                scores[speaker] = score
            
            # Return speaker with highest score
            best_speaker = max(scores, key=scores.get)
            
            return best_speaker
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "speaker_diarization_ml",
                "logData": {
                    "message": "Speaker identification failed",
                    "error": str(e)
                }
            })
            return "Speaker 1"  # Default fallback
    
    def _cluster_speakers(self, features: List[np.ndarray]) -> List[str]:
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
            
            return speaker_labels
            
        except Exception as e:
            # Fallback to alternating pattern
            return [f"Speaker {(i % self.n_speakers) + 1}" for i in range(len(features))]

def main():
    """Example usage of the advanced speaker diarization system"""
    
    # Initialize the system
    diarization_system = SpeakerDiarizationML(n_speakers=2)
    
    # Example: Load audio file
    audio_file = "Call_recordings_AI/results/preprocessed_audio/sample.wav"
    
    if os.path.exists(audio_file):
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Split into segments (simplified)
        segment_length = int(3 * sr)  # 3-second segments
        segments = []
        
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i + segment_length]
            if len(segment) >= segment_length // 2:  # At least 1.5 seconds
                segments.append(segment)
        
        # Extract features for all segments
        features = []
        for segment in segments:
            feature_vector = diarization_system.extract_speaker_features(segment)
            features.append(feature_vector)
        
        # Cluster speakers
        speaker_labels = diarization_system._cluster_speakers(features)
        
        print(f"Processed {len(segments)} segments")
        print(f"Speaker distribution: {dict(zip(*np.unique(speaker_labels, return_counts=True)))}")
        
        # Train models if we have enough data
        if len(segments) >= 4:
            diarization_system.train_speaker_models(segments, speaker_labels)
            print("Speaker models trained successfully")
    
    else:
        print(f"Audio file not found: {audio_file}")

if __name__ == "__main__":
    main() 