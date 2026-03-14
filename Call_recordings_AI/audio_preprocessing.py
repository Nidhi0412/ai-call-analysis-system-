import os
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Audio processing libraries
from pydub import AudioSegment, silence
import librosa
import soundfile as sf
import noisereduce as nr

# Common infrastructure imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mock_pylogger import pylogger

# Configure logging using pylogger
logger = pylogger("/home/saas/logs/innex", "call_recordings_ai")

class AudioPreprocessor:
    """
    Comprehensive audio preprocessing for call recordings
    Handles mono conversion, resampling, smart denoising, and silence removal
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 silence_threshold_db: int = -14,
                 min_silence_len: int = 1500,
                 keep_silence: int = 300,
                 snr_threshold: float = 15.0,
                 noise_threshold: float = 0.1):
        """
        Initialize the audio preprocessor
        
        Args:
            target_sr: Target sample rate (default: 16000 for Whisper)
            silence_threshold_db: Silence threshold in dB relative to audio level
            min_silence_len: Minimum silence length to remove (ms)
            keep_silence: Amount of silence to keep around speech (ms)
            snr_threshold: SNR threshold above which denoising is skipped
            noise_threshold: Noise level threshold for denoising decision
        """
        self.target_sr = target_sr
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.snr_threshold = snr_threshold
        self.noise_threshold = noise_threshold
        
        logger.log_it({
            "logType": "info",
            "prefix": "audio_preprocessor",
            "logData": {
                "message": "AudioPreprocessor initialized with smart denoising",
                "target_sr": target_sr,
                "silence_threshold_db": silence_threshold_db,
                "min_silence_len": min_silence_len,
                "keep_silence": keep_silence,
                "snr_threshold": snr_threshold,
                "noise_threshold": noise_threshold
            }
        })
    
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
            # Use spectral analysis to estimate noise floor
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Estimate noise floor from lower frequency bands
            noise_floor = np.percentile(magnitude, 10)
            signal_level = np.percentile(magnitude, 90)
            
            # Calculate SNR
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            # Analyze frequency spectrum
            freqs = librosa.fft_frequencies(sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            
            # Detect background noise characteristics
            # Look for consistent low-level noise
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
            
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": {
                    "message": "Audio quality analysis completed",
                    "file": os.path.basename(audio_path),
                    "needs_denoising": needs_denoising,
                    "snr_db": f"{snr_db:.1f}",
                    "reasons": quality_metrics["denoising_reason"]
                }
            })
            
            return quality_metrics
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "audio_preprocessor",
                "logData": {
                    "message": "Audio quality analysis failed",
                    "file": os.path.basename(audio_path),
                    "error": str(e)
                }
            })
            # Default to denoising if analysis fails
            return {
                "needs_denoising": True,
                "denoising_reason": ["Analysis failed, defaulting to denoising"],
                "error": str(e)
            }
    
    def preprocess_single_file(self, input_path: str, output_path: str) -> bool:
        """
        Preprocess a single audio file with smart denoising
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save preprocessed audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        temp_mono_path = None
        temp_denoised_path = None
        
        try:
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": f"Processing: {input_path}"
            })
            
            # Step 1: Load and convert to mono, resample
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": "Step 1: Converting to mono and resampling..."
            })
            sound = AudioSegment.from_wav(input_path)
            sound = sound.set_channels(1).set_frame_rate(self.target_sr)
            
            # Step 2: Export to temporary file for analysis and processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_mono:
                sound.export(temp_mono.name, format="wav")
                temp_mono_path = temp_mono.name
            
            # Step 3: Analyze audio quality
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": "Step 2: Analyzing audio quality..."
            })
            quality_metrics = self.analyze_audio_quality(temp_mono_path)
            
            # Step 4: Smart denoising decision
            if quality_metrics.get("needs_denoising", True):
                logger.log_it({
                    "logType": "info",
                    "prefix": "audio_preprocessor",
                    "logData": {
                        "message": "Step 3: Denoising audio (quality analysis indicated need)",
                        "reasons": quality_metrics.get("denoising_reason", [])
                    }
                })
                
                try:
                    y, sr = librosa.load(temp_mono_path, sr=self.target_sr)
                    y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False)
                    
                    # Export denoised audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_denoised:
                        sf.write(temp_denoised.name, y_denoised, sr)
                        temp_denoised_path = temp_denoised.name
                        
                    logger.log_it({
                        "logType": "info",
                        "prefix": "audio_preprocessor",
                        "logData": "Denoising completed successfully"
                    })
                    
                except Exception as denoise_error:
                    logger.log_it({
                        "logType": "warning",
                        "prefix": "audio_preprocessor",
                        "logData": {
                            "message": "Denoising failed, proceeding without denoising",
                            "error": str(denoise_error)
                        }
                    })
                    # If denoising fails, use the original audio
                    temp_denoised_path = temp_mono_path
            else:
                logger.log_it({
                    "logType": "info",
                    "prefix": "audio_preprocessor",
                    "logData": {
                        "message": "Step 3: Skipping denoising (audio quality is good)",
                        "snr_db": quality_metrics.get("snr_db", 0)
                    }
                })
                # Skip denoising, use mono audio directly
                temp_denoised_path = temp_mono_path
            
            # Step 5: Remove long silences
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": "Step 4: Removing long silences..."
            })
            processed = AudioSegment.from_wav(temp_denoised_path)
            silence_thresh = processed.dBFS + self.silence_threshold_db
            
            chunks = silence.split_on_silence(
                processed,
                min_silence_len=self.min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=self.keep_silence
            )
            
            # Step 6: Combine chunks and export final audio
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": "Step 5: Combining audio chunks..."
            })
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
            
            logger.log_it({
                "logType": "info",
                "prefix": "audio_preprocessor",
                "logData": {
                    "message": "Successfully processed with smart denoising",
                    "output_path": output_path,
                    "original_duration": len(sound),
                    "final_duration": len(final_audio),
                    "chunks_created": len(chunks) if chunks else 0,
                    "denoising_applied": quality_metrics.get("needs_denoising", True),
                    "quality_metrics": quality_metrics
                }
            })
            return True
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "audio_preprocessor",
                "logData": {
                    "message": "Error processing file",
                    "input_path": input_path,
                    "error": str(e)
                }
            })
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
    
    def batch_preprocess(self, input_dir: str, output_dir: str, use_parallel: bool = True) -> dict:
        """
        Preprocess all .wav files in a directory using parallel or sequential processing
        
        Args:
            input_dir: Directory containing input .wav files
            output_dir: Directory to save preprocessed files
            use_parallel: Whether to use parallel processing (default: True)
            
        Returns:
            dict: Statistics about the processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        wav_files = list(input_path.glob("*.wav"))
        total_files = len(wav_files)
        successful = 0
        failed = 0
        
        logger.log_it({
            "logType": "info",
            "prefix": "audio_preprocessor",
            "logData": {
                "message": f"Starting {'parallel' if use_parallel else 'sequential'} batch preprocessing",
                "total_files": total_files,
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "max_workers": min(3, total_files) if use_parallel else 1
            }
        })
        
        if use_parallel and total_files > 1:
            # Use parallel processing for faster execution
            with ThreadPoolExecutor(max_workers=min(3, total_files)) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.preprocess_single_file, str(wav_file), str(output_path / wav_file.name)): wav_file 
                    for wav_file in wav_files
                }
                
                # Process completed tasks with progress tracking
                for i, future in enumerate(as_completed(future_to_file), 1):
                    wav_file = future_to_file[future]
                    try:
                        if future.result():
                            successful += 1
                            logger.log_it({
                                "logType": "info",
                                "prefix": "audio_preprocessor",
                                "logData": {
                                    "message": f"Processed successfully ({i}/{total_files})",
                                    "file": wav_file.name,
                                    "progress": f"{i}/{total_files} ({i/total_files*100:.1f}%)"
                                }
                            })
                        else:
                            failed += 1
                            logger.log_it({
                                "logType": "error",
                                "prefix": "audio_preprocessor",
                                "logData": {
                                    "message": f"Failed to process ({i}/{total_files})",
                                    "file": wav_file.name,
                                    "progress": f"{i}/{total_files} ({i/total_files*100:.1f}%)"
                                }
                            })
                    except Exception as e:
                        failed += 1
                        logger.log_it({
                            "logType": "error",
                            "prefix": "audio_preprocessor",
                            "logData": {
                                "message": f"Exception processing file ({i}/{total_files})",
                                "file": wav_file.name,
                                "error": str(e),
                                "progress": f"{i}/{total_files} ({i/total_files*100:.1f}%)"
                            }
                        })
        else:
            # Sequential processing (faster for small files or when parallel causes issues)
            for i, wav_file in enumerate(wav_files, 1):
                logger.log_it({
                    "logType": "info",
                    "prefix": "audio_preprocessor",
                    "logData": {
                        "message": f"Processing file {i}/{total_files}",
                        "file": wav_file.name,
                        "progress": f"{i}/{total_files} ({i/total_files*100:.1f}%)"
                    }
                })
                
                if self.preprocess_single_file(str(wav_file), str(output_path / wav_file.name)):
                    successful += 1
                    logger.log_it({
                        "logType": "info",
                        "prefix": "audio_preprocessor",
                        "logData": {
                            "message": f"Processed successfully ({i}/{total_files})",
                            "file": wav_file.name
                        }
                    })
                else:
                    failed += 1
                    logger.log_it({
                        "logType": "error",
                        "prefix": "audio_preprocessor",
                        "logData": {
                            "message": f"Failed to process ({i}/{total_files})",
                            "file": wav_file.name
                        }
                    })
        
        stats = {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_files * 100) if total_files > 0 else 0
        }
        
        logger.log_it({
            "logType": "info",
            "prefix": "audio_preprocessor",
            "logData": {
                "message": f"{'Parallel' if use_parallel else 'Sequential'} batch preprocessing completed",
                "statistics": stats,
                "processing_mode": "parallel" if use_parallel else "sequential"
            }
        })
        return stats

def main():
    """Example usage of the AudioPreprocessor"""
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Define input and output directories
    input_dir = "Top 5 Query of Jan 2025 (1)"
    output_dir = "preprocessed_audio"
    
    # Process all files
    stats = preprocessor.batch_preprocess(input_dir, output_dir)
    print(f"Processing complete: {stats}")

if __name__ == "__main__":
    main() 