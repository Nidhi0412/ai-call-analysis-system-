import os
import glob
import librosa
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Common infrastructure imports
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

class PreprocessedAudioAnalyzer:
    """
    Analyze saved preprocessed audio files for transcription quality insights
    """
    
    def __init__(self):
        self.saved_dir = "Call_recordings_AI/saved_preprocessed_audio"
        self.analysis_results = {}
        
    def list_saved_files(self):
        """
        List all saved preprocessed audio files
        
        Returns:
            list: List of saved audio files
        """
        if not os.path.exists(self.saved_dir):
            print(f"❌ Saved directory not found: {self.saved_dir}")
            return []
        
        audio_files = glob.glob(os.path.join(self.saved_dir, "*.wav"))
        audio_files.extend(glob.glob(os.path.join(self.saved_dir, "*.mp3")))
        
        print(f"📁 Found {len(audio_files)} saved preprocessed audio files:")
        for i, file_path in enumerate(audio_files, 1):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {i}. {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        return audio_files
    
    def analyze_audio_file(self, file_path: str) -> dict:
        """
        Analyze a single preprocessed audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            dict: Analysis results
        """
        try:
            print(f"🔍 Analyzing: {os.path.basename(file_path)}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=16000)
            
            # Basic audio metrics
            duration = len(y) / sr
            rms = np.sqrt(np.mean(y**2))
            
            # Spectral analysis
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Calculate SNR
            noise_floor = np.percentile(magnitude, 10)
            signal_level = np.percentile(magnitude, 90)
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            
            # Voice activity detection
            energy = librosa.feature.rms(y=y).flatten()
            energy_threshold = np.percentile(energy, 30)
            speech_segments = energy > energy_threshold
            speech_ratio = np.sum(speech_segments) / len(speech_segments)
            
            # Silence analysis
            silence_threshold = np.percentile(energy, 20)
            silence_segments = energy < silence_threshold
            silence_ratio = np.sum(silence_segments) / len(silence_segments)
            
            # Pitch analysis (for voice quality)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            analysis = {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "file_size_kb": os.path.getsize(file_path) / 1024,
                "duration_seconds": float(duration),
                "sample_rate": sr,
                "rms": float(rms),
                "snr_db": float(snr_db),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "spectral_bandwidth": float(spectral_bandwidth),
                "speech_ratio": float(speech_ratio),
                "silence_ratio": float(silence_ratio),
                "pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0,
                "pitch_std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Quality assessment
            quality_score = self._calculate_quality_score(analysis)
            analysis["quality_score"] = quality_score
            analysis["quality_assessment"] = self._assess_quality(quality_score)
            analysis["transcription_readiness"] = self._assess_transcription_readiness(analysis)
            
            return analysis
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "preprocessed_audio_analyzer",
                "logData": {
                    "message": "Audio analysis failed",
                    "file": file_path,
                    "error": str(e)
                }
            })
            return {"error": str(e), "file_path": file_path}
    
    def _calculate_quality_score(self, analysis: dict) -> float:
        """
        Calculate overall quality score (0-100)
        
        Args:
            analysis: Audio analysis results
            
        Returns:
            float: Quality score (0-100)
        """
        score = 0.0
        
        # SNR contribution (40 points)
        snr_db = analysis.get("snr_db", 0)
        if snr_db > 20:
            score += 40
        elif snr_db > 15:
            score += 30
        elif snr_db > 10:
            score += 20
        elif snr_db > 5:
            score += 10
        
        # Speech ratio contribution (30 points)
        speech_ratio = analysis.get("speech_ratio", 0)
        if 0.3 <= speech_ratio <= 0.8:
            score += 30
        elif 0.2 <= speech_ratio <= 0.9:
            score += 20
        elif 0.1 <= speech_ratio <= 0.95:
            score += 10
        
        # Silence ratio contribution (20 points)
        silence_ratio = analysis.get("silence_ratio", 0)
        if 0.1 <= silence_ratio <= 0.7:
            score += 20
        elif 0.05 <= silence_ratio <= 0.8:
            score += 15
        elif 0.02 <= silence_ratio <= 0.9:
            score += 10
        
        # RMS contribution (10 points)
        rms = analysis.get("rms", 0)
        if 0.01 <= rms <= 0.5:
            score += 10
        elif 0.005 <= rms <= 1.0:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def _assess_quality(self, quality_score: float) -> str:
        """
        Assess audio quality based on score
        
        Args:
            quality_score: Quality score (0-100)
            
        Returns:
            str: Quality assessment
        """
        if quality_score >= 80:
            return "Excellent"
        elif quality_score >= 60:
            return "Good"
        elif quality_score >= 40:
            return "Fair"
        elif quality_score >= 20:
            return "Poor"
        else:
            return "Very Poor"
    
    def _assess_transcription_readiness(self, analysis: dict) -> dict:
        """
        Assess if audio is ready for transcription
        
        Args:
            analysis: Audio analysis results
            
        Returns:
            dict: Transcription readiness assessment
        """
        readiness = {
            "overall": "Unknown",
            "issues": [],
            "strengths": [],
            "recommendations": []
        }
        
        # Check SNR
        snr_db = analysis.get("snr_db", 0)
        if snr_db < 10:
            readiness["issues"].append("Low SNR - poor signal quality")
            readiness["recommendations"].append("Apply noise reduction")
        elif snr_db > 15:
            readiness["strengths"].append("Good SNR - clear signal")
        
        # Check speech ratio
        speech_ratio = analysis.get("speech_ratio", 0)
        if speech_ratio < 0.2:
            readiness["issues"].append("Too little speech content")
            readiness["recommendations"].append("Check if audio contains speech")
        elif speech_ratio > 0.9:
            readiness["issues"].append("Too much speech - may be noisy")
            readiness["recommendations"].append("Apply silence removal")
        else:
            readiness["strengths"].append("Good speech-to-silence ratio")
        
        # Check silence ratio
        silence_ratio = analysis.get("silence_ratio", 0)
        if silence_ratio < 0.05:
            readiness["issues"].append("Very little silence - may be continuous noise")
        elif silence_ratio > 0.8:
            readiness["issues"].append("Too much silence")
            readiness["recommendations"].append("Remove excessive silence")
        
        # Overall assessment
        if len(readiness["issues"]) == 0:
            readiness["overall"] = "Ready for transcription"
        elif len(readiness["issues"]) <= 2:
            readiness["overall"] = "May work with issues"
        else:
            readiness["overall"] = "Not ready - needs preprocessing"
        
        return readiness
    
    def analyze_all_files(self):
        """
        Analyze all saved preprocessed audio files
        """
        audio_files = self.list_saved_files()
        
        if not audio_files:
            print("❌ No saved audio files found")
            return
        
        print(f"\n🔍 Analyzing {len(audio_files)} files...")
        
        for file_path in audio_files:
            analysis = self.analyze_audio_file(file_path)
            
            if "error" not in analysis:
                self.analysis_results[file_path] = analysis
                self._print_analysis_summary(analysis)
            else:
                print(f"❌ Failed to analyze {os.path.basename(file_path)}: {analysis['error']}")
        
        # Save analysis results
        self._save_analysis_results()
        
        # Print overall summary
        self._print_overall_summary()
    
    def _print_analysis_summary(self, analysis: dict):
        """
        Print summary of analysis results
        
        Args:
            analysis: Analysis results
        """
        print(f"\n📊 Analysis Summary for: {analysis['filename']}")
        print(f"  Duration: {analysis['duration_seconds']:.1f} seconds")
        print(f"  File Size: {analysis['file_size_kb']:.1f} KB")
        print(f"  SNR: {analysis['snr_db']:.1f} dB")
        print(f"  Speech Ratio: {analysis['speech_ratio']:.2f}")
        print(f"  Quality Score: {analysis['quality_score']:.1f}/100 ({analysis['quality_assessment']})")
        
        readiness = analysis['transcription_readiness']
        print(f"  Transcription Readiness: {readiness['overall']}")
        
        if readiness['strengths']:
            print(f"  ✅ Strengths: {', '.join(readiness['strengths'])}")
        
        if readiness['issues']:
            print(f"  ⚠️ Issues: {', '.join(readiness['issues'])}")
        
        if readiness['recommendations']:
            print(f"  💡 Recommendations: {', '.join(readiness['recommendations'])}")
    
    def _save_analysis_results(self):
        """
        Save analysis results to JSON file
        """
        try:
            output_file = os.path.join(self.saved_dir, "analysis_results.json")
            
            # Convert numpy types to native Python types
            serializable_results = {}
            for file_path, analysis in self.analysis_results.items():
                serializable_results[file_path] = {}
                for key, value in analysis.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_results[file_path][key] = float(value)
                    else:
                        serializable_results[file_path][key] = value
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Analysis results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save analysis results: {e}")
    
    def _print_overall_summary(self):
        """
        Print overall summary of all analyzed files
        """
        if not self.analysis_results:
            return
        
        print(f"\n📈 Overall Summary")
        print(f"  Files Analyzed: {len(self.analysis_results)}")
        
        quality_scores = [analysis['quality_score'] for analysis in self.analysis_results.values()]
        avg_quality = np.mean(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        print(f"  Average Quality Score: {avg_quality:.1f}/100")
        print(f"  Quality Range: {min_quality:.1f} - {max_quality:.1f}")
        
        # Count by readiness
        readiness_counts = {}
        for analysis in self.analysis_results.values():
            readiness = analysis['transcription_readiness']['overall']
            readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        
        print(f"  Transcription Readiness:")
        for readiness, count in readiness_counts.items():
            print(f"    {readiness}: {count} files")
        
        # Best and worst files
        best_file = max(self.analysis_results.values(), key=lambda x: x['quality_score'])
        worst_file = min(self.analysis_results.values(), key=lambda x: x['quality_score'])
        
        print(f"\n🏆 Best Quality: {best_file['filename']} ({best_file['quality_score']:.1f}/100)")
        print(f"⚠️ Worst Quality: {worst_file['filename']} ({worst_file['quality_score']:.1f}/100)")

def main():
    """Main function to run the analysis"""
    analyzer = PreprocessedAudioAnalyzer()
    
    print("🎵 Preprocessed Audio Analysis Tool")
    print("=" * 50)
    
    # Analyze all saved files
    analyzer.analyze_all_files()
    
    print(f"\n💡 Tips for analyzing Hindi transcription quality:")
    print(f"  1. Listen to the saved preprocessed audio files")
    print(f"  2. Compare with original files to see what preprocessing did")
    print(f"  3. Check if denoising improved or hurt the audio")
    print(f"  4. Look for clear speech vs. noise patterns")
    print(f"  5. Analyze if silence removal was appropriate")
    print(f"  6. Check if the audio is now suitable for Whisper")

if __name__ == "__main__":
    main() 