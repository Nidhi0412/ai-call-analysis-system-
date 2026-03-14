from fastapi import APIRouter, FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
from pathlib import Path
import uuid
import tempfile
import asyncio
import time
import csv
import json
from datetime import datetime, timedelta
import concurrent.futures
import hashlib
from functools import lru_cache
import threading
# Import with fallback for different contexts
try:
    # Try absolute imports first (when running from main app directory)
    from Call_recordings_AI.unified_audio_processor import UnifiedAudioProcessor
    from Call_recordings_AI.transcription_with_speakers import TranscriptionWithSpeakersService
    from Call_recordings_AI.hindi_debug_helper import HindiDebugHelper
    from Call_recordings_AI.analyze_preprocessed_audio import PreprocessedAudioAnalyzer
    from Call_recordings_AI.call_analysis import CallAnalysisService
    print("✅ Using absolute imports")
except ImportError:
    try:
        # Fallback for when running from Call_recordings_AI directory
        from unified_audio_processor import UnifiedAudioProcessor
        from transcription_with_speakers import TranscriptionWithSpeakersService
        from hindi_debug_helper import HindiDebugHelper
        from analyze_preprocessed_audio import PreprocessedAudioAnalyzer
        from call_analysis import CallAnalysisService
        print("✅ Using relative imports")
    except ImportError as e:
        print(f"❌ All import attempts failed: {e}")
        # Create dummy classes to prevent import errors
        class UnifiedAudioProcessor:
            def __init__(self): pass
            def preprocess_audio(self, *args, **kwargs): return False
            def analyze_audio_quality(self, *args, **kwargs): return {}
        
        class TranscriptionWithSpeakersService:
            def __init__(self, *args, **kwargs): pass
            def transcribe_with_speakers(self, *args, **kwargs): return {"status": "error", "error": "Service not available"}
            def process_audio_file_with_speakers(self, *args, **kwargs): return {"status": "error", "error": "Service not available"}
        
        class HindiDebugHelper:
            def __init__(self): pass
            def debug_hindi_transcription(self, *args, **kwargs): return {"status": "error", "error": "Service not available"}
        
        class PreprocessedAudioAnalyzer:
            def __init__(self): pass
            def analyze_audio_file(self, *args, **kwargs): return {"status": "error", "error": "Service not available"}
        
        class CallAnalysisService:
            def __init__(self, *args, **kwargs): pass
            def analyze_call_transcription(self, *args, **kwargs): return {"status": "error", "error": "Service not available"}

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'packages'))

try:
    from config import CONFIG
except ImportError:
    # Fallback config if not available
    CONFIG = {
        'Development': {
            'APP_HOST': '0.0.0.0',
            'APP_PORT': 4000,
            'WORKERS': 1
        }
    }

try:
    from pylogger import pylogger
except ImportError:
    # Mock logger for testing
    class MockLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log_it(self, data): 
            print(f"LOG: {data}")
    pylogger = MockLogger

# Import optimized service at the top with fallback
try:
    from Call_recordings_AI.optimized_transcription_service import OptimizedTranscriptionService
    OPTIMIZED_SERVICE_AVAILABLE = True
    print("✅ Optimized transcription service loaded successfully (from main context)")
except ImportError:
    try:
        from optimized_transcription_service import OptimizedTranscriptionService
        OPTIMIZED_SERVICE_AVAILABLE = True
        print("✅ Optimized transcription service loaded successfully")
    except ImportError as e:
        print(f"⚠️ Optimized service not available: {e}")
        OPTIMIZED_SERVICE_AVAILABLE = False
        # Create dummy class
        class OptimizedTranscriptionService:
            def __init__(self, *args, **kwargs): pass
            def process_audio_optimized(self, *args, **kwargs): return {"status": "error", "error": "Optimized service not available"}
            def get_token_usage(self): return {}
            def get_performance_summary(self): return {}
        OptimizedTranscriptionService = OptimizedTranscriptionService

# Initialize optimized service if available
if OPTIMIZED_SERVICE_AVAILABLE:
    optimized_service = OptimizedTranscriptionService(max_workers=3)
else:
    optimized_service = None

logger = pylogger("/home/saas/logs/innex", "call_recordings_ai")

# Global cancellation mechanism
cancellation_requests = {}
cancellation_lock = threading.Lock()

# Base timeout configurations - Will be adjusted based on file size
BASE_TIMEOUTS = {
    "preprocessing": 120,  # 2 minutes base
    "transcription": 600,  # 10 minutes base (increased for local models)
    "analysis": 120,        # 2 minutes base
    "debug_analysis": 90,  # 90 seconds base
    "total": 900          # 15 minutes total base (increased)
}

def get_dynamic_timeouts(file_size_mb: float) -> dict:
    """Calculate dynamic timeouts based on file size"""
    # Base multipliers for different file sizes (increased for local models)
    if file_size_mb < 5:
        multiplier = 1.0  # Small files
    elif file_size_mb < 20:
        multiplier = 2.0  # Medium files (increased)
    elif file_size_mb < 50:
        multiplier = 3.0  # Large files (increased)
    else:
        multiplier = 4.0  # Very large files (increased)
    
    return {
        "preprocessing": int(BASE_TIMEOUTS["preprocessing"] * multiplier),
        "transcription": int(BASE_TIMEOUTS["transcription"] * multiplier),
        "analysis": int(BASE_TIMEOUTS["analysis"] * multiplier),
        "debug_analysis": int(BASE_TIMEOUTS["debug_analysis"] * multiplier),
        "total": int(BASE_TIMEOUTS["total"] * multiplier)
    }

def check_cancellation_request(client_ip: str) -> bool:
    """Check if processing should be cancelled for a client"""
    with cancellation_lock:
        return cancellation_requests.get(client_ip, False)

def get_file_hash(file_path: str) -> str:
    """Generate hash for file content to enable caching"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cache_path(file_hash: str, operation: str) -> str:
    """Get cache file path for operation"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, ".cache", "results_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{file_hash}_{operation}.json")

def load_cached_result(file_hash: str, operation: str) -> dict:
    """Load cached result if available and not expired"""
    cache_path = get_cache_path(file_hash, operation)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid (24 hours)
            cache_time = datetime.fromisoformat(cached_data.get('cached_at', ''))
            if datetime.now() - cache_time < timedelta(hours=24):
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui_cache",
                    "logData": {
                        "message": f"Cache hit for {operation}",
                        "file_hash": file_hash[:8],
                        "cache_age_hours": (datetime.now() - cache_time).total_seconds() / 3600
                    }
                })
                return cached_data.get('result', {})
        except Exception as e:
            logger.log_it({
                "logType": "warning",
                "prefix": "web_ui_cache",
                "logData": {
                    "message": f"Failed to load cache for {operation}",
                    "error": str(e)
                }
            })
    return None

def save_cached_result(file_hash: str, operation: str, result: dict):
    """Save result to cache"""
    try:
        cache_path = get_cache_path(file_hash, operation)
        cache_data = {
            'result': result,
            'cached_at': datetime.now().isoformat(),
            'operation': operation,
            'file_hash': file_hash
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui_cache",
            "logData": {
                "message": f"Result cached for {operation}",
                "file_hash": file_hash[:8],
                "cache_path": cache_path
            }
        })
    except Exception as e:
        logger.log_it({
            "logType": "warning",
            "prefix": "web_ui_cache",
            "logData": {
                "message": f"Failed to save cache for {operation}",
                "error": str(e)
            }
        })

def get_global_csv_path() -> str:
    """Get the path to the global results CSV file"""
    # Use absolute path to ensure consistency regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, ".cache", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "global_call_analysis_results.csv")
    
    # Debug: Log the path resolution
    logger.log_it({
        "logType": "info",
        "prefix": "csv_path_debug",
        "logData": {
            "message": "Global CSV path resolved",
            "current_dir": current_dir,
            "results_dir": results_dir,
            "csv_path": csv_path,
            "absolute_path": os.path.abspath(csv_path),
            "file_exists": os.path.exists(csv_path),
            "working_directory": os.getcwd()
        }
    })
    
    return csv_path

def initialize_global_csv():
    """Initialize the global CSV file with headers if it doesn't exist"""
    csv_path = get_global_csv_path()
    
    if not os.path.exists(csv_path):
        headers = [
            "Analysis ID", "File Name", "Upload Date", "File Size (MB)", "Audio Duration (seconds)", 
            "Processing Time (seconds)", "Processing Rate (real-time)", "Language", "Transcription Method",
            "Main Issue Description", "Main Issue Category", "Main Issue Urgency",
            "Support Given Description", "Support Effectiveness", "Support Steps",
            "Action Taken Description", "Action Status", "Resolution Steps",
            "Agent Tone", "Agent State", "Agent Communication",
            "Issue Resolved", "Resolution Method", "Customer Satisfaction",
            "Agent Engagement Level", "Active Listening", "Proactive Assistance", "Response Time",
            "Technical Knowledge", "Problem Solving", "Communication", "Patience",
            "Overall Satisfaction", "Recommendation Likelihood", "Key Satisfaction Factors",
            "Call Atmosphere", "Respect Level", "Conflict Present",
            "Overall Score", "Overall Rating", "Strengths", "Areas for Improvement",
            "Hotel/Restaurant Code", "Booking Details", "Billing Issues", "Technical Issues",
            "Primary Query", "Query Type", "Query Complexity", "Resolution Time",
            "Transcription Text", "English Translation", "Speaker Segments Count", "Full Transcription Text",
            "Token Usage - Transcription", "Token Usage - Translation", "Token Usage - Analysis",
            "Cost - Transcription", "Cost - Translation", "Cost - Analysis", "Total Cost",
            "Performance - Transcription Time", "Performance - Call Analysis Time", 
            "Performance - Preprocessing Time", "Performance - File Upload Time",
            "Performance - Transcription %", "Performance - Call Analysis %", 
            "Performance - Preprocessing %", "Performance - File Upload %",
            "Critical Bottlenecks", "Analysis Status", "File Hash", "Cache Status"
        ]
        
        try:
            # Create backup directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(current_dir, ".cache", "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            with open(csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(headers)
            
            # Create initial backup
            backup_path = os.path.join(backup_dir, f"global_csv_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            shutil.copy2(csv_path, backup_path)
            
            logger.log_it({
                "logType": "info",
                "prefix": "global_csv",
                "logData": {
                    "message": "Global CSV initialized with backup",
                    "csv_path": csv_path,
                    "backup_path": backup_path,
                    "headers_count": len(headers)
                }
            })
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "global_csv",
                "logData": {
                    "message": "Failed to initialize global CSV",
                    "error": str(e),
                    "csv_path": csv_path
                }
            })
            raise

def analyze_csv_insights(csv_path: str) -> dict:
    """Analyze CSV data to extract meaningful business insights"""
    try:
        import pandas as pd
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return {"error": "No data available"}
        
        # Get CSV creation and modification info
        csv_stats = os.stat(csv_path)
        csv_created = datetime.fromtimestamp(csv_stats.st_ctime)
        csv_modified = datetime.fromtimestamp(csv_stats.st_mtime)
        
        # Check if this is a fresh/reset CSV (created recently and has few entries)
        is_fresh_csv = len(df) <= 5 and (datetime.now() - csv_created).days <= 1
        
        insights = {
            "total_calls": len(df),
            "csv_info": {
                "is_fresh": is_fresh_csv,
                "created_date": csv_created.strftime("%Y-%m-%d %H:%M:%S"),
                "last_modified": csv_modified.strftime("%Y-%m-%d %H:%M:%S"),
                "days_active": (datetime.now() - csv_created).days,
                "reset_indicator": is_fresh_csv
            },
            "agent_performance": {},
            "customer_satisfaction": {},
            "issue_analysis": {},
            "business_insights": {},
            "query_trends": {},
            "call_quality": {}
        }
        
        # Agent Performance Analysis
        if "Overall Score" in df.columns:
            scores = pd.to_numeric(df["Overall Score"], errors='coerce')
            insights["agent_performance"] = {
                "avg_score": round(scores.mean(), 1) if not scores.isna().all() else 0,
                "max_score": int(scores.max()) if not scores.isna().all() else 0,
                "min_score": int(scores.min()) if not scores.isna().all() else 0,
                "excellent_calls": len(scores[scores >= 8]) if not scores.isna().all() else 0,
                "good_calls": len(scores[(scores >= 6) & (scores < 8)]) if not scores.isna().all() else 0,
                "poor_calls": len(scores[scores < 6]) if not scores.isna().all() else 0
            }
        
        # Customer Satisfaction Analysis
        if "Overall Satisfaction" in df.columns:
            satisfaction_counts = df["Overall Satisfaction"].value_counts()
            insights["customer_satisfaction"] = {
                "very_satisfied": satisfaction_counts.get("Very Satisfied", 0),
                "satisfied": satisfaction_counts.get("Satisfied", 0),
                "neutral": satisfaction_counts.get("Neutral", 0),
                "dissatisfied": satisfaction_counts.get("Dissatisfied", 0),
                "satisfaction_rate": round((satisfaction_counts.get("Very Satisfied", 0) + satisfaction_counts.get("Satisfied", 0)) / len(df) * 100, 1)
            }
        
        # Issue Analysis
        if "Main Issue Category" in df.columns:
            issue_counts = df["Main Issue Category"].value_counts()
            insights["issue_analysis"] = {
                "top_issues": issue_counts.head(5).to_dict(),
                "total_issue_types": len(issue_counts)
            }
        
        if "Issue Resolved" in df.columns:
            resolved_counts = df["Issue Resolved"].value_counts()
            insights["issue_analysis"]["resolution_rate"] = round(resolved_counts.get("Yes", 0) / len(df) * 100, 1)
        
        # Query Trends
        if "Query Type" in df.columns:
            query_counts = df["Query Type"].value_counts()
            insights["query_trends"] = {
                "top_query_types": query_counts.head(5).to_dict(),
                "total_query_types": len(query_counts)
            }
        
        if "Query Complexity" in df.columns:
            complexity_counts = df["Query Complexity"].value_counts()
            insights["query_trends"]["complexity_distribution"] = complexity_counts.to_dict()
        
        # Business Insights
        if "Hotel/Restaurant Code" in df.columns:
            codes = df["Hotel/Restaurant Code"].dropna()
            insights["business_insights"] = {
                "unique_properties": len(codes.unique()),
                "most_common_property": codes.value_counts().index[0] if len(codes) > 0 else "None"
            }
        
        if "Billing Issues" in df.columns:
            billing_issues = df["Billing Issues"].dropna()
            insights["business_insights"]["billing_issues_count"] = len(billing_issues[billing_issues != ""])
        
        # Call Quality Analysis
        if "Call Atmosphere" in df.columns:
            atmosphere_counts = df["Call Atmosphere"].value_counts()
            insights["call_quality"] = {
                "atmosphere_distribution": atmosphere_counts.to_dict(),
                "professional_calls": atmosphere_counts.get("Professional", 0),
                "friendly_calls": atmosphere_counts.get("Friendly", 0)
            }
        
        if "Conflict Present" in df.columns:
            conflict_counts = df["Conflict Present"].value_counts()
            insights["call_quality"]["conflict_rate"] = round(conflict_counts.get("Yes", 0) / len(df) * 100, 1)
        
        # Language Analysis
        if "Language" in df.columns:
            language_counts = df["Language"].value_counts()
            insights["language_analysis"] = {
                "top_languages": language_counts.head(3).to_dict(),
                "total_languages": len(language_counts)
            }
        
        return insights
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "csv_analysis",
            "logData": {
                "message": "Error analyzing CSV insights",
                "error": str(e)
            }
        })
        return {"error": str(e)}

def get_reset_history() -> dict:
    """Get information about recent CSV resets and backups"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(current_dir, ".cache", "backups")
        
        if not os.path.exists(backup_dir):
            return {"has_backups": False, "recent_resets": []}
        
        # Get all backup files
        backup_files = []
        for filename in os.listdir(backup_dir):
            if filename.startswith("global_csv_backup_before_reset_") and filename.endswith(".csv"):
                file_path = os.path.join(backup_dir, filename)
                file_stats = os.stat(file_path)
                
                # Extract timestamp from filename
                timestamp_str = filename.replace("global_csv_backup_before_reset_", "").replace(".csv", "")
                try:
                    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except:
                    backup_time = datetime.fromtimestamp(file_stats.st_mtime)
                
                # Count rows in backup
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        row_count = sum(1 for line in f) - 1  # Subtract header
                except:
                    row_count = 0
                
                backup_files.append({
                    "filename": filename,
                    "timestamp": backup_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rows_backed_up": row_count,
                    "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "days_ago": (datetime.now() - backup_time).days
                })
        
        # Sort by timestamp (most recent first)
        backup_files.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "has_backups": len(backup_files) > 0,
            "recent_resets": backup_files[:5],  # Show last 5 resets
            "total_backups": len(backup_files)
        }
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "reset_history",
            "logData": {
                "message": "Error getting reset history",
                "error": str(e)
            }
        })
        return {"has_backups": False, "recent_resets": [], "error": str(e)}

def format_full_transcription_text(result: dict, speakers_data: dict) -> str:
    """Format the full transcription text with speaker information for CSV reference"""
    try:
        transcribed_text = result.get("transcribed_text", "")
        translated_text = result.get("translated_text", "")
        segments = speakers_data.get("segments", [])
        
        if not transcribed_text:
            return "No transcription available"
        
        # If we have speaker segments, format with speaker labels
        if segments and len(segments) > 0:
            formatted_segments = []
            for segment in segments:
                speaker_label = segment.get("speaker_type", "Unknown")
                speaker_id = segment.get("speaker", "Unknown")
                text = segment.get("text", "")
                translated = segment.get("translated_text", "")
                
                # Format: [Speaker Label (Speaker ID)]: Text
                if translated and translated != text:
                    formatted_segments.append(f"[{speaker_label} ({speaker_id})]: {text} | Translation: {translated}")
                else:
                    formatted_segments.append(f"[{speaker_label} ({speaker_id})]: {text}")
            
            return " | ".join(formatted_segments)
        else:
            # No speaker information, return simple transcription
            if translated_text and translated_text != transcribed_text:
                return f"Original: {transcribed_text} | Translation: {translated_text}"
            else:
                return transcribed_text
                
    except Exception as e:
        logger.log_it({
            "logType": "warning",
            "prefix": "csv_formatting",
            "logData": {
                "message": "Error formatting full transcription text",
                "error": str(e)
            }
        })
        return result.get("transcribed_text", "Error formatting transcription")

def append_to_global_csv(
    file_name: str,
    file_size_bytes: int,
    file_hash: str,
    call_analysis_result: dict,
    result: dict,
    speakers_data: dict,
    token_usage: dict,
    performance_summary: dict,
    cache_status: str = "new"
):
    """Append analysis result to global CSV"""
    try:
        csv_path = get_global_csv_path()
        
        # Extract detailed analysis data
        main_issue = call_analysis_result.get("main_issue", {})
        support_given = call_analysis_result.get("support_given", {})
        action_taken = call_analysis_result.get("action_taken", {})
        agent_emotion = call_analysis_result.get("agent_emotion", {})
        issue_resolution = call_analysis_result.get("issue_resolution", {})
        agent_engagement = call_analysis_result.get("agent_engagement", {})
        agent_skill = call_analysis_result.get("agent_skill", {})
        customer_satisfaction = call_analysis_result.get("customer_satisfaction", {})
        call_tone = call_analysis_result.get("call_tone", {})
        overall_call_quality = call_analysis_result.get("overall_call_quality", {})
        business_insights = call_analysis_result.get("business_insights", {})
        query_analysis = call_analysis_result.get("query_analysis", {})
        
        # Performance data
        perf_metrics = performance_summary.get("performance_metrics", {})
        step_breakdown = performance_summary.get("step_breakdown", [])
        
        # Create step timing dictionary for easy lookup
        step_times = {}
        step_percentages = {}
        for step in step_breakdown:
            step_times[step["step"]] = step["duration_seconds"]
            step_percentages[step["step"]] = step["percentage"]
        
        # Critical bottlenecks
        bottlenecks = perf_metrics.get("bottlenecks", [])
        bottleneck_text = "; ".join([f"{b['step']} ({b['percentage']:.1f}%)" for b in bottlenecks])
        
        # Generate unique analysis ID
        analysis_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"
        
        # Write comprehensive data row
        row_data = [
            analysis_id,
            file_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(file_size_bytes / 1024 / 1024, 2),
            perf_metrics.get("audio_duration", 0),
            performance_summary.get("total_processing_time", 0),
            round(perf_metrics.get("processing_rate", 0), 2),
            speakers_data.get("language", "Unknown"),
            speakers_data.get("transcription_method", "Unknown"),
            main_issue.get("description", ""),
            main_issue.get("category", ""),
            main_issue.get("urgency", ""),
            support_given.get("description", ""),
            support_given.get("effectiveness", ""),
            "; ".join(support_given.get("steps_taken", [])),
            action_taken.get("description", ""),
            action_taken.get("completion_status", ""),
            "; ".join(action_taken.get("resolution_steps", [])),
            agent_emotion.get("overall_tone", ""),
            agent_emotion.get("emotional_state", ""),
            agent_emotion.get("communication_style", ""),
            "Yes" if issue_resolution.get("resolved") else "No",
            issue_resolution.get("resolution_method", ""),
            issue_resolution.get("customer_satisfaction", ""),
            agent_engagement.get("level", ""),
            "Yes" if agent_engagement.get("active_listening") else "No",
            "Yes" if agent_engagement.get("proactive_assistance") else "No",
            agent_engagement.get("response_time", ""),
            agent_skill.get("technical_knowledge", ""),
            agent_skill.get("problem_solving", ""),
            agent_skill.get("communication", ""),
            agent_skill.get("patience", ""),
            customer_satisfaction.get("overall_satisfaction", ""),
            customer_satisfaction.get("willingness_to_recommend", ""),
            "; ".join(customer_satisfaction.get("key_satisfaction_factors", [])),
            call_tone.get("overall_atmosphere", ""),
            call_tone.get("respect_level", ""),
            "Yes" if call_tone.get("conflict_present") else "No",
            overall_call_quality.get("score", ""),
            overall_call_quality.get("rating", ""),
            "; ".join(overall_call_quality.get("strengths", [])),
            "; ".join(overall_call_quality.get("areas_for_improvement", [])),
            business_insights.get("hotel_restaurant_code", ""),
            business_insights.get("booking_details", ""),
            business_insights.get("billing_issues", ""),
            business_insights.get("technical_issues", ""),
            query_analysis.get("primary_query", ""),
            query_analysis.get("query_type", ""),
            query_analysis.get("complexity", ""),
            query_analysis.get("resolution_time", ""),
            result.get("transcribed_text", ""),
            result.get("translated_text", ""),
            len(speakers_data.get("segments", [])),
            format_full_transcription_text(result, speakers_data),  # Full transcription text with speaker info
            token_usage.get("transcription_tokens", 0),
            token_usage.get("translation_tokens", 0),
            token_usage.get("analysis_tokens", 0),
            round(token_usage.get("transcription_cost", 0), 4),
            round(token_usage.get("translation_cost", 0), 4),
            round(token_usage.get("analysis_cost", 0), 4),
            round(token_usage.get("estimated_cost_usd", 0), 4),
            step_times.get("transcription", 0),
            step_times.get("call_analysis", 0),
            step_times.get("preprocessing", 0),
            step_times.get("file_upload", 0),
            step_percentages.get("transcription", 0),
            step_percentages.get("call_analysis", 0),
            step_percentages.get("preprocessing", 0),
            step_percentages.get("file_upload", 0),
            bottleneck_text,
            "Success" if call_analysis_result.get("status") != "failed" else "Failed",
            file_hash,
            cache_status
        ]
        
        # Debug: Log before appending
        logger.log_it({
            "logType": "info",
            "prefix": "csv_append_debug",
            "logData": {
                "message": "About to append to global CSV",
                "csv_path": csv_path,
                "absolute_path": os.path.abspath(csv_path),
                "file_exists_before": os.path.exists(csv_path),
                "analysis_id": analysis_id,
                "file_name": file_name,
                "row_data_length": len(row_data)
            }
        })
        
        # Append to global CSV
        with open(csv_path, mode="a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(row_data)
        
        # Debug: Log after appending
        file_stats_after = os.stat(csv_path)
        logger.log_it({
            "logType": "info",
            "prefix": "csv_append_debug",
            "logData": {
                "message": "Successfully appended to global CSV",
                "csv_path": csv_path,
                "file_size_after_bytes": file_stats_after.st_size,
                "file_size_after_mb": round(file_stats_after.st_size / (1024 * 1024), 2),
                "last_modified_after": datetime.fromtimestamp(file_stats_after.st_mtime).isoformat()
            }
        })
        
        logger.log_it({
            "logType": "info",
            "prefix": "global_csv",
            "logData": {
                "message": "Analysis result appended to global CSV",
                "analysis_id": analysis_id,
                "file_name": file_name,
                "file_hash": file_hash[:8],
                "cache_status": cache_status,
                "csv_path": csv_path
            }
        })
        
        return analysis_id
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "global_csv",
            "logData": {
                "message": "Failed to append to global CSV",
                "file_name": file_name,
                "error": str(e)
            }
        })
        return None

class ProcessingTimer:
    """Enhanced timer with detailed performance tracking"""
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.substep_times = {}
        self.current_step = None
        self.performance_metrics = {
            "file_size": 0,
            "audio_duration": 0,
            "processing_rate": 0,
            "bottlenecks": []
        }
    
    def start_step(self, step_name):
        """Start timing a step"""
        self.current_step = step_name
        self.step_times[step_name] = {"start": time.time()}
        logger.log_it({
            "logType": "info",
            "prefix": "processing_timer",
            "logData": {
                "message": f"Starting step: {step_name}",
                "step": step_name,
                "total_elapsed": self.get_total_elapsed(),
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def end_step(self, step_name):
        """End timing a step"""
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
            duration = self.step_times[step_name]["end"] - self.step_times[step_name]["start"]
            
            # Calculate percentage of total time
            total_elapsed = self.get_total_elapsed()
            percentage = (duration / total_elapsed * 100) if total_elapsed > 0 else 0
            
            logger.log_it({
                "logType": "info",
                "prefix": "processing_timer",
                "logData": {
                    "message": f"Completed step: {step_name}",
                    "step": step_name,
                    "duration_seconds": round(duration, 2),
                    "duration_formatted": self._format_duration(duration),
                    "percentage_of_total": round(percentage, 1),
                    "total_elapsed": round(total_elapsed, 2),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Note: Bottleneck tracking will be done in get_performance_summary() 
            # to ensure accurate percentages based on final total time
    
    def start_substep(self, substep_name):
        """Start timing a substep within current step"""
        if self.current_step:
            if self.current_step not in self.substep_times:
                self.substep_times[self.current_step] = {}
            self.substep_times[self.current_step][substep_name] = {"start": time.time()}
    
    def end_substep(self, substep_name):
        """End timing a substep"""
        if self.current_step and self.current_step in self.substep_times:
            if substep_name in self.substep_times[self.current_step]:
                self.substep_times[self.current_step][substep_name]["end"] = time.time()
                duration = (self.substep_times[self.current_step][substep_name]["end"] - 
                           self.substep_times[self.current_step][substep_name]["start"])
                
                logger.log_it({
                    "logType": "info",
                    "prefix": "processing_timer_substep",
                    "logData": {
                        "message": f"Completed substep: {substep_name}",
                        "parent_step": self.current_step,
                        "substep": substep_name,
                        "duration_seconds": round(duration, 2),
                        "duration_formatted": self._format_duration(duration)
                    }
                })
    
    def set_file_metrics(self, file_size_bytes, audio_duration_seconds):
        """Set file metrics for performance calculations"""
        self.performance_metrics["file_size"] = file_size_bytes
        self.performance_metrics["audio_duration"] = audio_duration_seconds
        if audio_duration_seconds > 0:
            self.performance_metrics["processing_rate"] = audio_duration_seconds / self.get_total_elapsed()
    
    def get_total_elapsed(self):
        """Get total elapsed time"""
        return time.time() - self.start_time
    
    def get_step_duration(self, step_name):
        """Get duration of a specific step"""
        if step_name in self.step_times and "end" in self.step_times[step_name]:
            return self.step_times[step_name]["end"] - self.step_times[step_name]["start"]
        return None
    
    def get_performance_summary(self):
        """Get detailed performance summary"""
        total_time = self.get_total_elapsed()
        step_summary = []
        
        for step_name, times in self.step_times.items():
            if "end" in times:
                duration = times["end"] - times["start"]
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                step_summary.append({
                    "step": step_name,
                    "duration_seconds": round(duration, 2),
                    "duration_formatted": self._format_duration(duration),
                    "percentage": round(percentage, 1)
                })
        
        # Sort by duration (longest first)
        step_summary.sort(key=lambda x: x["duration_seconds"], reverse=True)
        
        # Calculate bottlenecks based on final percentages
        bottlenecks = []
        for step in step_summary:
            if step["percentage"] > 20:
                bottlenecks.append({
                    "step": step["step"],
                    "duration": step["duration_seconds"],
                    "percentage": step["percentage"]
                })
        
        # Update performance metrics with correct bottlenecks
        self.performance_metrics["bottlenecks"] = bottlenecks
        
        return {
            "total_processing_time": round(total_time, 2),
            "total_processing_time_formatted": self._format_duration(total_time),
            "step_breakdown": step_summary,
            "performance_metrics": self.performance_metrics,
            "slowest_steps": step_summary[:3] if len(step_summary) >= 3 else step_summary
        }
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {minutes}m {remaining_seconds:.1f}s"

async def process_with_timeout(coro, timeout, step_name, timer):
    """Execute coroutine with timeout"""
    try:
        timer.start_step(step_name)
        result = await asyncio.wait_for(coro, timeout=timeout)
        timer.end_step(step_name)
        return result
    except asyncio.TimeoutError:
        timer.end_step(step_name)
        logger.log_it({
            "logType": "error",
            "prefix": "timeout",
            "logData": {
                "message": f"Timeout exceeded for {step_name}",
                "step": step_name,
                "timeout_seconds": timeout,
                "total_elapsed": timer.get_total_elapsed()
            }
        })
        raise TimeoutError(f"{step_name} took longer than {timeout} seconds")

async def stream_file_upload(file: UploadFile, output_path: str, chunk_size: int = 8192):
    """
    Optimized streaming file upload with chunked reading
    Reduces memory usage for large files
    """
    try:
        with open(output_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                buffer.write(chunk)
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {
                "message": "Streaming file upload completed",
                "filename": file.filename,
                "chunk_size": chunk_size,
                "output_path": output_path
            }
        })
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "web_ui",
            "logData": {
                "message": "Streaming file upload failed",
                "filename": file.filename,
                "error": str(e)
            }
        })
        raise

def run_with_hard_timeout(func, args=(), kwargs={}, timeout=300):
    """
    Run a blocking function in a separate thread with a hard timeout.
    Returns the result if completed in time, or raises TimeoutError.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

# Upload directory - use absolute path
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure logger
app_env = os.getenv('NODE_ENV', 'Development')
env_config = CONFIG.get(app_env, {})

def format_analysis_text(analysis):
    def bullet_list(items):
        return '\n'.join([f"    {i+1}. {item}" for i, item in enumerate(items)]) if items else "    None"
    def bullet_points(items):
        return '\n'.join([f"    • {item}" for item in items]) if items else "    None"
    lines = []
    # Main Issue
    mi = analysis.get('main_issue', {})
    lines.append("Main Issue:")
    lines.append(f"  • Description: {mi.get('description', '')}")
    lines.append(f"  • Category: {mi.get('category', '')}")
    lines.append(f"  • Urgency: {mi.get('urgency', '')}\n")
    # Support Given
    sg = analysis.get('support_given', {})
    lines.append("Support Given:")
    lines.append(f"  • Description: {sg.get('description', '')}")
    lines.append(f"  • Steps Taken:\n{bullet_list(sg.get('steps_taken', []))}")
    lines.append(f"  • Effectiveness: {sg.get('effectiveness', '')}\n")
    # Action Taken
    at = analysis.get('action_taken', {})
    lines.append("Action Taken:")
    lines.append(f"  • Description: {at.get('description', '')}")
    lines.append(f"  • Resolution Steps:\n{bullet_list(at.get('resolution_steps', []))}")
    lines.append(f"  • Completion Status: {at.get('completion_status', '')}\n")
    # Agent Emotion
    ae = analysis.get('agent_emotion', {})
    lines.append("Agent Emotion:")
    lines.append(f"  • Tone: {ae.get('overall_tone', '')}")
    lines.append(f"  • State: {ae.get('emotional_state', '')}")
    lines.append(f"  • Communication: {ae.get('communication_style', '')}\n")
    # Issue Resolution
    ir = analysis.get('issue_resolution', {})
    lines.append("Issue Resolution:")
    lines.append(f"  • Resolved: {'Yes' if ir.get('resolved') else 'No'}")
    lines.append(f"  • Method: {ir.get('resolution_method', '')}")
    lines.append(f"  • Customer Satisfaction: {ir.get('customer_satisfaction', '')}\n")
    # Agent Engagement
    ag = analysis.get('agent_engagement', {})
    lines.append("Agent Engagement:")
    lines.append(f"  • Level: {ag.get('level', '')}")
    lines.append(f"  • Active Listening: {'Yes' if ag.get('active_listening') else 'No'}")
    lines.append(f"  • Proactive Assistance: {'Yes' if ag.get('proactive_assistance') else 'No'}")
    lines.append(f"  • Response Time: {ag.get('response_time', '')}\n")
    # Agent Skill
    ask = analysis.get('agent_skill', {})
    lines.append("Agent Skills:")
    lines.append(f"  • Technical Knowledge: {ask.get('technical_knowledge', '')}")
    lines.append(f"  • Problem Solving: {ask.get('problem_solving', '')}")
    lines.append(f"  • Communication: {ask.get('communication', '')}")
    lines.append(f"  • Patience: {ask.get('patience', '')}\n")
    # Customer Satisfaction
    cs = analysis.get('customer_satisfaction', {})
    lines.append("Customer Satisfaction:")
    lines.append(f"  • Overall: {cs.get('overall_satisfaction', '')}")
    lines.append(f"  • Recommendation: {cs.get('willingness_to_recommend', '')}")
    lines.append(f"  • Key Factors:\n{bullet_points(cs.get('key_satisfaction_factors', []))}\n")
    # Call Tone
    ct = analysis.get('call_tone', {})
    lines.append("Call Tone:")
    lines.append(f"  • Atmosphere: {ct.get('overall_atmosphere', '')}")
    lines.append(f"  • Respect Level: {ct.get('respect_level', '')}")
    lines.append(f"  • Conflict Present: {'Yes' if ct.get('conflict_present') else 'No'}\n")
    # Overall Call Quality
    ocq = analysis.get('overall_call_quality', {})
    lines.append("Overall Call Quality:")
    lines.append(f"  • Score: {ocq.get('score', '')}")
    lines.append(f"  • Rating: {ocq.get('rating', '')}")
    lines.append(f"  • Strengths:\n{bullet_points(ocq.get('strengths', []))}")
    lines.append(f"  • Areas for Improvement:\n{bullet_points(ocq.get('areas_for_improvement', []))}\n")
    # Business Insights
    bi = analysis.get('business_insights', {})
    lines.append("Business Insights:")
    lines.append(f"  • Hotel/Restaurant Code: {bi.get('hotel_restaurant_code', '')}")
    lines.append(f"  • Booking Details: {bi.get('booking_details', '')}")
    lines.append(f"  • Billing Issues: {bi.get('billing_issues', '')}")
    lines.append(f"  • Technical Issues: {bi.get('technical_issues', '')}\n")
    # Query Analysis
    qa = analysis.get('query_analysis', {})
    lines.append("Query Analysis:")
    lines.append(f"  • Primary Query: {qa.get('primary_query', '')}")
    lines.append(f"  • Query Type: {qa.get('query_type', '')}")
    lines.append(f"  • Complexity: {qa.get('complexity', '')}")
    lines.append(f"  • Resolution Time: {qa.get('resolution_time', '')}\n")
    return '\n'.join(lines)

@router.get("/transcribe-ui", response_class=HTMLResponse)
async def transcribe_ui_form(request: Request):
    logger.log_it({
        "logType": "info",
        "prefix": "web_ui",
        "logData": {"message": "UI form loaded", "client": str(request.client.host) if hasattr(request, 'client') else None}
    })
    return templates.TemplateResponse("transcribe_ui.html", {"request": request, "result": None, "speakers_result": None})

@router.post("/cancel-processing")
async def cancel_processing(request: Request):
    """Cancel ongoing processing for a session"""
    try:
        # Get client IP as session identifier
        client_ip = str(request.client.host) if hasattr(request, 'client') else "unknown"
        
        with cancellation_lock:
            cancellation_requests[client_ip] = True
        
        logger.log_it({
            "logType": "info",
            "prefix": "cancellation",
            "logData": {
                "message": "Processing cancellation requested",
                "client_ip": client_ip,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {"status": "success", "message": "Cancellation request received"}
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "cancellation",
            "logData": {
                "message": "Error processing cancellation request",
                "error": str(e)
            }
        })
        return {"status": "error", "message": str(e)}

@router.get("/check-cancellation")
async def check_cancellation(request: Request):
    """Check if processing should be cancelled for a session"""
    try:
        client_ip = str(request.client.host) if hasattr(request, 'client') else "unknown"
        
        with cancellation_lock:
            should_cancel = cancellation_requests.get(client_ip, False)
            if should_cancel:
                # Reset the cancellation flag after checking
                cancellation_requests[client_ip] = False
        
        return {"cancelled": should_cancel}
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "cancellation",
            "logData": {
                "message": "Error checking cancellation status",
                "error": str(e)
            }
        })
        return {"cancelled": False}

@router.post("/reset-cancellation")
async def reset_cancellation(request: Request):
    """Reset cancellation flag for a client when starting new processing"""
    try:
        client_ip = str(request.client.host) if hasattr(request, 'client') else "unknown"
        
        with cancellation_lock:
            cancellation_requests[client_ip] = False
        
        logger.log_it({
            "logType": "info",
            "prefix": "cancellation",
            "logData": {
                "message": "Cancellation flag reset for new processing",
                "client_ip": client_ip,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {"status": "success", "message": "Cancellation flag reset"}
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "cancellation",
            "logData": {
                "message": "Error resetting cancellation flag",
                "error": str(e)
            }
        })
        return {"status": "error", "message": str(e)}

@router.get("/csv-management", response_class=HTMLResponse)
async def csv_management_page(request: Request):
    """CSV Management page for customer support"""
    try:
        # Get global CSV info
        csv_path = get_global_csv_path()
        csv_exists = os.path.exists(csv_path)
        
        csv_info = {}
        if csv_exists:
            file_stats = os.stat(csv_path)
            csv_info = {
                "exists": True,
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "csv_path": csv_path
            }
            
            # Count rows
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_info["total_analyses"] = sum(1 for line in f) - 1  # Subtract header
        else:
            csv_info = {
                "exists": False,
                "message": "No analyses yet. Upload and process your first audio file to create the global CSV."
            }
        
        logger.log_it({
            "logType": "info",
            "prefix": "csv_management",
            "logData": {
                "message": "CSV management page accessed",
                "csv_exists": csv_exists,
                "client": str(request.client.host) if hasattr(request, 'client') else None
            }
        })
        
        return templates.TemplateResponse("csv_management.html", {
            "request": request,
            "csv_info": csv_info
        })
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "csv_management",
            "logData": {
                "message": "Error loading CSV management page",
                "error": str(e)
            }
        })
        return templates.TemplateResponse("csv_management.html", {
            "request": request,
            "csv_info": {"exists": False, "error": str(e)}
        })

@router.post("/transcribe-ui", response_class=HTMLResponse)
async def transcribe_ui_submit(
    request: Request, 
    file: UploadFile = File(...),
    use_hindi_enhancements: bool = Form(False),
    run_debug_analysis: bool = Form(False),
    use_advanced_diarization: bool = Form(False),
    save_preprocessed_audio: bool = Form(False),
    use_hinglish: bool = Form(False),
    use_xglish: bool = Form(False),
    language_override: str = Form(""),
    clear_cache: bool = Form(False)
):
    # Initialize timer
    timer = ProcessingTimer()
    total_start_time = time.time()
    
    file_id = str(uuid.uuid4())
    original_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    preprocessed_file_path = None
    preprocessing_metrics = None
    hindi_debug_results = None
    preprocessed_analysis = None
    
    # Calculate dynamic timeouts based on file size (will be updated after file upload)
    file_size_mb = 0
    TIMEOUTS = BASE_TIMEOUTS.copy()
    
    # Get client IP for cancellation tracking
    client_ip = str(request.client.host) if hasattr(request, 'client') else "unknown"
    
    # Reset any previous cancellation flag for this client
    with cancellation_lock:
        cancellation_requests[client_ip] = False
    
    try:
        # Step 1: Optimized streaming file upload
        timer.start_step("file_upload")
        await stream_file_upload(file, original_file_path)
        timer.end_step("file_upload")
        
        # Calculate file size and update timeouts dynamically
        file_size_bytes = os.path.getsize(original_file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        TIMEOUTS = get_dynamic_timeouts(file_size_mb)
        
        # Generate file hash for caching
        file_hash = get_file_hash(original_file_path)
        
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {
                "message": "File uploaded with dynamic timeouts and caching",
                "filename": file.filename,
                "size_bytes": file_size_bytes,
                "size_mb": round(file_size_mb, 2),
                "file_hash": file_hash[:8],
                "timeout_multiplier": TIMEOUTS["transcription"] / BASE_TIMEOUTS["transcription"],
                "dynamic_timeouts": TIMEOUTS,
                "client": str(request.client.host) if hasattr(request, 'client') else None,
                "hindi_enhancements": use_hindi_enhancements,
                "debug_analysis": run_debug_analysis,
                "advanced_diarization": use_advanced_diarization,
                "save_preprocessed_audio": save_preprocessed_audio,
                "upload_time": timer.get_step_duration("file_upload"),
                "optimization": "streaming_upload_dynamic_timeouts_caching"
            }
        })
        
        # Check for cancellation before preprocessing (but don't show UI message yet)
        if check_cancellation_request(client_ip):
            logger.log_it({
                "logType": "info",
                "prefix": "web_ui",
                "logData": {
                    "message": "Processing cancelled by user before preprocessing",
                    "filename": file.filename,
                    "client_ip": client_ip
                }
            })
            # Return a simple cancellation response without showing the message in UI
            return templates.TemplateResponse(
                "transcribe_ui.html",
                {
                    "request": request, 
                    "result": None, 
                    "filename": file.filename, 
                    "speakers_result": None,
                    "cancelled": True,
                    "show_cancellation_message": False
                }
            )
        
        # Step 2: Preprocess audio with timeout
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {"message": "Preprocessing started", "filename": file.filename}
        })
        
        timer.start_step("preprocessing")
        preprocessor = UnifiedAudioProcessor()
        preprocessed_file_path = os.path.join(UPLOAD_DIR, f"preprocessed_{file_id}_{file.filename}")
        
        # Analyze audio quality first to get metrics
        timer.start_substep("audio_quality_analysis")
        quality_metrics = preprocessor.analyze_audio_quality(original_file_path)
        timer.end_substep("audio_quality_analysis")
        
        # Preprocessing with timeout
        timer.start_substep("audio_preprocessing")
        try:
            preprocessing_success = await asyncio.to_thread(
                run_with_hard_timeout,
                preprocessor.preprocess_audio,
                (original_file_path, preprocessed_file_path),
                {},
                TIMEOUTS["preprocessing"]
            )
        except TimeoutError as e:
            # handle timeout (show error to user, log, etc.)
            logger.log_it({
                "logType": "error",
                "prefix": "web_ui",
                "logData": {
                    "message": "Preprocessing timeout",
                    "filename": file.filename,
                    "timeout_seconds": TIMEOUTS["preprocessing"],
                    "total_elapsed": timer.get_total_elapsed()
                }
            })
            return templates.TemplateResponse(
                "transcribe_ui.html",
                {
                    "request": request, 
                    "result": f"Error: Preprocessing timeout after {TIMEOUTS['preprocessing']} seconds. File may be too large or corrupted.", 
                    "filename": file.filename, 
                    "speakers_result": None, 
                    "preprocessing_metrics": quality_metrics,
                    "hindi_debug_results": None,
                    "use_hindi_enhancements": use_hindi_enhancements,
                    "run_debug_analysis": run_debug_analysis,
                    "use_advanced_diarization": use_advanced_diarization,
                    "save_preprocessed_audio": save_preprocessed_audio,
                    "timeout_error": True
                }
            )
        
        timer.end_substep("audio_preprocessing")
        
        if not preprocessing_success:
            timer.end_step("preprocessing")
            logger.log_it({
                "logType": "error",
                "prefix": "web_ui",
                "logData": {"message": "Preprocessing failed", "filename": file.filename}
            })
            return templates.TemplateResponse(
                "transcribe_ui.html",
                {
                    "request": request, 
                    "result": "Error: Preprocessing failed", 
                    "filename": file.filename, 
                    "speakers_result": None, 
                    "preprocessing_metrics": quality_metrics,
                    "hindi_debug_results": None,
                    "use_hindi_enhancements": use_hindi_enhancements,
                    "run_debug_analysis": run_debug_analysis,
                    "use_advanced_diarization": use_advanced_diarization,
                    "save_preprocessed_audio": save_preprocessed_audio
                }
            )
        
        # Capture preprocessing metrics for display
        preprocessing_metrics = {
            "quality_analysis": quality_metrics,
            "denoising_applied": quality_metrics.get("needs_denoising", True),
            "snr_db": quality_metrics.get("snr_db", 0),
            "denoising_reasons": quality_metrics.get("denoising_reason", []),
            "processing_time": timer.get_step_duration("preprocessing")
        }
        
        # Save preprocessed audio file if requested
        saved_preprocessed_path = None
        if save_preprocessed_audio and preprocessing_success:
            try:
                # Create a permanent directory for saved preprocessed files
                current_dir = os.path.dirname(os.path.abspath(__file__))
                saved_dir = os.path.join(current_dir, ".cache", "saved_preprocessed_audio")
                os.makedirs(saved_dir, exist_ok=True)
                
                # Create a descriptive filename
                timestamp = int(time.time())
                original_name = os.path.splitext(file.filename)[0]
                saved_filename = f"preprocessed_{original_name}_{timestamp}.wav"
                saved_preprocessed_path = os.path.join(saved_dir, saved_filename)
                
                # Copy the preprocessed file to the saved location
                shutil.copy2(preprocessed_file_path, saved_preprocessed_path)
                
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Preprocessed audio saved",
                        "original_file": file.filename,
                        "saved_path": saved_preprocessed_path,
                        "file_size": os.path.getsize(saved_preprocessed_path)
                    }
                })
                
                # Add saved path to preprocessing metrics
                preprocessing_metrics["saved_preprocessed_path"] = saved_preprocessed_path
                
                # --- Analyze the saved preprocessed audio file ---
                analyzer = PreprocessedAudioAnalyzer()
                preprocessed_analysis = analyzer.analyze_audio_file(saved_preprocessed_path)
                
            except Exception as e:
                logger.log_it({
                    "logType": "error",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Failed to save preprocessed audio",
                        "error": str(e)
                    }
                })
        
        timer.end_step("preprocessing")
        
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {
                "message": "Preprocessing completed", 
                "filename": file.filename,
                "denoising_applied": preprocessing_metrics["denoising_applied"],
                "snr_db": preprocessing_metrics["snr_db"],
                "preprocessed_saved": saved_preprocessed_path is not None,
                "processing_time": preprocessing_metrics["processing_time"]
            }
        })
        
        # Step 3: Hindi Debug Analysis (if requested)
        if run_debug_analysis:
            logger.log_it({
                "logType": "info",
                "prefix": "web_ui",
                "logData": {"message": "Hindi debug analysis started", "filename": file.filename}
            })
            
            try:
                debug_helper = HindiDebugHelper()
                hindi_debug_results = await process_with_timeout(
                    debug_helper.debug_hindi_transcription(preprocessed_file_path),
                    TIMEOUTS["debug_analysis"],
                    "debug_analysis",
                    timer
                )
                
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Hindi debug analysis completed",
                        "filename": file.filename,
                        "debug_status": "success" if "error" not in hindi_debug_results else "failed",
                        "processing_time": timer.get_step_duration("debug_analysis")
                    }
                })
                
            except TimeoutError as e:
                logger.log_it({
                    "logType": "error",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Debug analysis timeout",
                        "filename": file.filename,
                        "timeout_seconds": TIMEOUTS["debug_analysis"]
                    }
                })
                hindi_debug_results = {
                    "error": f"Debug analysis timeout after {TIMEOUTS['debug_analysis']} seconds",
                    "status": "timeout"
                }
        
        # Step 4: Parallel Transcription and Analysis (Major Performance Boost)
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {
                "message": "Starting parallel transcription and analysis", 
                "filename": file.filename,
                "method": "enhanced_hindi" if use_hindi_enhancements else "standard",
                "optimization": "parallel_processing"
            }
        })
        
        timer.start_step("transcription")
        try:
            # Check cache first for transcription results
            cache_key = f"transcription_{use_hindi_enhancements}_{use_advanced_diarization}_{language_override or 'auto'}"
            
            logger.log_it({
                "logType": "info",
                "prefix": "web_ui",
                "logData": {
                    "message": "Starting transcription with language settings",
                    "filename": file.filename,
                    "language_override": language_override or "auto-detect",
                    "use_hindi_enhancements": use_hindi_enhancements,
                    "use_advanced_diarization": use_advanced_diarization,
                    "cache_key": cache_key
                }
            })
            
            cached_result = None
            if not clear_cache:
                cached_result = load_cached_result(file_hash, cache_key)
            else:
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Cache cleared - forcing fresh transcription",
                        "filename": file.filename,
                        "file_hash": file_hash,
                        "cache_key": cache_key
                    }
                })
            
            if cached_result:
                result = cached_result
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Using cached transcription result",
                        "filename": file.filename,
                        "file_hash": file_hash[:8],
                        "cache_key": cache_key,
                        "optimization": "cache_hit"
                    }
                })
            else:
                # Use optimized transcription service for better performance
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Using optimized transcription service with parallel processing",
                        "filename": file.filename,
                        "method": "optimized_parallel",
                        "file_hash": file_hash[:8]
                    }
                })
                
                # Use optimized service if available, otherwise fallback to original
                if OPTIMIZED_SERVICE_AVAILABLE and optimized_service:
                    timer.start_substep("optimized_transcription")
                    # Use optimized service with parallel processing
                    result = await process_with_timeout(
                        optimized_service.process_audio_optimized(preprocessed_file_path, use_advanced_diarization, language_override),
                        TIMEOUTS["transcription"],
                        "transcription",
                        timer
                    )
                    timer.end_substep("optimized_transcription")
                else:
                    timer.start_substep("parallel_transcription")
                    # Fallback to original service with parallel processing
                    transcription_service = TranscriptionWithSpeakersService(
                        use_advanced_diarization=use_advanced_diarization,
                        use_hinglish=use_hinglish,
                        use_xglish=use_xglish
                    )
                    
                    # Run transcription and analysis in parallel for better performance
                    if use_hindi_enhancements:
                        timer.start_substep("hindi_transcription")
                        # Use enhanced Hindi transcription with translation
                        transcription_task = process_with_timeout(
                            transcription_service.transcribe_hindi_with_translation(preprocessed_file_path),
                            TIMEOUTS["transcription"],
                            "transcription",
                            timer
                        )
                        timer.end_substep("hindi_transcription")
                    else:
                        timer.start_substep("standard_transcription_with_speakers")
                        # Use standard transcription with speakers
                        transcription_task = process_with_timeout(
                            transcription_service.process_audio_file_with_speakers(preprocessed_file_path, language_override),
                            TIMEOUTS["transcription"],
                            "transcription",
                            timer
                        )
                        timer.end_substep("standard_transcription_with_speakers")
                    
                    # Execute transcription
                    result = await transcription_task
                    timer.end_substep("parallel_transcription")
                
                # Cache the result for future use
                if result.get("status") == "success":
                    save_cached_result(file_hash, cache_key, result)
                
        except TimeoutError as e:
            logger.log_it({
                "logType": "error",
                "prefix": "web_ui",
                "logData": {
                    "message": "Transcription timeout",
                    "filename": file.filename,
                    "timeout_seconds": TIMEOUTS["transcription"],
                    "total_elapsed": timer.get_total_elapsed()
                }
            })
            return templates.TemplateResponse(
                "transcribe_ui.html",
                {
                    "request": request, 
                    "result": f"Error: Transcription timeout after {TIMEOUTS['transcription']} seconds. File may be too long or complex.", 
                    "filename": file.filename, 
                    "speakers_result": None, 
                    "preprocessing_metrics": preprocessing_metrics,
                    "hindi_debug_results": hindi_debug_results,
                    "use_hindi_enhancements": use_hindi_enhancements,
                    "run_debug_analysis": run_debug_analysis,
                    "use_advanced_diarization": use_advanced_diarization,
                    "save_preprocessed_audio": save_preprocessed_audio,
                    "preprocessed_analysis": preprocessed_analysis,
                    "use_hinglish": use_hinglish,
                    "timeout_error": True
                }
            )
        
        timer.end_step("transcription")
        
        if result["status"] == "error":
            logger.log_it({
                "logType": "error",
                "prefix": "web_ui",
                "logData": {
                    "message": "Transcription failed", 
                    "filename": file.filename, 
                    "error": result.get("error", "Unknown error"),
                    "method": "enhanced_hindi" if use_hindi_enhancements else "standard",
                    "processing_time": timer.get_step_duration("transcription")
                }
            })
            return templates.TemplateResponse(
                "transcribe_ui.html",
                {
                    "request": request, 
                    "result": f"Error: {result.get('error', 'Transcription failed')}", 
                    "filename": file.filename, 
                    "speakers_result": None, 
                    "preprocessing_metrics": preprocessing_metrics,
                    "hindi_debug_results": hindi_debug_results,
                    "use_hindi_enhancements": use_hindi_enhancements,
                    "run_debug_analysis": run_debug_analysis,
                    "use_advanced_diarization": use_advanced_diarization,
                    "save_preprocessed_audio": save_preprocessed_audio,
                    "preprocessed_analysis": preprocessed_analysis,
                    "use_hinglish": use_hinglish
                }
            )
        
        # Step 5: Process results for display
        english_text = result.get("translated_text", result.get("transcribed_text", ""))
        speakers_data = {
            "segments": result.get("segments", []),
            "speaker_stats": result.get("speaker_stats", {}),
            "language": result.get("language_name", "Unknown"),
            "duration": result.get("duration", 0),
            "transcription_method": result.get("transcription_method", "standard"),
            "auto_hinglish_enabled": result.get("auto_hinglish_enabled", False)
        }
        
        # Get token usage for display
        if OPTIMIZED_SERVICE_AVAILABLE and optimized_service:
            token_usage = optimized_service.get_token_usage()
            performance_metrics = optimized_service.get_performance_summary()
        else:
            # Fallback token usage
            token_usage = {
                "transcription_tokens": 0,
                "translation_tokens": 0,
                "analysis_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0
            }
            performance_metrics = {
                "total_processing_time": 0,
                "transcription_time": 0,
                "translation_time": 0,
                "analysis_time": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0,
                "parallel_savings": 0,
                "speedup_factor": 1
            }
        
        # Initialize analysis variables
        analysis_tokens = 0
        analysis_cost = 0.0
        
        # Calculate detailed cost breakdown
        transcription_cost = (token_usage.get("transcription_tokens", 0) / 1000) * 0.006  # Whisper cost
        translation_cost = (token_usage.get("translation_tokens", 0) / 1000) * 0.002  # GPT-3.5-turbo cost
        analysis_cost = (token_usage.get("analysis_tokens", 0) / 1000) * 0.00015  # GPT-4o-mini cost
        
        # Update token usage with detailed breakdown
        token_usage.update({
            "transcription_cost": transcription_cost,
            "translation_cost": translation_cost,
            "analysis_cost": analysis_cost,
            "analysis_tokens": analysis_tokens,
            "performance_metrics": performance_metrics
        })
        
        # Step 6: Parallel Call Analysis (Major Performance Boost)
        call_analysis_result = None
        if result["status"] == "success" and result.get("transcribed_text"):
            logger.log_it({
                "logType": "info",
                "prefix": "web_ui",
                "logData": {
                    "message": "Starting parallel call analysis",
                    "filename": file.filename,
                    "optimization": "parallel_analysis"
                }
            })
            
            timer.start_step("call_analysis")
            try:
                # Check cache first for analysis results
                analysis_cache_key = f"analysis_{use_hindi_enhancements}"
                cached_analysis = load_cached_result(file_hash, analysis_cache_key)
                
                if cached_analysis:
                    call_analysis_result = cached_analysis
                    logger.log_it({
                        "logType": "info",
                        "prefix": "web_ui",
                        "logData": {
                            "message": "Using cached analysis result",
                            "filename": file.filename,
                            "file_hash": file_hash[:8],
                            "cache_key": analysis_cache_key,
                            "optimization": "analysis_cache_hit"
                        }
                    })
                    # For cached results, use default token usage
                    analysis_tokens = 0
                    analysis_cost = 0.0
                else:
                    timer.start_substep("analysis_service_init")
                    analysis_service = CallAnalysisService()
                    timer.end_substep("analysis_service_init")
                    
                    timer.start_substep("parallel_analysis_execution")
                    # Run analysis in parallel with other operations
                    call_analysis_result = await process_with_timeout(
                        analysis_service.analyze_call_transcription(
                            result.get("transcribed_text", ""),
                            result.get("segments", [])
                        ),
                        TIMEOUTS["analysis"],
                        "call_analysis",
                        timer
                    )
                    timer.end_substep("parallel_analysis_execution")
                    
                    # Get analysis token usage only for non-cached results
                    analysis_token_usage = analysis_service.get_token_usage()
                    analysis_tokens = analysis_token_usage.get("analysis_tokens", 0)
                    analysis_cost = analysis_token_usage.get("estimated_cost_usd", 0.0)
                    
                    # Cache the analysis result
                    if call_analysis_result.get("status") != "failed":
                        save_cached_result(file_hash, analysis_cache_key, call_analysis_result)
                
                # Add analysis tokens to total
                token_usage["analysis_tokens"] = analysis_tokens
                token_usage["total_tokens"] += analysis_tokens
                token_usage["estimated_cost_usd"] += analysis_cost
                
                logger.log_it({
                    "logType": "info",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Parallel call analysis completed",
                        "filename": file.filename,
                        "analysis_status": "success" if "error" not in call_analysis_result else "failed",
                        "analysis_tokens": analysis_tokens,
                        "analysis_cost": analysis_cost,
                        "processing_time": timer.get_step_duration("call_analysis"),
                        "optimization": "parallel_processing"
                    }
                })
                
            except TimeoutError as e:
                logger.log_it({
                    "logType": "error",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Call analysis timeout",
                        "filename": file.filename,
                        "timeout_seconds": TIMEOUTS["analysis"]
                    }
                })
                call_analysis_result = {
                    "error": f"Call analysis timeout after {TIMEOUTS['analysis']} seconds",
                    "status": "timeout"
                }
            except Exception as e:
                logger.log_it({
                    "logType": "error",
                    "prefix": "web_ui",
                    "logData": {
                        "message": "Call analysis failed",
                        "filename": file.filename,
                        "error": str(e)
                    }
                })
                call_analysis_result = {
                    "error": str(e),
                    "status": "failed"
                }
            finally:
                timer.end_step("call_analysis")
        
        # Calculate detailed cost breakdown
        transcription_cost = (token_usage.get("transcription_tokens", 0) / 1000) * 0.006  # Whisper cost
        translation_cost = (token_usage.get("translation_tokens", 0) / 1000) * 0.002  # GPT-3.5-turbo cost
        analysis_cost = (token_usage.get("analysis_tokens", 0) / 1000) * 0.00015  # GPT-4o-mini cost
        
        # Update token usage with detailed breakdown
        token_usage.update({
            "transcription_cost": transcription_cost,
            "translation_cost": translation_cost,
            "analysis_cost": analysis_cost,
            "analysis_tokens": analysis_tokens
        })
        
        # Calculate total processing time and get performance summary
        total_processing_time = time.time() - total_start_time
        
        # Set file metrics for performance analysis
        file_size = os.path.getsize(original_file_path) if os.path.exists(original_file_path) else 0
        audio_duration = speakers_data.get("duration", 0)
        timer.set_file_metrics(file_size, audio_duration)
        
        # Get detailed performance summary
        performance_summary = timer.get_performance_summary()
        
        logger.log_it({
            "logType": "info",
            "prefix": "web_ui",
            "logData": {
                "message": "Transcription completed", 
                "filename": file.filename, 
                "status": result.get("status", "unknown"),
                "language": speakers_data["language"],
                "segments_count": len(speakers_data["segments"]),
                "method": speakers_data["transcription_method"],
                "token_usage": token_usage,
                "analysis_performed": call_analysis_result is not None,
                "total_processing_time": total_processing_time,
                "performance_summary": performance_summary,
                "step_times": timer.step_times
            }
        })
        
        # Initialize global CSV and save results automatically
        initialize_global_csv()
        
        # Determine cache status for logging
        cache_status = "cached" if cached_result else "new"
        
        # Save to global CSV automatically
        if call_analysis_result and call_analysis_result.get("status") != "failed":
            analysis_id = append_to_global_csv(
                file_name=file.filename,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                call_analysis_result=call_analysis_result,
                result=result,
                speakers_data=speakers_data,
                token_usage=token_usage,
                performance_summary=performance_summary,
                cache_status=cache_status
            )
            
            logger.log_it({
                "logType": "info",
                "prefix": "web_ui",
                "logData": {
                    "message": "Analysis automatically saved to global CSV",
                    "analysis_id": analysis_id,
                    "file_name": file.filename,
                    "file_hash": file_hash[:8],
                    "cache_status": cache_status,
                    "global_csv_path": get_global_csv_path()
                }
            })

        return templates.TemplateResponse(
            "transcribe_ui.html",
            {
                "request": request, 
                "result": english_text, 
                "filename": file.filename, 
                "speakers_result": speakers_data, 
                "preprocessing_metrics": preprocessing_metrics,
                "hindi_debug_results": hindi_debug_results,
                "use_hindi_enhancements": use_hindi_enhancements,
                "run_debug_analysis": run_debug_analysis,
                "use_advanced_diarization": use_advanced_diarization,
                "save_preprocessed_audio": save_preprocessed_audio,
                "token_usage": token_usage,
                "preprocessed_analysis": preprocessed_analysis,
                "use_hinglish": use_hinglish,
                "use_xglish": use_xglish,
                "language_override": language_override,
                "clear_cache": clear_cache,
                "call_analysis": call_analysis_result,
                "processing_time": total_processing_time,
                "step_times": timer.step_times,
                "performance_summary": performance_summary
            }
        )
        
    except Exception as e:
        total_processing_time = time.time() - total_start_time
        logger.log_it({
            "logType": "error",
            "prefix": "web_ui",
            "logData": {
                "message": "Error in processing", 
                "filename": file.filename, 
                "error": str(e),
                "total_processing_time": total_processing_time
            }
        })
        return templates.TemplateResponse(
            "transcribe_ui.html",
            {
                "request": request, 
                "result": f"Error: {str(e)}", 
                "filename": file.filename, 
                "speakers_result": None, 
                "preprocessing_metrics": preprocessing_metrics,
                "hindi_debug_results": hindi_debug_results,
                "use_hindi_enhancements": use_hindi_enhancements,
                "run_debug_analysis": run_debug_analysis,
                "use_advanced_diarization": use_advanced_diarization,
                "save_preprocessed_audio": save_preprocessed_audio,
                "preprocessed_analysis": preprocessed_analysis,
                "use_hinglish": use_hinglish,
                "processing_time": total_processing_time
            }
        )
    finally:
        # Clean up uploaded files
        if os.path.exists(original_file_path):
            os.remove(original_file_path)
        if preprocessed_file_path and os.path.exists(preprocessed_file_path):
            os.remove(preprocessed_file_path) 

@router.get("/download-analysis/{filename}")
async def download_analysis_csv(filename: str):
    """Download individual analysis CSV file"""
    try:
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.csv'):
            safe_filename += '.csv'
        
        # Construct file path
        results_dir = os.path.join("Call_recordings_AI", ".cache", "results")
        csv_path = os.path.join(results_dir, safe_filename)
        
        if not os.path.exists(csv_path):
            return {"error": "File not found", "filename": safe_filename}
        
        # Return file for download
        from fastapi.responses import FileResponse
        return FileResponse(
            path=csv_path,
            filename=safe_filename,
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "web_ui",
            "logData": {
                "message": "Error downloading CSV file",
                "filename": filename,
                "error": str(e)
            }
        })
        return {"error": str(e)}

@router.get("/download-global-results")
async def download_global_results_csv():
    """Download the global results CSV file with all analyses"""
    try:
        csv_path = get_global_csv_path()
        
        # Debug: Log the CSV path and file status
        logger.log_it({
            "logType": "info",
            "prefix": "download_debug",
            "logData": {
                "message": "Download request received",
                "csv_path": csv_path,
                "file_exists": os.path.exists(csv_path),
                "timestamp": datetime.now().isoformat()
            }
        })
        
        if not os.path.exists(csv_path):
            logger.log_it({
                "logType": "warning",
                "prefix": "download_debug",
                "logData": {
                    "message": "CSV file does not exist, initializing",
                    "csv_path": csv_path
                }
            })
            # Initialize the global CSV if it doesn't exist
            initialize_global_csv()
        
        # Get file stats for debugging
        file_stats = os.stat(csv_path)
        file_size = file_stats.st_size
        last_modified = datetime.fromtimestamp(file_stats.st_mtime)
        
        # Count rows for debugging
        with open(csv_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header row
        
        # Debug: Log file details
        logger.log_it({
            "logType": "info",
            "prefix": "download_debug",
            "logData": {
                "message": "Serving CSV file for download",
                "csv_path": csv_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "row_count": row_count,
                "last_modified": last_modified.isoformat(),
                "filename": "global_call_analysis_results.csv"
            }
        })
        
        # Return file for download
        from fastapi.responses import FileResponse
        return FileResponse(
            path=csv_path,
            filename="global_call_analysis_results.csv",
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "download_debug",
            "logData": {
                "message": "Error downloading global results CSV",
                "error": str(e),
                "csv_path": csv_path if 'csv_path' in locals() else "unknown"
            }
        })
        return {"error": str(e)}

@router.get("/debug-csv-content")
async def debug_csv_content():
    """Debug endpoint to check CSV content directly"""
    try:
        csv_path = get_global_csv_path()
        
        if not os.path.exists(csv_path):
            return {"error": "CSV file does not exist", "csv_path": csv_path}
        
        # Get file stats
        file_stats = os.stat(csv_path)
        file_size = file_stats.st_size
        last_modified = datetime.fromtimestamp(file_stats.st_mtime)
        
        # Read last few lines
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get last 3 data rows (excluding header)
        last_rows = []
        if len(lines) > 1:
            for i in range(max(1, len(lines) - 3), len(lines)):
                if i > 0:  # Skip header
                    parts = lines[i].strip().split(',')
                    if len(parts) >= 3:
                        last_rows.append({
                            "line_number": i + 1,
                            "analysis_id": parts[0].strip('"'),
                            "file_name": parts[1].strip('"'),
                            "upload_date": parts[2].strip('"'),
                            "file_size": parts[3].strip('"') if len(parts) > 3 else "N/A"
                        })
        
        return {
            "csv_path": csv_path,
            "absolute_path": os.path.abspath(csv_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_lines": len(lines),
            "total_data_rows": len(lines) - 1,
            "last_modified": last_modified.isoformat(),
            "last_rows": last_rows
        }
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "debug_csv",
            "logData": {
                "message": "Error in debug CSV content",
                "error": str(e)
            }
        })
        return {"error": str(e)}

@router.get("/global-results-info", response_class=HTMLResponse)
async def get_global_results_info(request: Request):
    """Get information about the global results CSV in a nice HTML format"""
    try:
        csv_path = get_global_csv_path()
        
        if not os.path.exists(csv_path):
            return templates.TemplateResponse("global_stats.html", {
                "request": request,
                "csv_exists": False,
                "message": "Global CSV not yet created. Upload and analyze a file to create it.",
                "csv_path": csv_path
            })
        
        # Get file stats
        file_stats = os.stat(csv_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Count rows (approximate)
        with open(csv_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header row
        
        # Calculate additional statistics
        last_modified = datetime.fromtimestamp(file_stats.st_mtime)
        days_since_created = (datetime.now() - last_modified).days
        
        # Analyze CSV data for business insights
        csv_insights = analyze_csv_insights(csv_path)
        
        # Get reset history
        reset_history = get_reset_history()
        
        # Basic file stats
        csv_stats = {
            "total_analyses": row_count,
            "file_size_mb": round(file_size_mb, 2),
            "last_modified": last_modified.strftime("%Y-%m-%d %H:%M:%S"),
            "days_active": days_since_created,
            "avg_size_per_analysis": round(file_size_mb / max(row_count, 1), 3),
            "csv_path": csv_path,
            "download_url": "/innex/download-global-results",
            "insights": csv_insights
        }
        
        logger.log_it({
            "logType": "info",
            "prefix": "global_stats",
            "logData": {
                "message": "Global statistics page accessed",
                "total_analyses": row_count,
                "file_size_mb": csv_stats["file_size_mb"],
                "client": str(request.client.host) if hasattr(request, 'client') else None
            }
        })
        
        return templates.TemplateResponse("global_stats.html", {
            "request": request,
            "csv_exists": True,
            "csv_stats": csv_stats,
            "csv_insights": csv_insights,
            "reset_history": reset_history
        })
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "web_ui",
            "logData": {
                "message": "Error getting global results info",
                "error": str(e)
            }
        })
        return templates.TemplateResponse("global_stats.html", {
            "request": request,
            "csv_exists": False,
            "error": str(e)
        })

@router.post("/reset-global-csv")
async def reset_global_csv(request: Request):
    """Reset the global CSV file to start fresh"""
    try:
        csv_path = get_global_csv_path()
        
        # Create backup before reset
        if os.path.exists(csv_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(current_dir, ".cache", "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Get current stats before backup
            file_stats = os.stat(csv_path)
            file_size_mb = file_stats.st_size / (1024 * 1024)
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header row
            
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"global_csv_backup_before_reset_{timestamp}.csv"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy current CSV to backup
            shutil.copy2(csv_path, backup_path)
            
            logger.log_it({
                "logType": "info",
                "prefix": "csv_reset",
                "logData": {
                    "message": "Global CSV reset initiated",
                    "backup_created": backup_path,
                    "original_rows": row_count,
                    "original_size_mb": round(file_size_mb, 2),
                    "client_ip": str(request.client.host) if hasattr(request, 'client') else "unknown"
                }
            })
        
        # Remove existing CSV
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        # Initialize fresh CSV
        initialize_global_csv()
        
        logger.log_it({
            "logType": "info",
            "prefix": "csv_reset",
            "logData": {
                "message": "Global CSV reset completed successfully",
                "new_csv_path": csv_path,
                "backup_location": backup_path if 'backup_path' in locals() else "No backup needed",
                "client_ip": str(request.client.host) if hasattr(request, 'client') else "unknown"
            }
        })
        
        return {
            "status": "success",
            "message": "Global CSV reset successfully",
            "backup_created": backup_path if 'backup_path' in locals() else None,
            "new_csv_path": csv_path,
            "reset_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "csv_reset",
            "logData": {
                "message": "Error resetting global CSV",
                "error": str(e),
                "client_ip": str(request.client.host) if hasattr(request, 'client') else "unknown"
            }
        })
        return {
            "status": "error",
            "message": f"Failed to reset CSV: {str(e)}"
        }

@router.get("/global-results-info-json")
async def get_global_results_info_json():
    """Get information about the global results CSV in JSON format (for API calls)"""
    try:
        csv_path = get_global_csv_path()
        
        if not os.path.exists(csv_path):
            return {
                "exists": False,
                "message": "Global CSV not yet created. Upload and analyze a file to create it.",
                "csv_path": csv_path
            }
        
        # Get file stats
        file_stats = os.stat(csv_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Count rows (approximate)
        with open(csv_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header row
        
        return {
            "exists": True,
            "csv_path": csv_path,
            "file_size_mb": round(file_size_mb, 2),
            "total_analyses": row_count,
            "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "download_url": "/innex/download-global-results"
        }
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "web_ui",
            "logData": {
                "message": "Error getting global results info",
                "error": str(e)
            }
        })
        return {"error": str(e)} 