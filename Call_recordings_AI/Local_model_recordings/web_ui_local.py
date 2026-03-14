#!/usr/bin/env python3
"""
Web UI with Local Models
========================

Modified version of web_ui.py that uses local models through the adapter.
This maintains the same UI and functionality but uses local models instead of OpenAI.
"""

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

# Import our local model adapter instead of direct services
from local_model_adapter import get_adapter
from unified_audio_processor_local import UnifiedAudioProcessorLocal

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import real pylogger, fallback to mock
try:
    from pylogger import pylogger
    logger = pylogger("/home/saas/logs/innex", "call_recordings_ai_local")
except ImportError:
    from .mock_pylogger import pylogger
    logger = pylogger

from config import CONFIG

# Timeout configurations
TIMEOUTS = {
    "preprocessing": 300,  # 5 minutes
    "transcription": 600,  # 10 minutes
    "analysis": 180,       # 3 minutes
    "total": 1200         # 20 minutes total
}

class ProcessingTimer:
    """Track processing time for each step"""
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None
    
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
                "total_elapsed": self.get_total_elapsed()
            }
        })
    
    def end_step(self, step_name):
        """End timing a step"""
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
            duration = self.step_times[step_name]["end"] - self.step_times[step_name]["start"]
            logger.log_it({
                "logType": "info",
                "prefix": "processing_timer",
                "logData": {
                    "message": f"Completed step: {step_name}",
                    "step": step_name,
                    "duration_seconds": duration,
                    "total_elapsed": self.get_total_elapsed()
                }
            })
    
    def get_total_elapsed(self):
        """Get total elapsed time"""
        return time.time() - self.start_time
    
    def get_step_duration(self, step_name):
        """Get duration of a specific step"""
        if step_name in self.step_times and "end" in self.step_times[step_name]:
            return self.step_times[step_name]["end"] - self.step_times[step_name]["start"]
        return None

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
        raise

def run_with_hard_timeout(func, args=(), kwargs={}, timeout=300):
    """Run function with hard timeout using ThreadPoolExecutor"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.log_it({
                "logType": "error",
                "prefix": "hard_timeout",
                "logData": {
                    "message": f"Hard timeout exceeded for {func.__name__}",
                    "timeout_seconds": timeout
                }
            })
            raise

def format_analysis_text(analysis):
    """Format analysis text for display"""
    def bullet_list(items):
        if not items:
            return "None"
        return "\n".join([f"• {item}" for item in items])
    
    def bullet_points(items):
        if not items:
            return "None"
        return "\n".join([f"• {item}" for item in items])
    
    if isinstance(analysis, str):
        return analysis
    
    if not isinstance(analysis, dict):
        return str(analysis)
    
    formatted_text = ""
    
    # Main issue
    if "main_issue" in analysis:
        formatted_text += f"<strong>Main Issue:</strong> {analysis['main_issue']}\n\n"
    
    # Customer needs
    if "customer_needs" in analysis:
        formatted_text += f"<strong>Customer Needs:</strong>\n{bullet_list(analysis['customer_needs'])}\n\n"
    
    # Agent actions
    if "agent_actions" in analysis:
        formatted_text += f"<strong>Agent Actions:</strong>\n{bullet_list(analysis['agent_actions'])}\n\n"
    
    # Resolution status
    if "resolution_status" in analysis:
        formatted_text += f"<strong>Resolution Status:</strong> {analysis['resolution_status']}\n\n"
    
    # Customer satisfaction
    if "customer_satisfaction" in analysis:
        formatted_text += f"<strong>Customer Satisfaction:</strong> {analysis['customer_satisfaction']}\n\n"
    
    # Agent performance
    if "agent_performance" in analysis and isinstance(analysis['agent_performance'], dict):
        formatted_text += "<strong>Agent Performance:</strong>\n"
        for metric, rating in analysis['agent_performance'].items():
            formatted_text += f"• {metric.replace('_', ' ').title()}: {rating}\n"
        formatted_text += "\n"
    
    # Recommendations
    if "recommendations" in analysis:
        formatted_text += f"<strong>Recommendations:</strong>\n{bullet_list(analysis['recommendations'])}\n\n"
    
    # Sentiment analysis
    if "sentiment" in analysis:
        formatted_text += f"<strong>Overall Sentiment:</strong> {analysis['sentiment']}\n\n"
    
    # Key topics
    if "key_topics" in analysis:
        formatted_text += f"<strong>Key Topics:</strong>\n{bullet_list(analysis['key_topics'])}\n\n"
    
    # Action items
    if "action_items" in analysis:
        formatted_text += f"<strong>Action Items:</strong>\n{bullet_list(analysis['action_items'])}\n\n"
    
    return formatted_text.strip()

# Create router
router = APIRouter()

# Templates
templates = Jinja2Templates(directory="../templates")

@router.get("/transcribe-ui-local", response_class=HTMLResponse)
async def transcribe_ui_form(request: Request):
    """Transcription UI form with local models"""
    return templates.TemplateResponse("transcribe.html", {"request": request, "use_local_models": True})

@router.post("/transcribe-ui-local", response_class=HTMLResponse)
async def transcribe_ui_submit(
    request: Request, 
    file: UploadFile = File(...),
    use_hindi_enhancements: bool = Form(False),
    run_debug_analysis: bool = Form(False),
    use_advanced_diarization: bool = Form(False),
    save_preprocessed_audio: bool = Form(False),
    use_hinglish: bool = Form(False)
):
    """Process audio file with local models"""
    # Initialize timer
    timer = ProcessingTimer()
    
    try:
        # Get the local model adapter
        adapter = get_adapter()
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename or "audio"
        file_extension = Path(original_filename).suffix or ".wav"
        filename = f"{timestamp}_{unique_id}{file_extension}"
        
        # Save uploaded file
        upload_dir = Path("results/audio_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.log_it({
            "logType": "info",
            "prefix": "file_upload",
            "logData": {
                "filename": filename,
                "original_filename": original_filename,
                "file_size": file_path.stat().st_size
            }
        })
        
        # Step 1: Audio Preprocessing
        preprocessed_path = None
        if use_hindi_enhancements or save_preprocessed_audio:
            timer.start_step("preprocessing")
            try:
                audio_processor = UnifiedAudioProcessorLocal()
                preprocessed_path = file_path.with_suffix("_preprocessed.wav")
                
                success = await process_with_timeout(
                    asyncio.to_thread(audio_processor.preprocess_audio, str(file_path), str(preprocessed_path)),
                    TIMEOUTS["preprocessing"],
                    "preprocessing",
                    timer
                )
                
                if success:
                    file_path = preprocessed_path
                    logger.log_it({
                        "logType": "info",
                        "prefix": "preprocessing",
                        "logData": {"message": "Audio preprocessing completed successfully"}
                    })
                else:
                    logger.log_it({
                        "logType": "warning",
                        "prefix": "preprocessing",
                        "logData": {"message": "Audio preprocessing failed, using original file"}
                    })
            except Exception as e:
                logger.log_it({
                    "logType": "error",
                    "prefix": "preprocessing",
                    "logData": {"error": str(e)}
                })
                timer.end_step("preprocessing")
        
        # Step 2: Transcription with Local Models
        timer.start_step("transcription")
        try:
            transcription_result = await process_with_timeout(
                asyncio.to_thread(adapter.transcribe_audio, str(file_path)),
                TIMEOUTS["transcription"],
                "transcription",
                timer
            )
            
            if not transcription_result.get("success", False):
                raise Exception(transcription_result.get("error", "Transcription failed"))
            
            transcription_text = transcription_result["transcription"]
            segments = transcription_result.get("segments", [])
            language = transcription_result.get("language", "unknown")
            provider = transcription_result.get("provider", "local")
            
            logger.log_it({
                "logType": "info",
                "prefix": "transcription",
                "logData": {
                    "language": language,
                    "provider": provider,
                    "text_length": len(transcription_text),
                    "segments_count": len(segments)
                }
            })
            
        except Exception as e:
            timer.end_step("transcription")
            logger.log_it({
                "logType": "error",
                "prefix": "transcription",
                "logData": {"error": str(e)}
            })
            raise
        
        # Step 3: Call Analysis with Local Models
        timer.start_step("analysis")
        try:
            analysis_result = await process_with_timeout(
                asyncio.to_thread(adapter.analyze_call, transcription_text, segments),
                TIMEOUTS["analysis"],
                "analysis",
                timer
            )
            
            if not analysis_result.get("success", False):
                raise Exception(analysis_result.get("error", "Analysis failed"))
            
            analysis = analysis_result["analysis"]
            analysis_provider = analysis_result.get("provider", "local")
            
            logger.log_it({
                "logType": "info",
                "prefix": "analysis",
                "logData": {
                    "provider": analysis_provider,
                    "analysis_keys": list(analysis.keys()) if isinstance(analysis, dict) else []
                }
            })
            
        except Exception as e:
            timer.end_step("analysis")
            logger.log_it({
                "logType": "error",
                "prefix": "analysis",
                "logData": {"error": str(e)}
            })
            raise
        
        # Format results
        formatted_analysis = format_analysis_text(analysis)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        csv_file = results_dir / f"call_analysis_{timestamp}_{unique_id}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Field', 'Value'])
            writer.writerow(['Timestamp', timestamp])
            writer.writerow(['Original Filename', original_filename])
            writer.writerow(['Language', language])
            writer.writerow(['Transcription Provider', provider])
            writer.writerow(['Analysis Provider', analysis_provider])
            writer.writerow(['Transcription', transcription_text])
            writer.writerow(['Analysis', json.dumps(analysis, ensure_ascii=False)])
            writer.writerow(['Processing Time', timer.get_total_elapsed()])
        
        # Prepare response data
        response_data = {
            "request": request,
            "transcription": transcription_text,
            "analysis": formatted_analysis,
            "raw_analysis": analysis,
            "language": language,
            "segments": segments,
            "processing_time": timer.get_total_elapsed(),
            "transcription_provider": provider,
            "analysis_provider": analysis_provider,
            "use_local_models": True,
            "success": True
        }
        
        return templates.TemplateResponse("results.html", response_data)
        
    except Exception as e:
        logger.log_it({
            "logType": "error",
            "prefix": "transcribe_ui",
            "logData": {"error": str(e)}
        })
        
        error_data = {
            "request": request,
            "error": str(e),
            "use_local_models": True,
            "success": False
        }
        
        return templates.TemplateResponse("error.html", error_data)

@router.get("/status-local")
async def get_status():
    """Get status of local models"""
    adapter = get_adapter()
    status = adapter.get_status()
    return {
        "status": "success",
        "data": status,
        "message": "Local models status retrieved successfully"
    }

@router.post("/switch-to-local")
async def switch_to_local():
    """Switch to using local models"""
    from local_model_adapter import switch_to_local_models
    switch_to_local_models()
    return {
        "status": "success",
        "message": "Switched to local models"
    }

@router.post("/switch-to-openai")
async def switch_to_openai():
    """Switch to using OpenAI models"""
    from local_model_adapter import switch_to_existing_models
    switch_to_existing_models()
    return {
        "status": "success",
        "message": "Switched to OpenAI models"
    }

# Add more endpoints as needed... 