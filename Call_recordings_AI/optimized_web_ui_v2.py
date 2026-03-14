#!/usr/bin/env python3
"""
Optimized Web UI v2
==================
High-performance web interface using optimized transcription service
with parallel processing, caching, and performance monitoring.
"""

from fastapi import FastAPI, Request, APIRouter, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import time
import asyncio
from pathlib import Path
import tempfile
import shutil

# Import optimized services
from optimized_transcription_service import OptimizedTranscriptionService

# Initialize router
router = APIRouter()

# Initialize optimized service
optimized_service = OptimizedTranscriptionService(max_workers=4)

class OptimizedProcessingTimer:
    """Enhanced timer with parallel processing tracking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.steps = {}
        self.parallel_steps = {}
        self.current_parallel = None
    
    def start_step(self, step_name):
        self.steps[step_name] = {"start": time.time()}
    
    def end_step(self, step_name):
        if step_name in self.steps:
            self.steps[step_name]["end"] = time.time()
            self.steps[step_name]["duration"] = self.steps[step_name]["end"] - self.steps[step_name]["start"]
    
    def start_parallel_step(self, step_name):
        self.current_parallel = step_name
        self.parallel_steps[step_name] = {"start": time.time()}
    
    def end_parallel_step(self, step_name):
        if step_name in self.parallel_steps:
            self.parallel_steps[step_name]["end"] = time.time()
            self.parallel_steps[step_name]["duration"] = self.parallel_steps[step_name]["end"] - self.parallel_steps[step_name]["start"]
    
    def get_total_elapsed(self):
        return time.time() - self.start_time
    
    def get_parallel_savings(self):
        """Calculate time saved through parallel processing"""
        if not self.parallel_steps:
            return 0
        
        sequential_time = sum(step.get("duration", 0) for step in self.parallel_steps.values())
        parallel_time = max(step.get("duration", 0) for step in self.parallel_steps.values())
        return sequential_time - parallel_time

@router.get("/transcribe-ui-optimized", response_class=HTMLResponse)
async def transcribe_ui_form_optimized(request: Request):
    """Optimized transcription UI form"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Call Recordings AI - Optimized</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"] { width: 100%; padding: 10px; border: 2px dashed #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .status { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .loading { display: none; text-align: center; margin: 20px 0; }
            .results { margin-top: 20px; }
            .section { margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff; }
            .section h3 { margin-top: 0; color: #333; }
            .performance-metrics { background: #e3f2fd; border-left-color: #2196f3; }
            .optimization-info { background: #f3e5f5; border-left-color: #9c27b0; }
            .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background: #007bff; width: 0%; transition: width 0.3s; }
            .metric-item { margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }
            .metric-label { font-weight: bold; color: #555; }
            .metric-value { margin-left: 10px; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Call Recordings AI - Optimized</h1>
            <p style="text-align: center; color: #666;">High-performance audio processing with parallel execution and caching</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Audio File:</label>
                    <input type="file" id="file" name="file" accept="audio/*" required>
                </div>
                
                <div class="form-group">
                    <label for="language">Language (Optional - Auto-detect if not specified):</label>
                    <select id="language" name="language">
                        <option value="">Auto-detect</option>
                        <option value="en">English</option>
                        <option value="hi">Hindi</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                        <option value="it">Italian</option>
                        <option value="pt">Portuguese</option>
                        <option value="ru">Russian</option>
                        <option value="ja">Japanese</option>
                        <option value="ko">Korean</option>
                        <option value="zh">Chinese</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="useAdvancedDiarization" name="useAdvancedDiarization">
                        Use Advanced Speaker Diarization
                    </label>
                </div>
                
                <button type="submit">🚀 Process with Optimized Pipeline</button>
            </form>
            
            <div id="loading" class="loading">
                <p>⚡ Processing with optimized pipeline...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="loadingText">Step 1: Transcribing audio...</p>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const languageSelect = document.getElementById('language');
                const diarizationCheckbox = document.getElementById('useAdvancedDiarization');
                
                formData.append('file', fileInput.files[0]);
                if (languageSelect.value) {
                    formData.append('language', languageSelect.value);
                }
                if (diarizationCheckbox.checked) {
                    formData.append('use_advanced_diarization', 'true');
                }
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const progressFill = document.getElementById('progressFill');
                const loadingText = document.getElementById('loadingText');
                
                loading.style.display = 'block';
                result.innerHTML = '';
                
                // Simulate progress with optimized pipeline
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += Math.random() * 15; // Faster progress due to optimization
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                    
                    if (progress < 40) {
                        loadingText.textContent = 'Step 1: Transcribing audio (optimized)...';
                    } else if (progress < 70) {
                        loadingText.textContent = 'Step 2: Parallel translation & analysis...';
                    } else {
                        loadingText.textContent = 'Step 3: Finalizing results...';
                    }
                }, 300); // Faster updates
                
                try {
                    const response = await fetch('/transcribe-optimized', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    clearInterval(progressInterval);
                    progressFill.style.width = '100%';
                    
                    if (data.success) {
                        const transcription = data.transcription;
                        const analysis = data.analysis;
                        const performance = data.performance_metrics;
                        
                        let resultHtml = `
                            <div class="status success">
                                <h3>⚡ Processing Complete!</h3>
                                <p><strong>Total Processing Time:</strong> ${performance.total_processing_time?.toFixed(2) || 'Unknown'} seconds</p>
                                <p><strong>Optimization Enabled:</strong> ✅ Yes</p>
                                <p><strong>Parallel Savings:</strong> ${performance.parallel_savings?.toFixed(2) || '0'} seconds</p>
                                <p><strong>Speedup Factor:</strong> ${performance.speedup_factor?.toFixed(2) || '1'}x</p>
                            </div>
                            
                            <div class="results">
                                <div class="section performance-metrics">
                                    <h3>📊 Performance Metrics</h3>
                                    <div class="metric-item">
                                        <span class="metric-label">Transcription Time:</span>
                                        <span class="metric-value">${performance.transcription_time?.toFixed(2) || 'Unknown'} seconds</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="metric-label">Translation Time:</span>
                                        <span class="metric-value">${performance.translation_time?.toFixed(2) || 'Unknown'} seconds</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="metric-label">Analysis Time:</span>
                                        <span class="metric-value">${performance.analysis_time?.toFixed(2) || 'Unknown'} seconds</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="metric-label">Cache Hit Rate:</span>
                                        <span class="metric-value">${(performance.cache_hit_rate * 100)?.toFixed(1) || '0'}%</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="metric-label">Cache Hits:</span>
                                        <span class="metric-value">${performance.cache_hits || 0}</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="metric-label">Cache Misses:</span>
                                        <span class="metric-value">${performance.cache_misses || 0}</span>
                                    </div>
                                </div>
                                
                                <div class="section">
                                    <h3>📝 Transcription Results</h3>
                                    <p><strong>Language:</strong> ${transcription.language || 'Unknown'}</p>
                                    <p><strong>Text:</strong></p>
                                    <div style="background: white; padding: 15px; border-radius: 5px; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${transcription.transcription}</div>
                                </div>
                        `;
                        
                        if (analysis.success) {
                            const analysisData = analysis.analysis;
                            resultHtml += `
                                <div class="section">
                                    <h3>🔍 Analysis Results</h3>
                            `;
                            
                            if (typeof analysisData === 'object') {
                                if (analysisData.main_issue) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Main Issue:</span><span class="metric-value">${analysisData.main_issue}</span></div>`;
                                }
                                
                                if (analysisData.customer_needs && Array.isArray(analysisData.customer_needs)) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Customer Needs:</span><span class="metric-value">${analysisData.customer_needs.join(', ')}</span></div>`;
                                }
                                
                                if (analysisData.agent_actions && Array.isArray(analysisData.agent_actions)) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Agent Actions:</span><span class="metric-value">${analysisData.agent_actions.join(', ')}</span></div>`;
                                }
                                
                                if (analysisData.resolution_status) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Resolution Status:</span><span class="metric-value">${analysisData.resolution_status}</span></div>`;
                                }
                                
                                if (analysisData.customer_satisfaction) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Customer Satisfaction:</span><span class="metric-value">${analysisData.customer_satisfaction}</span></div>`;
                                }
                                
                                if (analysisData.agent_performance && typeof analysisData.agent_performance === 'object') {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Agent Performance:</span></div>`;
                                    for (const [key, value] of Object.entries(analysisData.agent_performance)) {
                                        resultHtml += `<div class="metric-item" style="margin-left: 20px;"><span class="metric-label">${key.replace('_', ' ').toUpperCase()}:</span><span class="metric-value">${value}</span></div>`;
                                    }
                                }
                                
                                if (analysisData.recommendations && Array.isArray(analysisData.recommendations)) {
                                    resultHtml += `<div class="metric-item"><span class="metric-label">Recommendations:</span><span class="metric-value">${analysisData.recommendations.join(', ')}</span></div>`;
                                }
                            } else {
                                resultHtml += `<div class="metric-item"><span class="metric-label">Analysis:</span><span class="metric-value">${analysisData}</span></div>`;
                            }
                            
                            resultHtml += `</div>`;
                        }
                        
                        resultHtml += `</div>`;
                        result.innerHTML = resultHtml;
                        
                    } else {
                        result.innerHTML = `
                            <div class="status error">
                                <h3>❌ Processing Failed</h3>
                                <p>${data.error}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    result.innerHTML = `
                        <div class="status error">
                            <h3>❌ Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                } finally {
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.post("/transcribe-optimized")
async def transcribe_optimized(
    request: Request, 
    file: UploadFile = File(...),
    language: str = Form(None),
    use_advanced_diarization: bool = Form(False)
):
    """Optimized transcription endpoint"""
    timer = OptimizedProcessingTimer()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        timer.start_step("file_upload")
        timer.end_step("file_upload")
        
        # Process with optimized service
        timer.start_step("optimized_processing")
        result = await optimized_service.process_audio_optimized(
            temp_file_path, 
            use_advanced_diarization=use_advanced_diarization
        )
        timer.end_step("optimized_processing")
        
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        if result["status"] == "error":
            return {
                "success": False,
                "error": result.get("error", "Processing failed"),
                "performance_metrics": optimized_service.get_performance_summary()
            }
        
        # Get performance summary
        performance_summary = optimized_service.get_performance_summary()
        
        return {
            "success": True,
            "transcription": {
                "transcription": result.get("transcribed_text", ""),
                "language": result.get("detected_language", "unknown"),
                "segments": result.get("segments", [])
            },
            "analysis": result.get("analysis", {}),
            "performance_metrics": performance_summary,
            "processing_time": timer.get_total_elapsed()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "performance_metrics": optimized_service.get_performance_summary()
        }

@router.get("/performance-summary")
async def get_performance_summary():
    """Get performance summary from optimized service"""
    return optimized_service.get_performance_summary()

@router.post("/clear-cache")
async def clear_cache():
    """Clear all cached results"""
    optimized_service.clear_cache()
    return {"success": True, "message": "Cache cleared successfully"}

@router.get("/optimization-status")
async def get_optimization_status():
    """Get optimization status and configuration"""
    return {
        "max_workers": optimized_service.max_workers,
        "cache_enabled": True,
        "parallel_processing": True,
        "batch_translation": True,
        "performance_tracking": True
    } 