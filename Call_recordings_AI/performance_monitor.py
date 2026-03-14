#!/usr/bin/env python3
"""
Performance Monitor for Call Recordings AI
==========================================
Monitor and track performance improvements
"""

import time
import json
from datetime import datetime
from typing import Dict, List

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_processing_time": 0,
            "transcription_time": 0,
            "translation_time": 0,
            "analysis_time": 0,
            "debug_analysis_time": 0,
            "preprocessing_time": 0,
            "parallel_savings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "optimization_enabled": False
        }
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        print(f"🚀 Performance monitoring started at {datetime.now().strftime('%H:%M:%S')}")
    
    def record_step(self, step_name: str, duration: float):
        """Record step duration"""
        if step_name in self.metrics:
            self.metrics[step_name] = duration
            print(f"⏱️ {step_name}: {duration:.2f}s")
    
    def record_optimization_metrics(self, metrics: Dict):
        """Record optimization metrics"""
        if "parallel_savings" in metrics:
            self.metrics["parallel_savings"] = metrics["parallel_savings"]
        if "cache_hits" in metrics:
            self.metrics["cache_hits"] = metrics["cache_hits"]
        if "cache_misses" in metrics:
            self.metrics["cache_misses"] = metrics["cache_misses"]
        if "optimization_enabled" in metrics:
            self.metrics["optimization_enabled"] = metrics["optimization_enabled"]
    
    def end_monitoring(self):
        """End performance monitoring and generate report"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.metrics["total_processing_time"] = total_time
            
            print(f"\n📊 PERFORMANCE REPORT")
            print(f"=" * 50)
            print(f"⏱️ Total Processing Time: {total_time:.2f}s")
            print(f"🎯 Optimization Enabled: {'✅ Yes' if self.metrics['optimization_enabled'] else '❌ No'}")
            
            if self.metrics["parallel_savings"] > 0:
                print(f"⚡ Parallel Savings: {self.metrics['parallel_savings']:.2f}s")
            
            if self.metrics["cache_hits"] > 0 or self.metrics["cache_misses"] > 0:
                total_cache = self.metrics["cache_hits"] + self.metrics["cache_misses"]
                hit_rate = (self.metrics["cache_hits"] / total_cache) * 100 if total_cache > 0 else 0
                print(f"💾 Cache Hit Rate: {hit_rate:.1f}% ({self.metrics['cache_hits']}/{total_cache})")
            
            # Performance analysis
            self._analyze_performance()
    
    def _analyze_performance(self):
        """Analyze performance and provide recommendations"""
        print(f"\n🔍 PERFORMANCE ANALYSIS:")
        print(f"=" * 30)
        
        # Check if processing time is acceptable
        if self.metrics["total_processing_time"] < 60:
            print(f"✅ Excellent performance! Processing completed in {self.metrics['total_processing_time']:.2f}s")
        elif self.metrics["total_processing_time"] < 120:
            print(f"⚠️ Good performance. Consider optimization for faster processing.")
        else:
            print(f"❌ Slow performance detected. Processing took {self.metrics['total_processing_time']:.2f}s")
        
        # Identify bottlenecks
        bottlenecks = []
        if self.metrics["transcription_time"] > 30:
            bottlenecks.append("Transcription (main bottleneck)")
        if self.metrics["analysis_time"] > 20:
            bottlenecks.append("Analysis")
        if self.metrics["debug_analysis_time"] > 30:
            bottlenecks.append("Debug Analysis")
        
        if bottlenecks:
            print(f"🐌 Bottlenecks identified: {', '.join(bottlenecks)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"=" * 20)
        
        if not self.metrics["optimization_enabled"]:
            print(f"• Enable optimized transcription service")
        
        if self.metrics["transcription_time"] > 30:
            print(f"• Consider using faster transcription models")
        
        if self.metrics["cache_hits"] == 0:
            print(f"• Enable caching for repeated content")
        
        if self.metrics["parallel_savings"] == 0:
            print(f"• Enable parallel processing for translation and analysis")

# Global monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    """Start performance monitoring"""
    performance_monitor.start_monitoring()

def record_performance_step(step_name: str, duration: float):
    """Record performance step"""
    performance_monitor.record_step(step_name, duration)

def end_performance_monitoring():
    """End performance monitoring"""
    performance_monitor.end_monitoring()

def get_performance_summary() -> Dict:
    """Get performance summary"""
    return performance_monitor.metrics.copy()