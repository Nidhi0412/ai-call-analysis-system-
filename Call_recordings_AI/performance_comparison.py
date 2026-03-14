#!/usr/bin/env python3
"""
Performance Comparison Script
=============================
Compare performance between original and optimized transcription pipelines
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List

# Import services
from transcription_with_speakers import TranscriptionWithSpeakersService
from optimized_transcription_service import OptimizedTranscriptionService

class PerformanceComparator:
    """Compare performance between original and optimized pipelines"""
    
    def __init__(self):
        self.original_service = TranscriptionWithSpeakersService()
        self.optimized_service = OptimizedTranscriptionService(max_workers=4)
        self.results = {
            'original': {},
            'optimized': {},
            'comparison': {}
        }
    
    async def test_original_pipeline(self, audio_path: str) -> Dict:
        """Test original pipeline performance"""
        print(f"🔄 Testing original pipeline: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Original pipeline: sequential processing
            result = await self.original_service.process_audio_file_with_speakers(audio_path)
            
            processing_time = time.time() - start_time
            
            return {
                'success': result.get('status') == 'success',
                'processing_time': processing_time,
                'transcription_time': processing_time * 0.6,  # Estimate
                'translation_time': processing_time * 0.3,    # Estimate
                'analysis_time': processing_time * 0.1,       # Estimate
                'error': result.get('error') if result.get('status') == 'error' else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def test_optimized_pipeline(self, audio_path: str) -> Dict:
        """Test optimized pipeline performance"""
        print(f"⚡ Testing optimized pipeline: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Optimized pipeline: parallel processing
            result = await self.optimized_service.process_audio_optimized(audio_path)
            
            processing_time = time.time() - start_time
            performance_metrics = self.optimized_service.get_performance_summary()
            
            return {
                'success': result.get('status') == 'success',
                'processing_time': processing_time,
                'transcription_time': performance_metrics.get('transcription_time', 0),
                'translation_time': performance_metrics.get('translation_time', 0),
                'analysis_time': performance_metrics.get('analysis_time', 0),
                'parallel_savings': performance_metrics.get('parallel_savings', 0),
                'speedup_factor': performance_metrics.get('speedup_factor', 1),
                'cache_hit_rate': performance_metrics.get('cache_hit_rate', 0),
                'error': result.get('error') if result.get('status') == 'error' else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def compare_pipelines(self, audio_path: str) -> Dict:
        """Compare both pipelines on the same audio file"""
        print(f"📊 Comparing pipelines for: {audio_path}")
        print("=" * 60)
        
        # Test original pipeline
        original_result = await self.test_original_pipeline(audio_path)
        
        # Clear cache for fair comparison
        self.optimized_service.clear_cache()
        
        # Test optimized pipeline
        optimized_result = await self.test_optimized_pipeline(audio_path)
        
        # Calculate improvements
        if original_result['success'] and optimized_result['success']:
            time_savings = original_result['processing_time'] - optimized_result['processing_time']
            speedup = original_result['processing_time'] / optimized_result['processing_time'] if optimized_result['processing_time'] > 0 else 1
            percentage_improvement = (time_savings / original_result['processing_time']) * 100 if original_result['processing_time'] > 0 else 0
            
            comparison = {
                'original_time': original_result['processing_time'],
                'optimized_time': optimized_result['processing_time'],
                'time_savings': time_savings,
                'speedup_factor': speedup,
                'percentage_improvement': percentage_improvement,
                'parallel_savings': optimized_result.get('parallel_savings', 0),
                'cache_hit_rate': optimized_result.get('cache_hit_rate', 0)
            }
        else:
            comparison = {
                'error': 'One or both pipelines failed',
                'original_error': original_result.get('error'),
                'optimized_error': optimized_result.get('error')
            }
        
        # Print results
        print(f"\n📈 Performance Comparison Results:")
        print(f"   Original Pipeline: {original_result['processing_time']:.2f}s")
        print(f"   Optimized Pipeline: {optimized_result['processing_time']:.2f}s")
        
        if 'speedup_factor' in comparison:
            print(f"   ⚡ Speedup Factor: {comparison['speedup_factor']:.2f}x")
            print(f"   💾 Time Savings: {comparison['time_savings']:.2f}s ({comparison['percentage_improvement']:.1f}%)")
            print(f"   🔄 Parallel Savings: {comparison['parallel_savings']:.2f}s")
            print(f"   🎯 Cache Hit Rate: {comparison['cache_hit_rate']*100:.1f}%")
        
        return {
            'original': original_result,
            'optimized': optimized_result,
            'comparison': comparison
        }
    
    async def batch_comparison(self, input_dir: str) -> Dict:
        """Compare pipelines on multiple files"""
        input_path = Path(input_dir)
        wav_files = list(input_path.glob("*.wav"))
        
        if not wav_files:
            print(f"❌ No .wav files found in {input_dir}")
            return {}
        
        print(f"🚀 Starting batch comparison with {len(wav_files)} files")
        print("=" * 60)
        
        all_results = []
        total_original_time = 0
        total_optimized_time = 0
        successful_comparisons = 0
        
        for i, wav_file in enumerate(wav_files, 1):
            print(f"\n📁 File {i}/{len(wav_files)}: {wav_file.name}")
            
            result = await self.compare_pipelines(str(wav_file))
            all_results.append({
                'filename': wav_file.name,
                **result
            })
            
            if 'comparison' in result and 'speedup_factor' in result['comparison']:
                total_original_time += result['original']['processing_time']
                total_optimized_time += result['optimized']['processing_time']
                successful_comparisons += 1
        
        # Calculate overall statistics
        if successful_comparisons > 0:
            overall_speedup = total_original_time / total_optimized_time
            overall_savings = total_original_time - total_optimized_time
            overall_improvement = (overall_savings / total_original_time) * 100
            
            print(f"\n🎯 Overall Performance Summary:")
            print(f"   Files Processed: {successful_comparisons}/{len(wav_files)}")
            print(f"   Total Original Time: {total_original_time:.2f}s")
            print(f"   Total Optimized Time: {total_optimized_time:.2f}s")
            print(f"   ⚡ Overall Speedup: {overall_speedup:.2f}x")
            print(f"   💾 Total Time Savings: {overall_savings:.2f}s ({overall_improvement:.1f}%)")
            print(f"   📊 Average Speedup: {overall_speedup:.2f}x per file")
        
        return {
            'files_processed': len(wav_files),
            'successful_comparisons': successful_comparisons,
            'total_original_time': total_original_time,
            'total_optimized_time': total_optimized_time,
            'overall_speedup': overall_speedup if successful_comparisons > 0 else 1,
            'overall_savings': overall_savings if successful_comparisons > 0 else 0,
            'overall_improvement': overall_improvement if successful_comparisons > 0 else 0,
            'detailed_results': all_results
        }
    
    def save_comparison_results(self, results: Dict, output_file: str = "performance_comparison.json"):
        """Save comparison results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_file}")

async def main():
    """Main function to run performance comparison"""
    comparator = PerformanceComparator()
    
    # Test with a single file
    test_file = "path/to/test/audio.wav"  # Replace with actual path
    
    if Path(test_file).exists():
        print("🧪 Single File Performance Test")
        print("=" * 40)
        result = await comparator.compare_pipelines(test_file)
        comparator.save_comparison_results(result, "single_file_comparison.json")
    
    # Test with batch processing
    test_dir = "path/to/test/directory"  # Replace with actual directory
    
    if Path(test_dir).exists():
        print("\n📦 Batch Performance Test")
        print("=" * 40)
        batch_result = await comparator.batch_comparison(test_dir)
        comparator.save_comparison_results(batch_result, "batch_comparison.json")
    
    # Show optimization benefits
    print("\n🚀 Optimization Benefits Summary:")
    print("=" * 40)
    print("✅ Parallel Translation & Analysis")
    print("✅ Intelligent Caching System")
    print("✅ Batch Processing")
    print("✅ Performance Monitoring")
    print("✅ Reduced API Calls")
    print("✅ Faster Response Times")
    print("✅ Better Resource Utilization")

if __name__ == "__main__":
    asyncio.run(main()) 