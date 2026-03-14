#!/usr/bin/env python3
"""
Cost Calculator for Call Recordings AI
======================================
Calculate exact billable costs in Indian Rupees (INR)
"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional

class CostCalculator:
    """Calculate exact billable costs in INR"""
    
    def __init__(self):
        # OpenAI API pricing (as of 2024)
        self.pricing = {
            "whisper": {
                "model": "whisper-1",
                "cost_per_minute": 0.006  # USD per minute
            },
            "gpt-4o-mini": {
                "input_cost_per_1k": 0.00015,   # USD per 1k input tokens
                "output_cost_per_1k": 0.0006    # USD per 1k output tokens
            },
            "gpt-3.5-turbo": {
                "input_cost_per_1k": 0.0005,    # USD per 1k input tokens
                "output_cost_per_1k": 0.0015    # USD per 1k output tokens
            }
        }
        
        # Current USD to INR exchange rate (you can update this)
        self.usd_to_inr = 83.50  # Approximate rate as of 2024
    
    def get_current_exchange_rate(self) -> float:
        """Get current USD to INR exchange rate from API"""
        try:
            # Using a free exchange rate API
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data['rates']['INR']
            else:
                return self.usd_to_inr  # Fallback to default rate
        except:
            return self.usd_to_inr  # Fallback to default rate
    
    def calculate_transcription_cost(self, duration_minutes: float) -> Dict:
        """Calculate transcription cost"""
        cost_usd = duration_minutes * self.pricing["whisper"]["cost_per_minute"]
        cost_inr = cost_usd * self.usd_to_inr
        
        return {
            "service": "Transcription (Whisper)",
            "duration_minutes": duration_minutes,
            "cost_usd": round(cost_usd, 6),
            "cost_inr": round(cost_inr, 2),
            "rate_usd_per_minute": self.pricing["whisper"]["cost_per_minute"]
        }
    
    def calculate_llm_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> Dict:
        """Calculate LLM cost (translation/analysis)"""
        if model not in self.pricing:
            model = "gpt-4o-mini"  # Default to gpt-4o-mini
        
        pricing = self.pricing[model]
        input_cost_usd = (input_tokens / 1000) * pricing["input_cost_per_1k"]
        output_cost_usd = (output_tokens / 1000) * pricing["output_cost_per_1k"]
        total_cost_usd = input_cost_usd + output_cost_usd
        total_cost_inr = total_cost_usd * self.usd_to_inr
        
        return {
            "service": f"LLM ({model})",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(total_cost_usd, 6),
            "cost_inr": round(total_cost_inr, 2),
            "input_cost_usd": round(input_cost_usd, 6),
            "output_cost_usd": round(output_cost_usd, 6),
            "rate_input_usd_per_1k": pricing["input_cost_per_1k"],
            "rate_output_usd_per_1k": pricing["output_cost_per_1k"]
        }
    
    def calculate_total_cost(self, token_usage: Dict, duration_minutes: Optional[float] = None) -> Dict:
        """Calculate total billable cost"""
        # Update exchange rate
        self.usd_to_inr = self.get_current_exchange_rate()
        
        total_cost_usd = 0
        total_cost_inr = 0
        breakdown = []
        
        # Transcription cost
        if duration_minutes:
            transcription_cost = self.calculate_transcription_cost(duration_minutes)
            total_cost_usd += transcription_cost["cost_usd"]
            total_cost_inr += transcription_cost["cost_inr"]
            breakdown.append(transcription_cost)
        
        # Translation cost (if any)
        translation_tokens = token_usage.get("translation_tokens", 0)
        if translation_tokens > 0:
            # Estimate input/output split (typically 70/30 for translation)
            input_tokens = int(translation_tokens * 0.7)
            output_tokens = translation_tokens - input_tokens
            translation_cost = self.calculate_llm_cost(input_tokens, output_tokens, "gpt-3.5-turbo")
            translation_cost["service"] = "Translation (GPT-3.5-turbo)"
            total_cost_usd += translation_cost["cost_usd"]
            total_cost_inr += translation_cost["cost_inr"]
            breakdown.append(translation_cost)
        
        # Analysis cost
        analysis_tokens = token_usage.get("analysis_tokens", 0)
        if analysis_tokens > 0:
            # Estimate input/output split (typically 60/40 for analysis)
            input_tokens = int(analysis_tokens * 0.6)
            output_tokens = analysis_tokens - input_tokens
            analysis_cost = self.calculate_llm_cost(input_tokens, output_tokens, "gpt-4o-mini")
            analysis_cost["service"] = "Analysis (GPT-4o-mini)"
            total_cost_usd += analysis_cost["cost_usd"]
            total_cost_inr += analysis_cost["cost_inr"]
            breakdown.append(analysis_cost)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "exchange_rate_usd_to_inr": self.usd_to_inr,
            "total_cost_usd": round(total_cost_usd, 6),
            "total_cost_inr": round(total_cost_inr, 2),
            "breakdown": breakdown,
            "token_usage": token_usage,
            "duration_minutes": duration_minutes
        }
    
    def generate_cost_report(self, cost_data: Dict) -> str:
        """Generate a formatted cost report for managers"""
        report = f"""
📊 **CALL RECORDINGS AI - COST REPORT**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💱 **Exchange Rate:** 1 USD = ₹{cost_data['exchange_rate_usd_to_inr']:.2f}

💰 **TOTAL BILLABLE COST: ₹{cost_data['total_cost_inr']:.2f}** (${cost_data['total_cost_usd']:.6f})

📋 **DETAILED BREAKDOWN:**

"""
        
        for item in cost_data['breakdown']:
            report += f"**{item['service']}:**\n"
            if 'duration_minutes' in item:
                report += f"  • Duration: {item['duration_minutes']:.2f} minutes\n"
            if 'total_tokens' in item:
                report += f"  • Tokens: {item['total_tokens']:,} (Input: {item['input_tokens']:,}, Output: {item['output_tokens']:,})\n"
            report += f"  • Cost: ₹{item['cost_inr']:.2f} (${item['cost_usd']:.6f})\n\n"
        
        report += f"""
📈 **TOKEN USAGE SUMMARY:**
• Transcription Tokens: {cost_data['token_usage'].get('transcription_tokens', 0):,}
• Translation Tokens: {cost_data['token_usage'].get('translation_tokens', 0):,}
• Analysis Tokens: {cost_data['token_usage'].get('analysis_tokens', 0):,}
• Total Tokens: {cost_data['token_usage'].get('total_tokens', 0):,}

⚠️ **NOTE:** This is the exact billable amount based on actual API usage and current exchange rates.
"""
        
        return report

# Example usage
if __name__ == "__main__":
    calculator = CostCalculator()
    
    # Example token usage from your logs
    token_usage = {
        "transcription_tokens": 294,
        "translation_tokens": 0,
        "analysis_tokens": 2725,
        "total_tokens": 3019
    }
    
    # Calculate cost (assuming 1.44 minutes duration for 86.3 seconds)
    cost_data = calculator.calculate_total_cost(token_usage, duration_minutes=1.44)
    
    # Generate report
    report = calculator.generate_cost_report(cost_data)
    print(report) 