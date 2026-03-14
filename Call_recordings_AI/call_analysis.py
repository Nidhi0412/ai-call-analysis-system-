import os
import logging
import asyncio
from typing import Dict, List, Optional
import json
import time
import openai
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import CONFIG  # Commented out - using hardcoded API key instead

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

app_env = os.getenv('NODE_ENV', 'Development')
# env_config = CONFIG.get(app_env, {})  # Commented out - using hardcoded API key instead

class CallAnalysisService:
    """
    AI-powered call analysis service that analyzes transcriptions and provides insights
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the call analysis service
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
        """
        # Initialize OpenAI client
        if api_key is None:
            # Get API key from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Token usage tracking
        self.token_usage = {
            "analysis_tokens": 0,
            "total_tokens": 0
        }
        
        logger.log_it({
            "logType": "info",
            "prefix": "call_analysis",
            "logData": {
                "message": "CallAnalysisService initialized"
            }
        })
    
    async def analyze_call_transcription(self, transcription_text: str, segments: List[Dict] = None) -> Dict:
        """
        Analyze call transcription and provide structured insights
        
        Args:
            transcription_text: Full transcription text
            segments: List of speaker segments with timestamps
            
        Returns:
            dict: Structured analysis results
        """
        try:
            logger.log_it({
                "logType": "info",
                "prefix": "call_analysis",
                "logData": {
                    "message": "Starting call analysis",
                    "text_length": len(transcription_text),
                    "segments_count": len(segments) if segments else 0
                }
            })
            
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(transcription_text, segments)
            
            # Use OpenAI for analysis
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert call center analyst specializing in Indian customer service calls. Analyze the call transcription and provide structured insights in JSON format. 

SPECIAL EXPERTISE:
- Indian regional accents (Telugu, Bengali, Marathi, Punjabi, Gujarati)
- Hindi/English code-mixing (Hinglish)
- Indian business context and cultural nuances
- Regional pronunciation variations
- Indian customer service terminology

Focus on:
1. Main issue and customer needs with cultural context
2. Support provided by agent with regional adaptation
3. Actions taken and cultural sensitivity
4. Agent's emotional state, skills, and multilingual ability
5. Customer satisfaction with cultural understanding
6. Call quality metrics including code-mixing handling
7. Business-specific insights with Indian context
8. Regional considerations and improvements

Provide detailed, professional analysis with specific examples from the conversation, considering Indian cultural and linguistic context."""
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Track token usage
            if response.usage:
                self.token_usage["analysis_tokens"] += response.usage.total_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens
                
                logger.log_it({
                    "logType": "info",
                    "prefix": "call_analysis",
                    "logData": {
                        "message": "Analysis token usage tracked",
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                })
            
            # Parse JSON response
            try:
                analysis_result = json.loads(analysis_text)
            except json.JSONDecodeError:
                # Fallback to structured text if JSON parsing fails
                analysis_result = self._parse_analysis_fallback(analysis_text)
            
            # Add metadata
            analysis_result["metadata"] = {
                "analysis_timestamp": time.time(),
                "text_length": len(transcription_text),
                "segments_count": len(segments) if segments else 0,
                "token_usage": self.token_usage.copy()
            }
            
            logger.log_it({
                "logType": "info",
                "prefix": "call_analysis",
                "logData": {
                    "message": "Call analysis completed successfully",
                    "main_issue": analysis_result.get("main_issue", "Unknown"),
                    "call_quality_score": analysis_result.get("overall_call_quality", {}).get("score", 0),
                    "issue_resolved": analysis_result.get("issue_resolution", {}).get("resolved", False)
                }
            })
            
            return analysis_result
            
        except Exception as e:
            logger.log_it({
                "logType": "error",
                "prefix": "call_analysis",
                "logData": {
                    "message": "Call analysis failed",
                    "error": str(e),
                    "text_length": len(transcription_text) if transcription_text else 0
                }
            })
            return {
                "error": str(e),
                "status": "failed",
                "metadata": {
                    "analysis_timestamp": time.time(),
                    "error": str(e)
                }
            }
    
    def _create_analysis_prompt(self, transcription_text: str, segments: List[Dict] = None) -> str:
        """
        Create analysis prompt for OpenAI with Indian context and code-mixing support
        
        Args:
            transcription_text: Full transcription text
            segments: List of speaker segments
            
        Returns:
            str: Formatted analysis prompt
        """
        # Prepare speaker analysis if segments available
        speaker_analysis = ""
        if segments:
            agent_segments = [seg for seg in segments if seg.get("speaker_type") == "Agent"]
            caller_segments = [seg for seg in segments if seg.get("speaker_type") == "Caller"]
            
            speaker_analysis = f"""
Speaker Analysis:
- Agent segments: {len(agent_segments)}
- Caller segments: {len(caller_segments)}
- Total segments: {len(segments)}

Agent's key statements:
{chr(10).join([f"- {seg.get('text', '')}" for seg in agent_segments[:5]])}

Caller's key statements:
{chr(10).join([f"- {seg.get('text', '')}" for seg in caller_segments[:5]])}
"""
        
        # Enhanced prompt with Indian context and code-mixing support
        prompt = f"""Analyze this Indian customer service call transcription and provide structured insights in JSON format.

IMPORTANT CONTEXT: This is an Indian customer service call that may contain:
- Hindi/Hinglish with regional accents (Telugu/Bengali/Marathi/Punjabi/Gujarati)
- Code-mixing between Hindi, regional languages, and English
- Indian business context and cultural nuances
- Regional pronunciation variations
- Customer service terminology in Indian context

BUSINESS CONTEXT: This is a customer service call for a hospitality/SaaS service provider. Common issues include:
- Booking problems, Service modifications
- Technical support for cloud solutions, Remote access
- Remote desktop connection issues
- Email/contact problems
- Billing and reservation issues
- Business Terms: Revenue, Front Desk, Check-in, Check-out, Service Management, System Access
- Financial Terms: Invoice, Payment, Receipt, Tax, Service Charge, Cancellation Fee
- Operations: Maintenance, Customer Service, Support
- Common Agent Requests: Service Code, Property Code, Access Credentials, Login Information
- Number Recognition Patterns: Direct numbers, Spelled out numbers, Mixed format, Repeated digits
- Authentication Terms: Username, Password, Login, Credentials, Access Code, PIN, OTP, Verification Code

TRANSCRIPTION:
{transcription_text}

{speaker_analysis}

Please provide analysis in this JSON structure, considering Indian cultural context and company-specific business domain:
{{
    "main_issue": {{
        "description": "Brief description of the main customer issue",
        "category": "Technical/Booking/Billing/Support/etc",
        "urgency": "High/Medium/Low",
        "cultural_context": "Any Indian-specific context or regional factors"
    }},
    "support_given": {{
        "description": "What support was provided by the agent",
        "steps_taken": ["Step 1", "Step 2", "Step 3"],
        "effectiveness": "Excellent/Good/Fair/Poor",
        "cultural_sensitivity": "How well the agent handled Indian cultural context"
    }},
    "action_taken": {{
        "description": "Specific actions taken during the call",
        "resolution_steps": ["Action 1", "Action 2"],
        "completion_status": "Complete/Partial/Pending"
    }},
    "agent_emotion": {{
        "overall_tone": "Patient/Professional/Friendly/Stressed",
        "emotional_state": "Calm/Anxious/Confident/Uncertain",
        "communication_style": "Clear/Unclear/Professional/Casual",
        "regional_adaptation": "How well agent adapted to regional accent/language"
    }},
    "issue_resolution": {{
        "resolved": true/false,
        "resolution_method": "Immediate/Follow-up/Escalation",
        "customer_satisfaction": "Satisfied/Neutral/Dissatisfied"
    }},
    "agent_engagement": {{
        "level": "High/Medium/Low",
        "active_listening": true/false,
        "proactive_assistance": true/false,
        "response_time": "Immediate/Quick/Slow",
        "code_mixing_handling": "How well agent handled Hindi/English code-mixing"
    }},
    "agent_skill": {{
        "technical_knowledge": "Excellent/Good/Fair/Poor",
        "problem_solving": "Excellent/Good/Fair/Poor",
        "communication": "Excellent/Good/Fair/Poor",
        "patience": "Excellent/Good/Fair/Poor",
        "multilingual_ability": "How well agent handled multiple languages/accents"
    }},
    "customer_satisfaction": {{
        "overall_satisfaction": "Very Satisfied/Satisfied/Neutral/Dissatisfied",
        "willingness_to_recommend": "Likely/Maybe/Unlikely",
        "key_satisfaction_factors": ["Factor 1", "Factor 2"],
        "cultural_satisfaction": "How satisfied customer was with cultural understanding"
    }},
    "call_tone": {{
        "overall_atmosphere": "Professional/Friendly/Formal/Casual",
        "respect_level": "High/Medium/Low",
        "conflict_present": true/false,
        "cultural_respect": "Level of cultural respect shown"
    }},
    "overall_call_quality": {{
        "score": 1-10,
        "rating": "Excellent/Good/Fair/Poor",
        "strengths": ["Strength 1", "Strength 2"],
        "areas_for_improvement": ["Area 1", "Area 2"],
        "regional_considerations": "Any regional-specific improvements needed"
    }},
    "business_insights": {{
        "hotel_restaurant_code": "Code if found or 'None'",
        "booking_details": "Any booking information found",
        "billing_issues": "Any billing problems mentioned",
        "technical_issues": "Any technical problems discussed",
        "indian_business_context": "Any India-specific business context"
    }},
    "query_analysis": {{
        "primary_query": "Main customer query",
        "query_type": "Booking/Billing/Technical/Support/Other",
        "complexity": "Simple/Moderate/Complex",
        "resolution_time": "Immediate/Quick/Extended"
    }}
}}

Focus on providing specific, actionable insights based on the actual conversation content."""
        
        return prompt
    
    def _parse_analysis_fallback(self, analysis_text: str) -> Dict:
        """
        Parse analysis text as fallback if JSON parsing fails
        
        Args:
            analysis_text: Raw analysis text
            
        Returns:
            dict: Structured analysis result
        """
        # Simple parsing fallback
        result = {
            "main_issue": {"description": "Analysis parsing failed", "category": "Unknown", "urgency": "Unknown"},
            "support_given": {"description": "Unable to parse", "steps_taken": [], "effectiveness": "Unknown"},
            "action_taken": {"description": "Unable to parse", "resolution_steps": [], "completion_status": "Unknown"},
            "agent_emotion": {"overall_tone": "Unknown", "emotional_state": "Unknown", "communication_style": "Unknown"},
            "issue_resolution": {"resolved": False, "resolution_method": "Unknown", "customer_satisfaction": "Unknown"},
            "agent_engagement": {"level": "Unknown", "active_listening": False, "proactive_assistance": False, "response_time": "Unknown"},
            "agent_skill": {"technical_knowledge": "Unknown", "problem_solving": "Unknown", "communication": "Unknown", "patience": "Unknown"},
            "customer_satisfaction": {"overall_satisfaction": "Unknown", "willingness_to_recommend": "Unknown", "key_satisfaction_factors": []},
            "call_tone": {"overall_atmosphere": "Unknown", "respect_level": "Unknown", "conflict_present": False},
            "overall_call_quality": {"score": 0, "rating": "Unknown", "strengths": [], "areas_for_improvement": []},
            "business_insights": {"hotel_restaurant_code": "None", "booking_details": "None", "billing_issues": "None", "technical_issues": "None"},
            "query_analysis": {"primary_query": "Unknown", "query_type": "Unknown", "complexity": "Unknown", "resolution_time": "Unknown"},
            "raw_analysis": analysis_text
        }
        
        return result
    
    def get_token_usage(self) -> Dict:
        """
        Get current token usage statistics
        
        Returns:
            dict: Token usage summary
        """
        return {
            "analysis_tokens": self.token_usage["analysis_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
            "estimated_cost_usd": self._estimate_cost()
        }
    
    def reset_token_usage(self):
        """
        Reset token usage tracking
        """
        self.token_usage = {
            "analysis_tokens": 0,
            "total_tokens": 0
        }
        logger.log_it({
            "logType": "info",
            "prefix": "call_analysis",
            "logData": {
                "message": "Token usage tracking reset"
            }
        })
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost based on token usage
        
        Returns:
            float: Estimated cost in USD
        """
        # GPT-4o-mini: ~$0.00015 per 1K tokens
        analysis_cost = (self.token_usage["analysis_tokens"] / 1000) * 0.00015
        return analysis_cost

def main():
    """Example usage of the CallAnalysisService"""
    # Initialize service
    analysis_service = CallAnalysisService()
    
    # Reset token usage at start
    analysis_service.reset_token_usage()
    
    # Example transcription
    sample_transcription = """
    Agent: Hello, thank you for calling our support line. How may I assist you today?
    Customer: Hi, I'm having trouble adding GST details to my booking.
    Agent: I understand. Let me help you with that. Can you tell me which booking you're referring to?
    Customer: It's booking number 12345 for the hotel reservation.
    Agent: Perfect. I can see your booking. Let me guide you through adding the GST details.
    Customer: That would be great, thank you.
    Agent: First, you'll need to go to the company details section. Do you see that option?
    Customer: Yes, I can see it.
    Agent: Excellent. Now click on 'Add Company' and enter your company name.
    Customer: Okay, I've done that.
    Agent: Great! Now in the GST field, enter your GST number.
    Customer: I've entered it. What's next?
    Agent: Perfect! Now click 'Save' and then 'Proceed to Reservation'.
    Customer: Done! It's working now. Thank you so much for your help.
    Agent: You're very welcome! Is there anything else you need assistance with?
    Customer: No, that's all. You've been very helpful.
    Agent: Thank you for calling. Have a great day!
    Customer: You too, bye!
    """
    
    # Analyze the transcription
    result = asyncio.run(analysis_service.analyze_call_transcription(sample_transcription))
    
    # Display results
    print("Call Analysis Results:")
    print(json.dumps(result, indent=2))
    
    # Display token usage
    token_usage = analysis_service.get_token_usage()
    print(f"\nToken Usage: {token_usage}")

if __name__ == "__main__":
    main() 