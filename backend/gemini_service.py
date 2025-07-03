import google.generativeai as genai
import os
import time
import logging
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Response structure for Gemini API calls."""
    success: bool
    content: str
    error = None
    metadata = None


class GeminiProService:
    """Service for interacting with Google Gemini Pro 2.5 API."""
    
    def __init__(self):
        """
        Initialize Gemini Pro service.
        
        Args:
            api_key: Google AI API key. If None, will try to get from environment variable.
        """
        logger.info("Initializing GeminiProService")
        
        self.api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            logger.error("Google AI API key not found in environment variables")
            raise ValueError("Google AI API key is required. Set GOOGLE_AI_API_KEY environment variable.")
        
        logger.debug("API key found, configuring Gemini API")
        
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            logger.info("Gemini model initialized: gemini-2.0-flash-lite")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Default generation config
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        logger.debug(f"Generation config set: {self.generation_config}")
        logger.info("GeminiProService initialization completed")
    
    def enhance_transcription(self, 
                            raw_transcription: str, 
                            context: str = None,
                            language: str = "Arabic") -> GeminiResponse:
        """
        Enhance and correct the raw lip reading transcription using Gemini Pro.
        
        Args:
            raw_transcription: Raw text output from lip reading model
            context: Additional context about the video content
            language: Language of the transcription (default: Arabic)
            
        Returns:
            GeminiResponse with enhanced transcription
        """
        logger.info(f"Starting transcription enhancement for {language} text")
        logger.debug(f"Raw transcription: '{raw_transcription}'")
        logger.debug(f"Context: {context}")
        
        try:
            # Construct the enhancement prompt
            logger.debug("Building enhancement prompt")
            prompt = self._build_enhancement_prompt(raw_transcription, context, language)
            
            # Generate response
            logger.info("Sending request to Gemini API")
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract the enhanced text
            enhanced_text = response.text.strip()
            logger.info(f"Enhancement completed successfully")
            logger.debug(f"Enhanced text: '{enhanced_text}'")
            
            return GeminiResponse(
                success=True,
                content=enhanced_text,
                metadata={
                    'original_text': raw_transcription,
                    'language': language,
                    'context': context,
                    'model': 'gemini-2.0-flash-lite'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to enhance transcription: {e}")
            return GeminiResponse(
                success=False,
                content=raw_transcription,  # Return original text as fallback
                error=str(e),
                metadata={'original_text': raw_transcription}
            )
    
    def _build_enhancement_prompt(self, 
                                raw_text: str, 
                                context: str = None, 
                                language: str = "Arabic") -> str:
        """Build the prompt for text enhancement."""
        
        base_prompt = f"""
You are an expert in {language} language processing and lip reading transcription enhancement.

Your task is to enhance and correct a raw transcription obtained from a lip reading model. The transcription may contain:
- Missing diacritical marks (especially important for Arabic)
- Spelling errors
- Missing spaces between words
- Grammatical inconsistencies
- Character recognition errors
Provide the enhanced version with no text else and preserve diacritical marks.
Raw transcription: "{raw_text}"
"""
        
        if context:
            base_prompt += f"\nAdditional context: {context}"
        
        base_prompt += f"""

Please provide an enhanced version that:
1. Corrects spelling and grammar errors
2. Adds appropriate diacritical marks for Arabic text
3. Ensures proper word spacing
4. Maintains the original meaning and intent
5. Uses natural, fluent {language}

Enhanced transcription:"""
        
        return base_prompt
    
    def summarize_content(self, text: str, max_length: int = 200) -> GeminiResponse:
        """
        Generate a summary of the transcribed content.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            GeminiResponse with summary
        """
        try:
            prompt = f"""
Please provide a concise summary of the following Arabic text in approximately {max_length} words or less:

Text: "{text}"

Summary:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    **self.generation_config,
                    'max_output_tokens': max_length * 2  # Rough estimate for tokens
                }
            )
            
            summary = response.text.strip()
            
            return GeminiResponse(
                success=True,
                content=summary,
                metadata={
                    'original_text': text,
                    'max_length': max_length,
                    'model': 'gemini-2.0-flash-lite'
                }
            )
            
        except Exception as e:
            return GeminiResponse(
                success=False,
                content="",
                error=str(e),
                metadata={'original_text': text}
            )
    
    def translate_text(self, text: str, target_language: str = "English") -> GeminiResponse:
        """
        Translate the transcribed text to another language.
        
        Args:
            text: Text to translate
            target_language: Target language for translation
            
        Returns:
            GeminiResponse with translation
        """
        try:
            prompt = f"""
Please translate the following Arabic text to {target_language}. 
Maintain the original meaning and context as much as possible.

Arabic text: "{text}"

{target_language} translation:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            translation = response.text.strip()
            
            return GeminiResponse(
                success=True,
                content=translation,
                metadata={
                    'original_text': text,
                    'source_language': 'Arabic',
                    'target_language': target_language,
                    'model': 'gemini-2.0-flash-lite'
                }
            )
            
        except Exception as e:
            return GeminiResponse(
                success=False,
                content="",
                error=str(e),
                metadata={
                    'original_text': text,
                    'target_language': target_language
                }
            )
    
    def analyze_content(self, text: str) -> GeminiResponse:
        """
        Analyze the content for key topics, sentiment, and other insights.
        
        Args:
            text: Text to analyze
            
        Returns:
            GeminiResponse with analysis
        """
        try:
            prompt = f"""
Please analyze the following Arabic text and provide:
1. Main topics or themes
2. Sentiment (positive, negative, neutral)
3. Key entities (people, places, organizations)
4. Brief context or domain (if identifiable)

Text: "{text}"

Analysis:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            analysis = response.text.strip()
            
            return GeminiResponse(
                success=True,
                content=analysis,
                metadata={
                    'original_text': text,
                    'analysis_type': 'content_analysis',
                    'model': 'gemini-2.0-flash-lite'
                }
            )
            
        except Exception as e:
            return GeminiResponse(
                success=False,
                content="",
                error=str(e),
                metadata={'original_text': text}
            )
    
    def batch_process(self, 
                     texts, 
                     operation: str = "enhance", 
                     **kwargs):
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            operation: Type of operation ('enhance', 'summarize', 'translate', 'analyze')
            **kwargs: Additional arguments for the specific operation
            
        Returns:
            List of GeminiResponse objects
        """
        results = []
        operation_map = {
            'enhance': self.enhance_transcription,
            'summarize': self.summarize_content,
            'translate': self.translate_text,
            'analyze': self.analyze_content
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unsupported operation: {operation}")
        
        func = operation_map[operation]
        
        for text in texts:
            try:
                result = func(text, **kwargs)
                results.append(result)
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                results.append(GeminiResponse(
                    success=False,
                    content="",
                    error=str(e),
                    metadata={'original_text': text}
                ))
        
        return results
    
    def set_generation_config(self, **config):
        """Update the generation configuration."""
        self.generation_config.update(config)
    
    def get_available_models(self):
        """Get list of available Gemini models."""
        try:
            models = genai.list_models()
            return [model.name for model in models if 'gemini' in model.name.lower()]
        except Exception as e:
            return [f"Error fetching models: {str(e)}"]
