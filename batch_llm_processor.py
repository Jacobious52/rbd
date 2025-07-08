import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchLLMProcessor:
    """
    Batch processor for LLM operations on voice clip transcripts.
    Consolidates correction, summarization, and classification into single calls.
    """
    
    def __init__(self, model_name: str = "gemma3:4b", max_batch_size: int = 10, timeout: int = 300):
        """
        Initialize the batch processor.
        
        Args:
            model_name: Ollama model to use for processing
            max_batch_size: Maximum number of transcripts per batch
            timeout: Timeout in seconds for LLM calls
        """
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        
    def create_batch_prompt(self, transcripts: List[Tuple[str, str]]) -> str:
        """
        Create a comprehensive batch processing prompt.
        
        Args:
            transcripts: List of (clip_id, transcript_text) tuples
            
        Returns:
            Formatted prompt string
        """
        # Build the transcript entries
        transcript_entries = []
        for clip_id, transcript_text in transcripts:
            transcript_entries.append(f"""
TRANSCRIPT_{clip_id}:
{transcript_text}
""")
        
        prompt = f"""You are a sleep talk analysis system processing surveillance camera audio transcripts. Process each transcript to:

1. CORRECT: Fix grammar, spelling, and transcription errors
2. SUMMARIZE: Create a 2-4 word filename-safe title  
3. CLASSIFY: Rate sleep talk likelihood (0.0-1.0)

GUIDELINES:
- Grammar correction: Fix obvious errors while preserving meaning
- Summaries: Use only alphanumeric characters, underscores, and hyphens (no spaces)
- Sleep talk scoring: Consider incoherence, fragmented speech, [inaudible] markers, and brevity
- Shorter, more incoherent text = higher sleep talk score
- Clear, coherent conversations = lower sleep talk score

TRANSCRIPTS TO PROCESS:
{''.join(transcript_entries)}

REQUIRED OUTPUT FORMAT (valid JSON only):
{{
  "results": [
    {{
      "clip_id": "TRANSCRIPT_1",
      "corrected_text": "corrected transcript text here",
      "summary": "Two_Three_Words",
      "sleep_talk_score": 0.85
    }},
    {{
      "clip_id": "TRANSCRIPT_2",
      "corrected_text": "another corrected transcript", 
      "summary": "Brief_Summary",
      "sleep_talk_score": 0.23
    }}
  ]
}}

IMPORTANT: Return ONLY the JSON object. No additional text, explanations, or formatting."""

        return prompt
    
    def parse_batch_response(self, response_text: str, expected_count: int) -> List[Dict]:
        """
        Parse and validate the batch LLM response.
        
        Args:
            response_text: Raw response from LLM
            expected_count: Expected number of results
            
        Returns:
            List of validated processing results
            
        Raises:
            ValueError: If response is invalid or unparseable
        """
        try:
            response_data = json.loads(response_text.strip())
            
            if not isinstance(response_data, dict) or 'results' not in response_data:
                raise ValueError("Invalid response structure - missing 'results' key")
                
            results = response_data['results']
            
            if len(results) != expected_count:
                logger.warning(f"Expected {expected_count} results, got {len(results)}")
            
            processed_results = []
            for i, result_item in enumerate(results):
                # Validate required fields
                required_fields = ['clip_id', 'corrected_text', 'summary', 'sleep_talk_score']
                missing_fields = [field for field in required_fields if field not in result_item]
                
                if missing_fields:
                    logger.warning(f"Result {i} missing fields: {missing_fields}")
                    continue
                
                # Validate and clamp sleep talk score
                try:
                    score = float(result_item['sleep_talk_score'])
                    result_item['sleep_talk_score'] = max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid sleep talk score for result {i}, using 0.0")
                    result_item['sleep_talk_score'] = 0.0
                
                # Sanitize summary for filename use
                summary = result_item.get('summary', '')
                sanitized_summary = "".join(c for c in summary if c.isalnum() or c in ("-", "_"))
                
                # Ensure summary is 2-4 words and valid
                if not sanitized_summary:
                    sanitized_summary = f"clip_{i+1:03d}"
                
                # Validate word count (split by underscores)
                word_count = len(sanitized_summary.split('_'))
                if word_count < 2 or word_count > 4:
                    logger.warning(f"Summary '{sanitized_summary}' doesn't meet 2-4 word requirement")
                    
                result_item['summary'] = sanitized_summary
                processed_results.append(result_item)
            
            return processed_results
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def process_batch(self, transcript_files: List[Path]) -> List[Dict]:
        """
        Process multiple transcript files in a single LLM call.
        
        Args:
            transcript_files: List of paths to transcript files
            
        Returns:
            List of processing results with corrections, summaries, and scores
            
        Raises:
            Exception: If batch processing fails
        """
        if not transcript_files:
            return []
        
        # Read all transcripts
        transcripts = []
        for i, transcript_file in enumerate(transcript_files):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty transcripts
                        transcripts.append((f"{i+1:03d}", content))
            except Exception as e:
                logger.error(f"Error reading {transcript_file}: {e}")
                continue
        
        if not transcripts:
            logger.warning("No valid transcripts to process")
            return []
        
        # Create batch prompt
        prompt = self.create_batch_prompt(transcripts)
        
        logger.info(f"Processing batch of {len(transcripts)} transcripts")
        
        try:
            # Call Ollama with batch prompt
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            response_text = result.stdout.strip()
            
            # Parse and validate response
            processed_results = self.parse_batch_response(response_text, len(transcripts))
            
            logger.info(f"Successfully processed {len(processed_results)} transcripts")
            return processed_results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama execution error: {e}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr.decode()}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"Batch processing timeout after {self.timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch processing: {e}")
            raise
    
    def process_with_fallback(self, transcript_files: List[Path], fallback_functions: Dict) -> List[Dict]:
        """
        Process transcripts with fallback to individual processing.
        
        Args:
            transcript_files: List of transcript file paths
            fallback_functions: Dict with 'correct', 'summarize', 'classify' functions
            
        Returns:
            List of processing results
        """
        if not transcript_files:
            return []
        
        # Split into batches
        all_results = []
        
        for i in range(0, len(transcript_files), self.max_batch_size):
            batch = transcript_files[i:i+self.max_batch_size]
            
            try:
                # Try batch processing first
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"Batch processing failed for batch {i//self.max_batch_size + 1}: {e}")
                logger.info("Falling back to individual processing...")
                
                # Fallback to individual processing
                for j, transcript_file in enumerate(batch):
                    try:
                        # Use original individual functions as fallback
                        fallback_functions['correct'](transcript_file)
                        summary = fallback_functions['summarize'](transcript_file)
                        sleep_score = fallback_functions['classify'](transcript_file)
                        
                        # Read corrected transcript
                        with open(transcript_file, 'r', encoding='utf-8') as f:
                            corrected_text = f.read().strip()
                        
                        individual_result = {
                            'clip_id': f"{i+j+1:03d}",
                            'corrected_text': corrected_text,
                            'summary': summary,
                            'sleep_talk_score': sleep_score
                        }
                        all_results.append(individual_result)
                        logger.info(f"Individual processing successful for {transcript_file.name}")
                        
                    except Exception as individual_error:
                        logger.error(f"Individual processing failed for {transcript_file}: {individual_error}")
                        continue
        
        return all_results