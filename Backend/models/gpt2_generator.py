"""
GPT-2 Poetry Generator Module
Loads and uses the fine-tuned GPT-2 model from Hugging Face
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def post_process_gpt2_output(text: str, max_chars: int = 55) -> str:
    """
    Post-process GPT-2 generated text by iteratively splitting long lines
    at punctuation boundaries.
    
    - Preserves all existing line breaks exactly.
    - Processes text line by line.
    - While a line exceeds the max length, inserts a line break at the nearest
      punctuation mark (. , ; ! ? : - --) BEFORE the threshold.
    - Continues processing remaining text until all lines are within threshold.
    - Does not merge lines.
    - Does not break words or incomplete phrases.
    - If no suitable punctuation is found before threshold, leaves the line unchanged.
    
    Args:
        text: The generated text to process
        max_chars: Maximum number of characters per line before splitting
    
    Returns:
        Processed text with additional line breaks where appropriate
    """
    # Split by original line breaks to preserve them exactly
    original_lines = text.split('\n')
    final_lines = []
    
    for line in original_lines:
        # If line is empty or within threshold, keep as-is
        if not line.strip() or len(line) <= max_chars:
            final_lines.append(line)
            continue
        
        # Line exceeds threshold - iteratively split at punctuation
        split_lines = _iterative_split_line(line, max_chars)
        final_lines.extend(split_lines)
    
    # Rejoin with original line break character
    return '\n'.join(final_lines)


def _iterative_split_line(line: str, max_chars: int) -> List[str]:
    """
    Iteratively split a long line at punctuation boundaries until all
    resulting segments are within the threshold.
    
    Args:
        line: The line to split
        max_chars: Maximum characters per segment
    
    Returns:
        List of line segments
    """
    result_lines: List[str] = []
    remaining = line
    
    while len(remaining) > max_chars:
        # Find the best punctuation break point BEFORE the threshold
        break_pos, skip_chars = _find_best_break_position(remaining, max_chars)
        
        if break_pos == -1:
            # No suitable punctuation found before threshold
            # Try to find ANY punctuation in the line
            break_pos, skip_chars = _find_any_punctuation_break(remaining)
            if break_pos == -1:
                # No punctuation at all, keep the line as-is
                result_lines.append(remaining)
                remaining = ""
                break
        
        # Split at the break position
        segment = remaining[:break_pos + 1].strip()
        # Clean trailing dashes from segment
        segment = segment.rstrip('-').strip()
        result_lines.append(segment)
        
        # Continue with the remaining text (skip punctuation/dashes and whitespace)
        remaining = remaining[break_pos + 1 + skip_chars:].lstrip('-').lstrip()
    
    # Add any remaining text
    if remaining:
        result_lines.append(remaining)
    
    return result_lines


def _find_best_break_position(text: str, max_chars: int) -> tuple:
    """
    Find the best punctuation position to break at, looking for the
    rightmost punctuation mark within the max_chars limit.
    
    Handles:
    - Standard punctuation: . , ; ! ? :
    - Punctuation followed by dashes: .- ,- ;-
    - Double dashes: --
    - Ellipsis: ...
    
    Args:
        text: The text to search
        max_chars: Maximum position to search up to
    
    Returns:
        Tuple of (position of break, additional chars to skip), or (-1, 0) if not found
    """
    punctuation_marks = '.!?;,:'
    search_range = min(max_chars, len(text))
    
    # Search backwards from the threshold for the best break point
    for i in range(search_range - 1, -1, -1):
        char = text[i]
        
        # Check for double dash --
        if char == '-' and i > 0 and text[i-1] == '-':
            return (i, 0)
        
        # Check for standard punctuation
        if char in punctuation_marks:
            next_pos = i + 1
            skip_chars = 0
            
            # Handle ellipsis (...)
            if char == '.' and i + 2 < len(text) and text[i+1] == '.' and text[i+2] == '.':
                next_pos = i + 3
                skip_chars = 2
            
            # Check what follows the punctuation
            if next_pos >= len(text):
                return (i + skip_chars, 0)
            
            next_char = text[next_pos]
            
            # Valid break: followed by space, dash, or capital letter
            if next_char.isspace():
                return (i + skip_chars, 0)
            elif next_char == '-':
                # Count consecutive dashes to skip
                dash_count = 0
                j = next_pos
                while j < len(text) and text[j] == '-':
                    dash_count += 1
                    j += 1
                return (i + skip_chars, dash_count)
            elif next_char.isupper():
                # Punctuation directly followed by capital letter (e.g., ".And")
                return (i + skip_chars, 0)
    
    return (-1, 0)


def _find_any_punctuation_break(text: str) -> tuple:
    """
    Find the first punctuation mark in the text that could serve as a break point.
    
    Args:
        text: The text to search
    
    Returns:
        Tuple of (position of break, additional chars to skip), or (-1, 0) if not found
    """
    punctuation_marks = '.!?;,:'
    
    for i, char in enumerate(text):
        # Check for double dash
        if char == '-' and i + 1 < len(text) and text[i + 1] == '-':
            return (i + 1, 0)
        
        if char in punctuation_marks:
            next_pos = i + 1
            skip_chars = 0
            
            # Handle ellipsis
            if char == '.' and i + 2 < len(text) and text[i+1] == '.' and text[i+2] == '.':
                next_pos = i + 3
                skip_chars = 2
            
            if next_pos >= len(text):
                return (i + skip_chars, 0)
            
            next_char = text[next_pos]
            
            if next_char.isspace():
                return (i + skip_chars, 0)
            elif next_char == '-':
                dash_count = 0
                j = next_pos
                while j < len(text) and text[j] == '-':
                    dash_count += 1
                    j += 1
                return (i + skip_chars, dash_count)
            elif next_char.isupper():
                return (i + skip_chars, 0)
    
    return (-1, 0)


class GPT2Generator:
    """
    GPT-2 based poetry generator using Hugging Face Transformers.
    Loads the fine-tuned model: kriteekathapa/gpt2-poems-finetuned-v1
    """
    
    def __init__(
        self,
        model_name: str = "kriteekathapa/gpt2-poems-finetuned-v1",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the GPT-2 generator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ("cuda", "cpu", or "auto")
            cache_dir: Optional cache directory for models
            hf_token: Optional Hugging Face API token
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        logger.info(f"GPT2Generator initialized (model={model_name}, device={self.device})")
    
    def load(self) -> None:
        """Load the model and tokenizer from Hugging Face"""
        if self._loaded:
            logger.info("GPT-2 model already loaded")
            return
            
        logger.info(f"Loading GPT-2 model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info(f"GPT-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            raise RuntimeError(f"Failed to load GPT-2 model: {e}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.80,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate poetry based on the given prompt.
        
        Args:
            prompt: Starting text for poem generation
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more creative)
            top_p: Cumulative probability for nucleus sampling
            top_k: Number of top tokens to consider for sampling
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of poems to generate
            
        Returns:
            Generated poem text
        """
        if not self._loaded:
            self.load()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Apply post-processing to format long lines
        generated_text = post_process_gpt2_output(generated_text)
        
        return generated_text
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._loaded
    
    def unload(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("GPT-2 model unloaded")
