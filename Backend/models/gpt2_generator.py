"""
GPT-2 Poetry Generator Module
Loads and uses the fine-tuned GPT-2 model from Hugging Face
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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
