"""
Configuration settings for the Poetry Generation Backend
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # Server settings
    PORT = int(os.environ.get("PORT", 5000))
    HOST = os.environ.get("HOST", "0.0.0.0")
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Development mode: fake responses for testing
    FAKE_MODE = os.environ.get("FAKE_MODE", "0") == "1"
    
    # Hugging Face model identifiers
    HF_GPT2_MODEL = os.environ.get("HF_GPT2_MODEL", "kriteekathapa/gpt2-poems-finetuned-v1")
    HF_LSTM_MODEL = os.environ.get("HF_LSTM_MODEL", "kriteekathapa/lstm-poem-generator-v1")
    
    # Optional HF token for private models
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    
    # Device configuration
    DEVICE = os.environ.get("DEVICE", "auto")
    
    # Model cache directory (optional)
    MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", None)
    
    @classmethod
    def get_device(cls):
        """Determine the best device for model inference"""
        import torch
        
        if cls.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.DEVICE
