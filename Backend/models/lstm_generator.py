"""
LSTM Poetry Generator Module
Loads and uses the custom LSTM model from Hugging Face: kriteekathapa/lstm-poem-generator-v1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from typing import Optional, List, Dict
import json
import logging

logger = logging.getLogger(__name__)


class CharVocabulary:
    """
    Character-level vocabulary for the LSTM model.
    Handles encoding/decoding between characters and token indices.
    """
    
    def __init__(
        self,
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>"
    ):
        self.char_to_idx = char_to_idx
        self.idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        self.pad_idx = self.char_to_idx.get(pad_token, 0)
        self.unk_idx = self.char_to_idx.get(unk_token, 1)
        self.sos_idx = self.char_to_idx.get(sos_token, 2)
        self.eos_idx = self.char_to_idx.get(eos_token, 3)
        
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text to list of token indices"""
        tokens = []
        if add_special_tokens:
            tokens.append(self.sos_idx)
        
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_idx))
            
        if add_special_tokens:
            tokens.append(self.eos_idx)
            
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token indices back to text"""
        special_indices = {self.pad_idx, self.sos_idx, self.eos_idx}
        
        chars = []
        for idx in tokens:
            if skip_special_tokens and idx in special_indices:
                continue
            char = self.idx_to_char.get(idx, self.unk_token)
            if skip_special_tokens and char in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            chars.append(char)
            
        return "".join(chars)
    
    @classmethod
    def load(cls, filepath: str) -> "CharVocabulary":
        """Load vocabulary from a JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            char_to_idx=data.get("char_to_idx", data.get("stoi", {})),
            idx_to_char=data.get("idx_to_char", data.get("itos", {})),
            pad_token=data.get("pad_token", "<PAD>"),
            unk_token=data.get("unk_token", "<UNK>"),
            sos_token=data.get("sos_token", "<SOS>"),
            eos_token=data.get("eos_token", "<EOS>")
        )


class UnidirectionalLSTMPoetryModel(nn.Module):
    """
    Unidirectional LSTM model for poetry generation.
    Architecture matches the trained model on Hugging Face.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input token indices (batch_size, seq_len)
            hidden: Optional tuple of (h_n, c_n) hidden states
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            hidden: Updated hidden states
        """
        # Embed input
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        
        # LSTM forward
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        # Project to vocabulary
        logits = self.fc(output)  # (batch, seq, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: str = "cpu"):
        """Initialize hidden states with zeros"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)
    
    @torch.no_grad()
    def generate(
        self,
        start_tokens: List[int],
        vocab: CharVocabulary,
        max_length: int = 300,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        device: str = "cpu"
    ) -> str:
        """
        Generate text starting from given tokens.
        
        Args:
            start_tokens: List of starting token indices
            vocab: Vocabulary for decoding
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            device: Device to use
            
        Returns:
            Generated text string
        """
        self.eval()
        
        # Initialize
        generated = list(start_tokens)
        hidden = self.init_hidden(1, device)
        
        # Process starting tokens
        if len(start_tokens) > 0:
            input_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)
            _, hidden = self.forward(input_tensor, hidden)
        
        # Get last token for generation loop
        current_token = start_tokens[-1] if start_tokens else vocab.sos_idx
        
        # Generate tokens
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([[current_token]], dtype=torch.long, device=device)
            
            # Forward pass
            logits, hidden = self.forward(input_tensor, hidden)
            logits = logits[0, -1, :]  # Get last token logits
            
            # Apply temperature
            logits = logits / max(temperature, 0.1)
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_top_k = top_k_values[-1]
                logits = torch.where(logits < min_top_k, torch.tensor(float('-inf'), device=device), logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for EOS
            if next_token == vocab.eos_idx:
                break
                
            generated.append(next_token)
            current_token = next_token
        
        # Decode generated tokens
        return vocab.decode(generated, skip_special_tokens=True)


class LSTMGenerator:
    """
    High-level LSTM poetry generator that handles model loading from Hugging Face.
    """
    
    def __init__(
        self,
        model_name: str = "kriteekathapa/lstm-poem-generator-v1",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the LSTM generator.
        
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
        self.vocab = None
        self.config = None
        self._loaded = False
        
        logger.info(f"LSTMGenerator initialized (model={model_name}, device={self.device})")
    
    def load(self) -> None:
        """Load the model, vocabulary, and config from Hugging Face"""
        if self._loaded:
            logger.info("LSTM model already loaded")
            return
            
        logger.info(f"Loading LSTM model: {self.model_name}")
        
        try:
            # Download files from Hugging Face Hub
            vocab_path = hf_hub_download(
                repo_id=self.model_name,
                filename="vocab.json",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename="best_model.pt",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            config_path = hf_hub_download(
                repo_id=self.model_name,
                filename="config.json",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            # Load vocabulary
            self.vocab = CharVocabulary.load(vocab_path)
            logger.info(f"Vocabulary loaded: {self.vocab.vocab_size} tokens")
            
            # Load config
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            # Initialize model with config
            self.model = UnidirectionalLSTMPoetryModel(
                vocab_size=self.vocab.vocab_size,
                embedding_dim=self.config.get("embedding_dim", 256),
                hidden_dim=self.config.get("hidden_dim", 512),
                num_layers=self.config.get("num_layers", 3),
                dropout=0.0,  # No dropout for inference
                pad_idx=self.vocab.pad_idx
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info(f"LSTM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise RuntimeError(f"Failed to load LSTM model: {e}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 300,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        **kwargs  # Accept but ignore GPT-2 specific params
    ) -> str:
        """
        Generate poetry based on the given prompt.
        
        Args:
            prompt: Starting text for poem generation
            max_length: Maximum number of characters to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Number of top tokens to consider for sampling
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated poem text
        """
        if not self._loaded:
            self.load()
        
        # Format the prompt as used during training
        formatted_prompt = f"Title: {prompt}\n\n"
        
        # Encode prompt
        start_tokens = self.vocab.encode(formatted_prompt, add_special_tokens=False)
        
        # Generate
        generated_text = self.model.generate(
            start_tokens=start_tokens,
            vocab=self.vocab,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device
        )
        
        return generated_text
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._loaded
    
    def unload(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        self.vocab = None
        self.config = None
        self._loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("LSTM model unloaded")
